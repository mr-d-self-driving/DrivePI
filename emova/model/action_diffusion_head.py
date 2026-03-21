import copy
import math

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from diffusers.schedulers import DDIMScheduler
from .occ.occ_loss import MyFocalLoss

from .transformer_decoder import TransformerDecoderLayer, gen_sineembed_for_position
from .fusion_modules import CrossAttentionFusion, GatedFusion, FiLMFusion, FiLMFusionv2, DualPathFusion, NoneFusion


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class ConvFuser(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ConvFuser, self).__init__()
        self.fuser = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, out_channels, 3, padding=1, bias=False),
            nn.LayerNorm(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x1, x2):
        assert x2.size() == x2.size()
        x = torch.cat([x1, x2], dim=1)
        y = self.fuser(x)
        return y


class ModulationLayer(nn.Module):
    def __init__(
            self,
            embed_dims: int = 256,
            if_global_cond: bool = False,
            if_zeroinit_scale: bool = True,
    ):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale = if_zeroinit_scale
        self.if_global_cond = if_global_cond
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dims, embed_dims * 2) if not if_global_cond else nn.Linear(embed_dims * 2, embed_dims * 2),
        )

    def init_weight(self):
        # Zero initialize the last layer of scale_shift_mlp
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
            self,
            traj_feature,
            time_embed,
            global_cond=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                global_cond, time_embed
            ], axis=-1)
        else:
            global_feature = time_embed
        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffAnchorPlannerHead(nn.Module):
    def __init__(
            self,
            hidden_size,
            seq=False,
            use_ego=False,
            planning_anchor=None,
            rescore_traj=False,
            in_channels=128 * 3,
            hidden_channel=128,
            # config for Transformer
            num_decoder_layers=3,
            decoder_layer=dict(),
            num_heads=8,
            bn_momentum=0.1,
            bias='auto',
            feature_proj=True,
            # loss
            loss_plan_cls=dict(),
            loss_plan_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
            loss_ego_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
            # others
            planning_config=None,
            trainable=True,
            fusion_type='none'
    ):
        super(DiffAnchorPlannerHead, self).__init__()

        self.use_ego = False
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_channel = hidden_channel
        self.bn_momentum = bn_momentum
        self.fut_steps = 6
        self.ego_fut_mode = 3
        self.seq = seq
        self.rescore_traj = rescore_traj
        self.planning_config = planning_config
        self.fusion_type = fusion_type
        self.feature_proj = feature_proj
        self.hidden_size = hidden_size

        self.use_text_feats = True
        print('### self.fusion_type:', self.fusion_type)
        print("### self.use_ego:", self.use_ego)
        if self.use_text_feats:
            # Select fusion module
            if fusion_type == 'film':
                self.fusion_module = FiLMFusionv2(hidden_size)
            elif fusion_type == 'gated':
                self.fusion_module = GatedFusion(hidden_size)
            elif fusion_type == 'dual_path':
                self.fusion_module = DualPathFusion(hidden_size)
            elif fusion_type == 'cross_att':
                self.fusion_module = CrossAttentionFusion(hidden_size)
            else:
                self.fusion_module = NoneFusion(hidden_size)

        # Projection from LLM features to BEV features
        if self.feature_proj:
            self.feature_projector = nn.Linear(hidden_size, 384 * 3 * 3)

        self.action_head_conv = nn.Sequential(
            nn.Conv2d(384, hidden_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.loss_plan_cls = MyFocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=5.0,
                                         activated=False)

        self.plan_decoder = TransformerDecoderLayer(
            hidden_channel,
            num_heads,
            1024,
        )

        self.decoder = TransformerDecoderLayer(
            hidden_channel,
            num_heads,
            1024,
        )

        planning_anchor = np.load(planning_anchor)
        self.planning_anchor = nn.Parameter(
            torch.tensor(planning_anchor, dtype=torch.float32),
            requires_grad=False,
        )

        # Prediction Head
        ego_fut_decoder = list()
        for i in range(2):
            ego_fut_decoder.append(nn.Linear(self.hidden_channel, self.hidden_channel))
            ego_fut_decoder.append(nn.ReLU())

        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        self.reg = nn.Linear(self.hidden_channel, self.fut_steps * 2)
        self.cls = nn.Linear(self.hidden_channel, 1)

        ego = list()
        for i in range(2):
            ego.append(nn.Linear(self.hidden_channel, self.hidden_channel))
            ego.append(nn.ReLU())
        ego.append(nn.Linear(self.hidden_channel, 10))
        self.ego = nn.Sequential(*ego)

        x_size = self.planning_config["grid_size"][0] // self.planning_config["out_size_factor"]
        y_size = self.planning_config["grid_size"][1] // self.planning_config["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        kernel_size = tuple([int(x / 2) for x in [180, 180]])
        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size),
        )

        self.ego_anchor = nn.Parameter(
            torch.tensor([[0, 0.5, -1.84 + 1.56 / 2, np.log(4.08), np.log(1.73), np.log(1.56), 1, 0, 0, 0, 0], ],
                         dtype=torch.float32),
            requires_grad=False,
        )

        self.ego_anchor_encoder = nn.Sequential(
            nn.Linear(11, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.ego_encoder = nn.Sequential(
            nn.Linear(10, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel, bias=False),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.plan_pos_encoder = nn.Sequential(
            nn.Linear(768, hidden_channel, bias=False),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        self.time_pos = SinusoidalPosEmb(hidden_channel)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel * 4),
            nn.Mish(),
            nn.Linear(hidden_channel * 4, hidden_channel),
        )

        self.time_mod = ModulationLayer(embed_dims=hidden_channel)

        self.init_weights()
        self.reset()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def reset(self):
        self.prev_token = list()
        self.prev_ego = list()
        self.prev_ego_features = list()
        self.prev_timestamps = list()

    def update(self, i, token, timestamps, ego_status, ego_features):
        self.prev_token[i] = token
        self.prev_ego[i] = copy.deepcopy(ego_status.detach())
        self.prev_ego_features[i] = copy.deepcopy(ego_features.detach())
        if timestamps is not None:
            self.prev_timestamps[i] = timestamps

    def get(self, i):
        if len(self.prev_token) <= i:
            self.prev_token.append(None)
            self.prev_ego.append(None)
            self.prev_ego_features.append(None)
            self.prev_timestamps.append(None)
        return self.prev_token[i], self.prev_timestamps[i], self.prev_ego[i], self.prev_ego_features[i]

    def forward_temporal(self, ego_status, ego_features, img_metas):
        start_flags = list()
        prev_ego_features_list = list()
        ego_anchor = ego_status.clone()

        for i, meta in enumerate(img_metas):
            scene_token = meta['scene_token']
            timestamp = meta['timestamp']
            prev_token, prev_timestamp, prev_ego, prev_ego_features = self.get(i)

            if prev_token is None:
                is_start = True
            else:
                is_start = prev_token != scene_token
                is_start |= 2 < (timestamp - prev_timestamp)
            start_flags.append(is_start)

            if not is_start:
                ego_anchor[i, -2] = prev_ego[6]
                prev_ego_features_list.append(torch.zeros_like(ego_features[i]))
            else:
                prev_ego_features_list.append(torch.zeros_like(ego_features[i]))

        return start_flags, ego_features, ego_anchor

    def store_temporal(self, ego_status, ego_features, start_flags, img_metas):
        for i in range(len(ego_status)):
            scene_token = img_metas[i]['scene_token']
            if start_flags[i]:
                timestamp = img_metas[i]['timestamp']
            else:
                timestamp = None

            self.update(i, scene_token, timestamp, ego_status[i], ego_features[i])

    def normalize_ego_fut_trajs(self, gt_ego_fut_trajs):
        odo_info_fut_x = gt_ego_fut_trajs[..., 0:1]
        odo_info_fut_y = gt_ego_fut_trajs[..., 1:2]

        odo_info_fut_x = odo_info_fut_x / 3
        odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y + 0.5) / 8.1
        odo_info_fut_y = odo_info_fut_y.clamp(0, 1)
        odo_info_fut_y = odo_info_fut_y * 2 - 1
        odo_info_fut = torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        return odo_info_fut

    def denormalize_ego_fut_trajs(self, noisy_traj_points):
        odo_info_fut_x = noisy_traj_points[..., 0:1]
        odo_info_fut_y = noisy_traj_points[..., 1:2]

        odo_info_fut_x = odo_info_fut_x * 3
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * 8.1 - 0.5
        odo_info_fut = torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        return odo_info_fut

    def forward_single(self, hidden_states, ego_status, gt_ego_fut_cmd, gt_trajs, ori_bev_feats=None,
                       return_results=False, text_features=None, img_metas=None):
        input_dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        if (text_features is not None) and self.use_text_feats:
            # Use selected fusion module for feature fusion
            hidden_states = self.fusion_module(hidden_states, text_features)

        occ_bev_feats = self.feature_projector(hidden_states).reshape(-1, 3600, 9, 384).reshape(-1, 180 * 180, 384)

        feats = occ_bev_feats.reshape(batch_size, 180, 180, -1).permute(0, 3, 1, 2)
        feats = self.action_head_conv(feats)

        bev_pos = self.bev_pos.repeat(feats.shape[0], 1, 1).to(feats.device)

        gt_ego_fut_cmd_feat = gt_ego_fut_cmd.to(feats.dtype).unsqueeze(-1)
        gt_ego_fut_cmd_feat = rearrange(gt_ego_fut_cmd_feat, 'b c l -> b l c')
        ego_anchor = self.ego_anchor.repeat(feats.shape[0], 1).to(input_dtype)

        ego_query = self.ego_feature_encoder(feats)
        ego_query = ego_query.unsqueeze(1).squeeze(-1).squeeze(-1)
        ego_query = self.query_encoder(ego_query)

        feats = feats.flatten(-2)
        ego_query = self.decoder(rearrange(ego_query, 'b l c -> b c l'), feats, None, bev_pos)
        ego_query = rearrange(ego_query, 'b c l -> b l c')

        ego_anchor_embed = self.ego_anchor_encoder(ego_anchor)
        ego_anchor_embed = ego_anchor_embed.unsqueeze(1)
        ego_status_query = ego_query + ego_anchor_embed
        outputs_ego_status = self.ego(ego_status_query)

        bs_indices = torch.arange(len(feats), device=feats.device)
        cmd = gt_ego_fut_cmd.argmax(dim=-1)
        cmd_plan_anchor = self.planning_anchor.unsqueeze(0).repeat(len(feats), 1, 1, 1, 1)[bs_indices, cmd]
        zeros_cat = torch.zeros(len(feats), 6, 1, 2, device=feats.device)
        cmd_plan_anchor = torch.cat([zeros_cat, cmd_plan_anchor], dim=2)
        tgt_cmd_plan_anchor = cmd_plan_anchor[:, :, 1:, :] - cmd_plan_anchor[:, :, :-1, :]
        odo_info_fut = self.normalize_ego_fut_trajs(tgt_cmd_plan_anchor)
        traj_mode = odo_info_fut.shape[1]
        odo_info_fut = odo_info_fut.view(len(feats) * traj_mode, self.fut_steps, 2)

        # magic number 40 means that we add little noise for each anchor
        timesteps = torch.randint(
            0, 40,
            (len(feats),), device=feats.device
        )

        repeat_timesteps = timesteps.repeat_interleave(traj_mode)
        noise = torch.randn(odo_info_fut.shape, device=feats.device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=repeat_timesteps,
        ).to(input_dtype)
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denormalize_ego_fut_trajs(noisy_traj_points)

        diff_plan_reg = noisy_traj_points
        traj_pos_embed = gen_sineembed_for_position(diff_plan_reg, hidden_dim=128)
        traj_pos_embed = traj_pos_embed.flatten(-2).to(input_dtype)
        traj_feature = self.plan_pos_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(len(feats), traj_mode, -1)

        time_pos = self.time_pos(repeat_timesteps.to(torch.float32)).to(input_dtype)
        time_embed = self.time_mlp(time_pos)
        time_embed = time_embed.view(len(feats), traj_mode, -1)
        traj_feature = self.time_mod(traj_feature, time_embed)
        traj_feature = self.plan_decoder(rearrange(traj_feature, 'b l c -> b c l'), feats, None, bev_pos)
        traj_feature = rearrange(traj_feature, 'b c l -> b l c')
        
        if self.use_ego:
            ego_status_feat = self.ego_encoder(ego_status.type_as(outputs_ego_status)).unsqueeze(1)
        else:
            ego_status_feat = self.ego_encoder(outputs_ego_status.squeeze(1)).unsqueeze(1)
        traj_feature = traj_feature + ego_status_feat

        traj_feature = self.ego_fut_decoder(traj_feature)
        outputs_ego_trajs = self.reg(traj_feature)
        outputs_ego_trajs_cls = self.cls(traj_feature)
        outputs_ego_trajs = outputs_ego_trajs.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        outputs_ego_trajs = rearrange(outputs_ego_trajs, 'b m k (n c) -> b m k n c', k=self.planning_anchor.shape[1],
                                      m=self.ego_fut_mode, n=self.fut_steps)
        outputs_ego_trajs_cls = outputs_ego_trajs_cls.unsqueeze(1).squeeze(-1).repeat(1, self.ego_fut_mode, 1)
        outputs_ego_trajs_cls = rearrange(outputs_ego_trajs_cls, 'b m k -> b m k', k=self.planning_anchor.shape[1],
                                          m=self.ego_fut_mode)

        res = {'ego_fut_preds': outputs_ego_trajs, 'ego_fut_preds_cls': outputs_ego_trajs_cls,
               'ego_status_preds': outputs_ego_status, 'ego_fut_cmd': gt_ego_fut_cmd}
        return [res]

    def forward(self, feats, ego_status, gt_ego_fut_cmd, gt_trajs=None, gt_ego_fut_masks=None, ori_bev_feats=None,
                return_results=False, text_features=None, img_metas=None):
        res = [self.forward_single(feats, ego_status, gt_ego_fut_cmd, gt_trajs, ori_bev_feats, return_results,
                                   text_features,
                                   img_metas)]
        output_dict = {}
        if self.training and gt_trajs is not None:
            loss = self.loss(res, img_metas, gt_trajs, gt_ego_fut_masks, gt_ego_fut_cmd, ego_status)
            output_dict['losses'] = loss

        if return_results:
            results_planning = self.get_bboxes(res)
            planning_results = [
                self.planning3d2result(pts_plan_reg)
                for pts_plan_reg in results_planning
            ]
            output_dict['results'] = planning_results[0]

        return output_dict

    def planning3d2result(self, plan_reg):
        result_dict = dict(
            plan_reg=plan_reg.cpu().numpy())

        return result_dict

    def get_best_reg(self, pred_trajs, pred_trajs_cls, gt_trajs, ego_fut_cmd, reg_weight):
        batch_size, num_pred, mode, ts, d = pred_trajs.shape
        bs_indices = torch.arange(batch_size, device=pred_trajs.device)
        cmd = ego_fut_cmd.argmax(dim=-1)
        pred_trajs = pred_trajs[bs_indices, cmd].unsqueeze(1)
        best_cls = pred_trajs_cls[bs_indices, cmd].unsqueeze(1)

        reg_preds_cum = pred_trajs.cumsum(dim=-2)
        reg_gt_cum = gt_trajs.cumsum(dim=-2)
        dist = torch.linalg.norm(reg_gt_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
        dist = dist * reg_weight

        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        target_cls = mode_idx

        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
        best_reg = torch.gather(pred_trajs, 2, mode_idx).squeeze(1)
        return best_cls, best_reg, target_cls

    def loss(self, preds_dicts, metas, gt_ego_fut_trajs, gt_ego_fut_masks, gt_ego_fut_cmd, gt_ego_status):
        ego_fut_preds = preds_dicts[0][0]["ego_fut_preds"]
        ego_fut_cmd = preds_dicts[0][0]["ego_fut_cmd"]
        ego_fut_preds_cls = preds_dicts[0][0]["ego_fut_preds_cls"]
        ego_status_preds = preds_dicts[0][0]['ego_status_preds']

        gt_ego_fut_trajs = gt_ego_fut_trajs.unsqueeze(1)

        loss_plan_l1_weight = gt_ego_fut_masks[:, None, None, :]

        cls_pred, reg_pred, cls_target = self.get_best_reg(ego_fut_preds, ego_fut_preds_cls, gt_ego_fut_trajs,
                                                           ego_fut_cmd, loss_plan_l1_weight)

        # Calculate L1 loss with manual weighting
        element_wise_loss = torch.abs(reg_pred.reshape(-1, self.fut_steps * 2) - gt_ego_fut_trajs.reshape(-1, self.fut_steps * 2))
        # Apply weights
        weight = loss_plan_l1_weight.repeat(1, 1, 1, reg_pred.shape[-1]).reshape(-1, self.fut_steps * 2)
        loss_plan_reg = (element_wise_loss * weight).sum() / (weight.sum() + 1.0)  # Avoid division by zero

        loss_plan_cls_weight = loss_plan_l1_weight.squeeze(-1).any(dim=-1)
        loss_plan_cls = self.loss_plan_cls(cls_pred.flatten(end_dim=1), cls_target.flatten(end_dim=1),
                                           loss_plan_cls_weight.flatten(end_dim=1))

        loss_ego_reg = torch.abs(ego_status_preds.reshape(-1, 10) - gt_ego_status.reshape(-1, 10)).mean()

        action_loss = loss_plan_reg + loss_plan_cls + loss_ego_reg

        return action_loss

    def get_bboxes(self, preds_dicts):
        ego_fut_preds = preds_dicts[0][0]['ego_fut_preds']
        ego_fut_cmd = preds_dicts[0][0]['ego_fut_cmd']
        ego_fut_preds_cls = preds_dicts[0][0]["ego_fut_preds_cls"].sigmoid()
        select = ego_fut_cmd.argmax(dim=-1)

        ret_layer = []
        for i in range(len(ego_fut_preds)):
            plan_select = select[i]
            plan_reg = ego_fut_preds[i].cumsum(dim=-2)
            if self.rescore_traj:
                plan_cls = ego_fut_preds_cls[i]
            else:
                plan_cls = ego_fut_preds_cls[i][plan_select]
            mode_idx = plan_cls.argmax(dim=-1)
            ret_layer.append(plan_reg[plan_select][mode_idx])

        assert len(ret_layer) == 1

        return ret_layer

    def get_yaw(self, traj, start_yaw=0, static_dis_thresh=0.5):
        yaw = traj.new_zeros(traj.shape[:-1])
        yaw[..., 1:-1] = torch.atan2(
            traj[..., 2:, 1] - traj[..., :-2, 1],
            traj[..., 2:, 0] - traj[..., :-2, 0],
        )
        yaw[..., -1] = torch.atan2(
            traj[..., -1, 1] - traj[..., -2, 1],
            traj[..., -1, 0] - traj[..., -2, 0],
        )
        yaw[..., 0] = start_yaw
        # for static object, estimated future yaw would be unstable
        start = traj[..., 0, :]
        end = traj[..., -1, :]
        dist = torch.linalg.norm(end - start, dim=-1)
        mask = dist < static_dis_thresh
        start_yaw = yaw[..., 0].unsqueeze(-1)
        yaw = torch.where(
            mask.unsqueeze(-1),
            start_yaw,
            yaw,
        )
        return yaw.unsqueeze(-1)


class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out