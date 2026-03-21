import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast

from .occ.bev_grid_transform import BEVGridTransform
from .occ.occ_loss import lovasz_softmax, geo_scal_loss, sem_scal_loss, CustomFocalLoss
from .fusion_modules import CrossAttentionFusion, GatedFusion, FiLMFusion,FiLMFusionv2, BilinearFusion, DualPathFusion, NoneFusion

nusc_class_frequencies_dict = dict(
    others=944004,
    barrier=1897170,
    bicycle=152386,
    bus=2391677,
    car=16957802,
    construction_vehicle=724139,
    motorcycle=189027,
    pedestrian=2074468,
    traffic_cone=413451,
    trailer=2384460,
    truck=5916653,
    driveable_surface=175883646,
    other_flat=4275424,
    sidewalk=51393615,
    terrain=61411620,
    manmade=105975596,
    vegetation=116424404,
    free=1892500630
)

flow_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian',
]


class OCCHead(nn.Module):
    def __init__(self,
                 hidden_size,
                 in_channels=2048,
                 hidden_channel=256,
                 Dz=16,
                 flow=False,
                 bev_occ_config=None,
                 use_mask=True,
                 class_names=None,
                 num_classes=18,
                 size=None,
                 class_balance=False,
                 loss_occ=None,
                 loss_occ_flow=None,
                 feature_proj=True,
                 trainable=True,
                 fusion_type='none'): 
        super(OCCHead, self).__init__()

        bev_occ_config = dict(input_scope=[[-54.0, 54.0, 0.6], [-54.0, 54.0, 0.6]],
                              output_scope=[[-40, 40, 0.4], [-40, 40, 0.4]])
        # loss_occ = dict(use_sigmoid=True,
        #     loss_weight=1.0)
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.Dz = Dz
        self.flow = flow
        self.use_mask = use_mask
        self.class_names = class_names if class_names is not None else [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
            'manmade', 'vegetation', 'free'
        ]
        self.num_classes = num_classes
        self.size = size if size is not None else [200, 200]
        self.class_balance = class_balance
        self.feature_proj = feature_proj
        self.trainable = trainable
        self.fusion_type = fusion_type

        self.use_text_feats = True
        print('### self.fusion_type:', self.fusion_type)
        if self.use_text_feats:
            if fusion_type == 'film':
                self.fusion_module = FiLMFusionv2(hidden_size) #FiLMFusion(hidden_size)
            elif fusion_type == 'gated':
                self.fusion_module = GatedFusion(hidden_size)
            # elif fusion_type == 'bilinear':
            #     self.fusion_module = BilinearFusion(hidden_size)
            elif fusion_type == 'dual_path':
                self.fusion_module = DualPathFusion(hidden_size)
            elif fusion_type == 'cross_att':
                self.fusion_module = CrossAttentionFusion(hidden_size)
            else:
                self.fusion_module = NoneFusion(hidden_size)


        if self.feature_proj:
            self.feature_projector = nn.Linear(hidden_size, 384 * 3 * 3)

        self.final_conv = nn.Sequential(
            nn.Conv2d(384, hidden_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.predicter = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel * 2),
            nn.Softplus(),
            nn.Linear(hidden_channel * 2, num_classes * Dz),
        )

        if self.flow:
            self.flow_predicter = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel * 2),
                nn.Linear(hidden_channel * 2, 2 * Dz),
            )
            self.flow_index = []
            for n in flow_class_names:
                if n in self.class_names:
                    self.flow_index.append(self.class_names.index(n))

        # BEV网格变换
        self.transform = None
        if bev_occ_config is not None:
            self.transform = BEVGridTransform(**bev_occ_config)

        # 类别平衡权重
        if self.class_balance:
            nusc_class_frequencies = []
            for n in self.class_names:
                if n in nusc_class_frequencies_dict:
                    nusc_class_frequencies.append(nusc_class_frequencies_dict[n])
                else:
                    nusc_class_frequencies.append(1)
            nusc_class_frequencies = np.array(nusc_class_frequencies)
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
            self.register_buffer('cls_weights', class_weights)
        else:
            self.register_buffer('cls_weights', torch.ones(self.num_classes))

        # 损失函数
        self.loss_occ = CustomFocalLoss(use_sigmoid=True, loss_weight=1.0)

        if self.flow:
            self.loss_occ_flow = nn.L1Loss(reduction='none')

        # 如果不可训练，冻结所有参数
        if not self.trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, hidden_states, image_sizes=None, metas=None, gt_occ=None, mask_camera=None, gt_occ_flow=None,
                ori_bev_feats=None, return_results=False, text_features=None):
        """
        前向传播
        """

        input_dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        if (text_features is not None) and self.use_text_feats:
            # 使用选择的融合模块进行特征融合
            hidden_states = self.fusion_module(hidden_states, text_features)

            # # 原始的融合方法（作为备选）
            # text_features = text_features.mean(dim=1, keepdim=True)
            # scale_c = self.scale_c(text_features)
            # shift_c = self.shift_c(text_features)
            # hidden_states = (1.0 + scale_c.view(batch_size, 1, self.hidden_size)) * hidden_states + shift_c

        occ_bev_feats = self.feature_projector(hidden_states).reshape(-1, 3600, 9, 384).reshape(-1, 180 * 180, 384)

        features = occ_bev_feats.reshape(batch_size, 180, 180, -1).permute(0, 3, 1, 2)

        if self.transform is not None:
            features = self.transform(features.to(torch.float32)).to(occ_bev_feats.dtype)

        H, W = self.size
        occ_feat = self.final_conv(features).permute(0, 3, 2, 1)  # [B, Dx, Dy, hidden_channel]

        # 预测占用网格
        occ_pred = self.predicter(occ_feat)  # [B, Dx, Dy, Dz*num_classes]
        occ_pred = occ_pred.view(batch_size, H, W, self.Dz, self.num_classes)  # [B, Dx, Dy, Dz, num_classes]

        occ_flow_pred = None
        if self.flow:
            occ_flow_pred = self.flow_predicter(occ_feat)  # [B, Dx, Dy, Dz*2]
            occ_flow_pred = occ_flow_pred.view(batch_size, H, W, self.Dz, 2)  # [B, Dx, Dy, Dz, 2]

        # 计算损失
        losses = {}
        if gt_occ is not None:
            # 确保gt_occ和occ_pred在同一设备上
            gt_occ = gt_occ.to(occ_pred.device)

            # 计算占用网格损失
            preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()  # [B, num_classes, Dx, Dy, Dz]
            voxel_semantics = gt_occ.long()

            # 转换为float32进行损失计算
            preds_full = preds.to(torch.float32)

            # 计算CrossEntropy损失
            loss_occ = self.loss_occ(
                preds_full,
                voxel_semantics,
                weight=self.cls_weights.to(preds_full),
            ) * 100.0

            occ_weight = 1.0
            losses['occ_loss_occ'] = loss_occ.to(torch.float32) * occ_weight

            sem_loss = sem_scal_loss(preds_full, voxel_semantics)
            losses['occ_loss_voxel_sem_scal'] = sem_loss.to(torch.float32) * occ_weight

            geo_loss = geo_scal_loss(preds_full, voxel_semantics, non_empty_idx=self.num_classes - 1)
            losses['occ_loss_voxel_geo_scal'] = geo_loss.to(torch.float32) * occ_weight

            lovasz_loss = lovasz_softmax(torch.softmax(preds_full, dim=1), voxel_semantics)
            losses['occ_loss_voxel_lovasz'] = lovasz_loss.to(torch.float32) * occ_weight

            # 如果需要预测流场且提供了流场真值
            if self.flow and gt_occ_flow is not None and occ_flow_pred is not None:
                # 确保gt_occ_flow和flow_pred在同一设备上
                gt_occ_flow = gt_occ_flow.to(occ_flow_pred.device)

                # 转换为float32进行损失计算
                occ_flow_pred_full = occ_flow_pred.to(torch.float32)
                gt_occ_flow_full = gt_occ_flow.to(torch.float32)

                # 计算流场损失
                occ_flow_pred_flat = occ_flow_pred_full.reshape(-1, 2)
                gt_occ_flow_flat = gt_occ_flow_full.reshape(-1, 2)

                # 只对特定类别计算流场损失
                visible_mask = torch.zeros_like(occ_flow_pred_flat[:, 0], dtype=torch.bool)
                occ_label = voxel_semantics.reshape(-1)

                for index in self.flow_index:
                    visible_mask |= (occ_label == index)

                if visible_mask.sum() > 0:
                    occ_flow_pred_vis = occ_flow_pred_flat[visible_mask]
                    gt_occ_flow_vis = gt_occ_flow_flat[visible_mask]

                    # 计算权重
                    gt_occ_weight = torch.norm(gt_occ_flow_vis, dim=-1, keepdim=True)
                    gt_occ_weight[gt_occ_weight == 0] = 0.01

                    loss_flow = torch.abs(occ_flow_pred_vis - gt_occ_flow_vis)
                    loss_flow = (loss_flow * gt_occ_weight).mean()

                else:
                    loss_flow = occ_flow_pred_flat.sum() * 0.0
                losses['occ_loss_flow'] = loss_flow.to(torch.float32) * 0.5

        if return_results:
            results_occ = self.get_bboxes(occ_pred, occ_flow_pred)
            occ_results = [
                self.occ3d2result(occ)
                for occ in results_occ
            ]
            for pts_occ in occ_results:
                occ_results_tmp = pts_occ

            return {
                'occ_pred': occ_pred,
                'flow_pred': occ_flow_pred,
                'losses': losses,
                'occ_results': occ_results_tmp
            }
        else:
            return {
                'occ_pred': occ_pred,
                'flow_pred': occ_flow_pred,
                'losses': losses
            }

    def get_bboxes(self, occ_pred, occ_flow_pred=None):
        """
        获取预测结果
        """
        occ_score = occ_pred.softmax(-1)  # [B, Dx, Dy, Dz, num_classes]
        occ_res = occ_score.argmax(-1)  # [B, Dx, Dy, Dz]

        ret_layer = []
        for i in range(len(occ_res)):
            occ = occ_res[i]
            if self.flow and occ_flow_pred is not None:
                occ_flow = occ_flow_pred[i]
                ret = dict(occ=occ, flow=occ_flow)
            else:
                ret = dict(occ=occ)
            ret_layer.append(ret)

        return ret_layer

    def occ3d2result(self, occ):
        result_dict = dict(
            occ=occ['occ'].cpu().numpy().astype(np.uint8))
        if 'flow' in occ:
            result_dict['flow'] = occ['flow'].cpu().numpy()

        return result_dict