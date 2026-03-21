import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from .builder import LANGUAGE_MODEL
from ..emova_arch import EmovaMetaModel, EmovaMetaForCausalLM

from typing import List, Optional, Tuple, Union

import transformers
from transformers.utils import is_torch_npu_available
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.generation.utils import GenerateOutput

from transformers.loss.loss_utils import LOSS_MAPPING, fixed_cross_entropy


# Modified to match the memory efficient loss.
def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    shift_logits = logits.view(-1, vocab_size)
    shift_labels = labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss

LOSS_MAPPING['ForCausalLM'] = ForCausalLMLoss


######################
# Overload for NPU
######################
# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L473
def Qwen2SdpaAttention_forward_npu(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    ###################
    # Modify for NPU
    ###################
    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type in ["cuda", "npu"] and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


if is_torch_npu_available():
    transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward = Qwen2SdpaAttention_forward_npu


class EmovaQwen2Config(Qwen2Config):
    model_type = "emova_qwen2"

    def __init__(self, **kwargs):
        self.hidden_fusion_method = kwargs.pop("hidden_fusion_method", "None")
        self.learn_fusion_weights = kwargs.pop("learn_fusion_weights", True)
        self.use_text_last_fetas = kwargs.pop("use_text_last_fetas", False)
        self.occ_loss_weight = kwargs.pop("occ_loss_weight", 1.0)
        self.occ_flow_loss_weight = kwargs.pop("occ_flow_loss_weight", 1.0)
        self.plan_loss_weight = kwargs.pop("plan_loss_weight", 1.0)
        self.text_loss_weight = kwargs.pop("text_loss_weight", 1.0)
        
        super().__init__(**kwargs)


from transformers.loss.loss_utils import fixed_cross_entropy


def ForCausalLMLossRemovePad(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # remove the pad token
    unpad_mask = shift_labels > -1
    logits = logits[unpad_mask]
    shift_labels = shift_labels[unpad_mask]
    
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


class EmovaQwen2Model(EmovaMetaModel, Qwen2Model):
    config_class = EmovaQwen2Config

    def __init__(self, config: EmovaQwen2Config):
        super(EmovaQwen2Model, self).__init__(config)

    def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
                self.config._attn_implementation == "sdpa"
                and not (using_static_cache or using_sliding_window_cache)
                and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    sliding_window=self.config.sliding_window,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        ###################
        # Modify for NPU
        ###################
        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type in ['cuda', 'npu']
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

# Custom output class extending CausalLMOutputWithPast to include segmentation outputs and loss
class CausalLMOutputWithPastAndSegmentation(CausalLMOutputWithPast):
    """
    Extended output class that includes semantic segmentation results and loss
    """
    def __init__(self,
                 loss=None,
                 logits=None,
                 past_key_values=None,
                 hidden_states=None,
                 attentions=None,
                 segmentation_outputs=None,
                 segmentation_loss=None,
                 text_loss=None):
        super().__init__(loss=loss, logits=logits, past_key_values=past_key_values,
                         hidden_states=hidden_states, attentions=attentions)
        self.segmentation_outputs = segmentation_outputs
        self.segmentation_loss = segmentation_loss
        self.text_loss = text_loss
        
# Custom output class for occupancy grid prediction
class CausalLMOutputWithPastAndOccupancy(CausalLMOutputWithPast):
    """
    Extended output class that includes BEV occupancy grid prediction results and loss
    """
    def __init__(self,
                 loss=None,
                 logits=None,
                 past_key_values=None,
                 hidden_states=None,
                 attentions=None,
                 occ_results=None,
                 occ_loss=None,
                 text_loss=None):
        super().__init__(loss=loss, logits=logits, past_key_values=past_key_values,
                         hidden_states=hidden_states, attentions=attentions)
        self.occ_results = occ_results
        self.occ_loss = occ_loss
        self.text_loss = text_loss
        

# Custom output class for path planning prediction
class CausalLMOutputWithPastAndPlanning(CausalLMOutputWithPast):
    """
    Extended output class that includes path planning prediction results and loss
    """
    def __init__(self,
                 loss=None,
                 logits=None,
                 past_key_values=None,
                 hidden_states=None,
                 attentions=None,
                 planning_results=None,
                 planning_loss=None,
                 text_loss=None):
        super().__init__(loss=loss, logits=logits, past_key_values=past_key_values,
                         hidden_states=hidden_states, attentions=attentions)
        self.planning_results = planning_results
        self.planning_loss = planning_loss
        self.text_loss = text_loss
        
        
# Custom output class that includes both occupancy grid and path planning
class CausalLMOutputWithPastAndMultiTask(CausalLMOutputWithPast):
    """
    Extended output class that includes both BEV occupancy grid and path planning prediction results and losses
    """
    def __init__(self,
                 loss=None,
                 logits=None,
                 past_key_values=None,
                 hidden_states=None,
                 attentions=None,
                 occ_results=None,
                 occ_loss=None,
                 planning_results=None,
                 planning_loss=None,
                 text_loss=None):
        super().__init__(loss=loss, logits=logits, past_key_values=past_key_values,
                         hidden_states=hidden_states, attentions=attentions)
        self.occ_results = occ_results
        self.occ_loss = occ_loss
        self.planning_results = planning_results
        self.planning_loss = planning_loss
        self.text_loss = text_loss


# Feature fusion module
class HiddenStatesFusion(nn.Module):
    """
    Module for fusing hidden states from different layers
    """

    def __init__(self, hidden_size, num_layers, fusion_method='weighted_sum', learn_weights=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fusion_method = fusion_method
        print('####self.fusion_method:', self.fusion_method)
        self.temporture = 1.0

        if fusion_method == 'weighted_sum':
            # Initialize weights, can be learnable or fixed
            if learn_weights:
                # Initialize with increasing weights, deeper layers have higher weights
                initial_weights = torch.linspace(0.5, 1.5, num_layers)
                initial_weights = initial_weights / initial_weights.sum()
                self.layer_weights = nn.Parameter(initial_weights)
            else:
                # Fixed weights, deeper layers have higher weights
                weights = torch.linspace(0.5, 1.5, num_layers)
                weights = weights / weights.sum()  # Normalize
                self.register_buffer('layer_weights', weights)

        elif fusion_method == 'gated':
            # Gated mechanism: uses few parameters but effectively fuses different layers
            self.gates = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        elif fusion_method == 'weighted_sum_v2':
                initial_weights = torch.linspace(0.5, 1.5, num_layers)
                initial_weights = initial_weights / initial_weights.sum()
                self.layer_weights = nn.Parameter(initial_weights)
    
        elif fusion_method == 'weighted_sum_v3':
            initial_weights = torch.linspace(1.0, 10.0, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            
        elif fusion_method == 'weighted_sum_v4':
            initial_weights = torch.linspace(0.1, 1.0, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            
        elif fusion_method == 'weighted_sum_v5':
            initial_weights = torch.linspace(1.0, 10.0, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            self.temporture = 0.2
            
        elif fusion_method == 'weighted_sum_v5_eh':
            initial_weights = torch.linspace(1.0, 10.0, num_layers-1)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            self.temporture = 0.2

        elif fusion_method == 'weighted_sum_v6':
            initial_weights = torch.linspace(0.1, 10, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            self.temporture = 0.2
            
        elif fusion_method == 'weighted_sum_v6_eh':
            initial_weights = torch.linspace(10, 10, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            self.temporture = 0.2
            
        elif fusion_method == 'weighted_sum_v7':
            initial_weights = torch.linspace(1.0, 10.0, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            self.temporture = 0.1

        elif fusion_method == 'weighted_sum_v8':
            initial_weights = torch.linspace(0.1, 10, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            self.temporture = 0.1
            
        elif fusion_method == 'weighted_sum_v9':
            initial_weights = torch.linspace(1.0, 10.0, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            self.temporture = 0.5

        elif fusion_method == 'weighted_sum_v10':
            initial_weights = torch.linspace(0.1, 10, num_layers)
            initial_weights = initial_weights / initial_weights.sum()
            self.layer_weights = nn.Parameter(initial_weights)
            self.temporture = 0.5
        else:
            pass
        
        print('####self.temporture:', self.temporture)

    def forward(self, hidden_states):
        """
        Fuse hidden states from all layers

        Args:
            hidden_states: List of hidden states from all layers [num_layers, batch_size, seq_len, hidden_size]

        Returns:
            Fused hidden states [batch_size, seq_len, hidden_size]
        """
        stacked_states = torch.stack(hidden_states, dim=0)  # [num_layers, batch, seq_len, hidden_size]

        if self.fusion_method == 'weighted_sum':
            # Weighted sum fusion - computationally efficient
            weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
        
        elif self.fusion_method == 'weighted_sum_v2':
            weights = self.layer_weights.view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
        
        elif self.fusion_method == 'weighted_sum_v3':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]

        elif self.fusion_method == 'weighted_sum_v4':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
            
        elif self.fusion_method == 'weighted_sum_v5':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            print('#####weights:', weights.view(-1))
            print('#####self.layer_weights/self.temporture:', self.layer_weights/self.temporture)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
            
        elif self.fusion_method == 'weighted_sum_v5_eh':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states[:-1] * weights).sum(dim=0) + stacked_states[-1]  # [layer_num, batch, seq_len, hidden_size]

        elif self.fusion_method == 'weighted_sum_v6':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
            
        elif self.fusion_method == 'weighted_sum_v6_eh':
            weights = (self.layer_weights/self.temporture).view(-1, 1, 1, 1)
            print('#####weights:', weights.view(-1))
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
            
        elif self.fusion_method == 'weighted_sum_v7':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
            
        elif self.fusion_method == 'weighted_sum_v8':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
            
        elif self.fusion_method == 'weighted_sum_v9':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]

        elif self.fusion_method == 'weighted_sum_v10':
            weights = torch.softmax(self.layer_weights/self.temporture, dim=0).view(-1, 1, 1, 1)
            fused = (stacked_states * weights).sum(dim=0)  # [batch, seq_len, hidden_size]
            
        elif self.fusion_method == 'gated':
            # Gated fusion - more efficient than attention
            gates = torch.softmax(self.gates, dim=0).view(-1, 1, 1, 1)
            weighted = stacked_states * gates
            fused = weighted.sum(dim=0)
            fused = self.layer_norm(fused)

        elif self.fusion_method == 'max_pooling':
            stacked_states = torch.stack(hidden_states, dim=0)  # [num_layers, batch, seq_len, hidden_size]
            fused, _ = torch.max(stacked_states, dim=0)  # [batch, seq_len, hidden_size]

        elif self.fusion_method == 'avg_pooling':
            stacked_states = torch.stack(hidden_states, dim=0)  # [num_layers, batch, seq_len, hidden_size]
            fused = torch.mean(stacked_states, dim=0)  # [batch, seq_len, hidden_size]

        else:
            # Default: use the last layer
            fused = hidden_states[-1]

        return fused


@LANGUAGE_MODEL.register_module()
class EmovaQwen2ForCausalLM(Qwen2ForCausalLM, EmovaMetaForCausalLM):
    config_class = EmovaQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = EmovaQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize hidden states fusion module
        self.hidden_fusion = HiddenStatesFusion(
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers + 1,
            fusion_method=config.hidden_fusion_method,
            learn_weights=config.learn_fusion_weights
        )

        self.use_text_last_fetas = config.use_text_last_fetas

        # Initialize weights and apply final processing
        self.post_init()
        self._loss_function = ForCausalLMLoss
        
        # Save loss weights
        self.occ_loss_weight = getattr(config, 'occ_loss_weight', 1.0)
        self.occ_flow_loss_weight = getattr(config, 'occ_flow_loss_weight', 1.0)
        self.plan_loss_weight = getattr(config, 'plan_loss_weight', 1.0)
        self.text_loss_weight = getattr(config, 'text_loss_weight', 1.0)
        print('###self.occ_loss_weight:', self.occ_loss_weight)
        print('###self.occ_flow_loss_weight:', self.occ_flow_loss_weight)
        print('###self.plan_loss_weight:', self.plan_loss_weight)
        print('###self.text_loss_weight:', self.text_loss_weight)

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            cache_position=None,  # Required for inference
            # Occupancy grid related parameters
            generate_occ: bool = False,  # Control whether to generate occupancy grid results
            gt_occ: Optional[torch.LongTensor] = None,  # Occupancy grid labels
            mask_camera: Optional[torch.Tensor] = None,  # Camera mask
            gt_occ_flow: Optional[torch.Tensor] = None,  # Occupancy flow labels
            # Path planning related parameters
            generate_plan: bool = False,  # Control whether to generate path planning results
            gt_planning: Optional[dict] = None,  # Path planning labels
            metas: Optional[List[dict]] = None,  # Metadata
            num_logits_to_keep: int = 0,
            **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # Save original input_ids for later feature extraction
        original_input_ids = input_ids
    
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # Force output hidden states if we have BEV occupancy grid estimator, or path planning head
        if ((self.get_bev_occ_estimator() is not None and generate_occ) or
            (self.get_diff_anchor_planner_head() is not None and generate_plan)):
            output_hidden_states = True

        # Ensure return_dict is True so we can add occupancy grid outputs
        if return_dict is None:
            return_dict = True

        if True:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss

            hidden_states = hidden_states[:, -num_logits_to_keep:, :]


            loss = None
            if labels is not None:
                # memory efficient.

                # Shift so that tokens < n predict n
                shift_hidden_states = hidden_states[..., :-1, :]
                shift_labels = labels[..., 1:]
                shift_mask = shift_labels > -1

                shift_labels = shift_labels[shift_mask].contiguous()
                logits = self.lm_head(shift_hidden_states[shift_mask].contiguous())
                logits = logits.float()
                loss = self.loss_function(logits, shift_labels, self.vocab_size, **loss_kwargs)
            else:
                logits = self.lm_head(hidden_states)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            llm_output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            llm_output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        # If return_dict is False, we can't add occupancy grid outputs
        if not return_dict:
            return llm_output

        # Initialize occupancy grid related variables
        occ_results = None
        occ_loss = None
        planning_results = None
        planning_loss = None
        text_loss = None

        # If we have hidden states, we can generate occupancy grid
        if hasattr(llm_output, 'hidden_states') and llm_output.hidden_states is not None:
            fused_hidden_states = self.hidden_fusion(llm_output.hidden_states)
            
            # If we have BEV occupancy grid estimator and need to generate occupancy grid or have occupancy grid labels
            if self.get_bev_occ_estimator() is not None and (generate_occ or gt_occ is not None):
                occ_output = self.generate_occupancy(
                    hidden_states=fused_hidden_states,
                    image_sizes=image_sizes,
                    metas=metas,
                    gt_occ=gt_occ,
                    mask_camera=mask_camera,
                    gt_occ_flow=gt_occ_flow,
                    input_ids=original_input_ids,
                    attention_mask=attention_mask,
                    ori_bev_feats=images,
                    return_results = False if self.training else True,  # Only return results during inference
                    hidden_states_last_feats= llm_output.hidden_states[-1] if self.use_text_last_fetas else None,
                )
                
                # If in training mode and occupancy grid labels are provided, compute loss
                if self.training and gt_occ is not None and occ_output:
                    # Get losses from occ_output
                    occ_loss_dict = {}
                    for k, v in occ_output.items():
                        if 'loss' in k:
                            occ_loss_dict[k] = v
                    
                    # Compute total occupancy grid loss
                    total_occ_loss = 0
                    for k, v in occ_loss_dict['losses'].items():
                        if 'flow' in k:
                            # Flow loss
                            weighted_loss = v * self.occ_flow_loss_weight
                        else:
                            # Occupancy grid loss
                            weighted_loss = v * self.occ_loss_weight
                        total_occ_loss += weighted_loss
                    
                    occ_loss = total_occ_loss
                    
                    # Add occupancy grid loss to total loss
                    if llm_output.loss is not None:
                        text_loss = llm_output.loss
                        llm_output.loss = text_loss + occ_loss
                
                # Get prediction results
                if 'occ_results' in occ_output:
                    occ_results = occ_output['occ_results']
                    
            # If we have path planning head and need to generate path planning or have path planning labels
            if self.get_diff_anchor_planner_head() is not None and (generate_plan or gt_planning is not None):
                planning_output = self.generate_planning(
                    hidden_states=fused_hidden_states,
                    image_sizes=image_sizes,
                    metas=metas,
                    gt_planning=gt_planning,
                    input_ids=original_input_ids,
                    attention_mask=attention_mask,
                    ori_bev_feats=images,
                    return_results=False if self.training else True,  # Only return results during inference
                    hidden_states_last_feats=llm_output.hidden_states[-1] if self.use_text_last_fetas else None,
                )

                if self.training and gt_planning is not None:
                    planning_loss = planning_output['losses']

                    # Add path planning loss to total loss
                    if llm_output.loss is not None:
                        plan_loss_weight = getattr(self, 'plan_loss_weight', 1.0)
                        # If text_loss is None, we need to save the original text loss
                        if text_loss is None:
                            text_loss = llm_output.loss
                        llm_output.loss = llm_output.loss + planning_loss * plan_loss_weight

                # Process results
                if 'results' in planning_output:
                    planning_results = planning_output['results']

        # Return appropriate output based on task type
        # If both occupancy grid and path planning are needed, return output containing both
        if (generate_occ or gt_occ is not None) and (generate_plan or gt_planning is not None):
            return CausalLMOutputWithPastAndMultiTask(
                loss=llm_output.loss,
                logits=llm_output.logits,
                past_key_values=llm_output.past_key_values,
                hidden_states=llm_output.hidden_states,
                attentions=llm_output.attentions,
                occ_results=occ_results,
                occ_loss=occ_loss,
                planning_results=planning_results,
                planning_loss=planning_loss,
                text_loss=text_loss,
            )
        # If only occupancy grid is needed, return output containing occupancy grid results
        elif generate_occ or gt_occ is not None:
            return CausalLMOutputWithPastAndOccupancy(
                loss=llm_output.loss,
                logits=llm_output.logits,
                past_key_values=llm_output.past_key_values,
                hidden_states=llm_output.hidden_states,
                attentions=llm_output.attentions,
                occ_results=occ_results,
                occ_loss=occ_loss,
                text_loss=text_loss,
            )
        # If only path planning is needed, return output containing path planning results
        elif generate_plan or gt_planning is not None:
            return CausalLMOutputWithPastAndPlanning(
                loss=llm_output.loss,
                logits=llm_output.logits,
                past_key_values=llm_output.past_key_values,
                hidden_states=llm_output.hidden_states,
                attentions=llm_output.attentions,
                planning_results=planning_results,
                planning_loss=planning_loss,
                text_loss=text_loss,
            )
        # Default: return original output
        else:
            return llm_output

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            generate_occ: bool = True,  # Control whether to generate occupancy grid results
            generate_plan: bool = True,
            metas: Optional[List[dict]] = None,  # Metadata
            val_gt_plan=None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # Save original inputs for later feature extraction
        original_inputs = inputs
        
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        print(inputs_embeds.shape)

        # If occupancy grid generation is needed, ensure hidden states are output
        if ((generate_occ and self.get_bev_occ_estimator() is not None) or
                (generate_plan and self.get_diff_anchor_planner_head() is not None)):
            kwargs['output_hidden_states'] = True
            kwargs['return_dict_in_generate'] = True

        generation_output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        # If occupancy grid generation is needed and generation output includes hidden states
        if (hasattr(generation_output, 'hidden_states') and generation_output.hidden_states is not None and val_gt_plan is not None):

            # Text is generated in an autoregressive manner with multiple iterations, but 3D perception heads only need the last layer hidden states
            # and are only forwarded once, so we can use the features saved from the text autoregression
            last_hidden_state = generation_output.hidden_states[0][-1]

            fused_hidden_states = self.hidden_fusion(generation_output.hidden_states[0])

            # If generation_output is not a dictionary, convert it to one
            if not isinstance(generation_output, dict):
                generation_output = generation_output._asdict()
            
            # Generate occupancy grid
            if generate_occ and self.get_bev_occ_estimator() is not None:
                occ_output = self.generate_occupancy(
                    hidden_states=fused_hidden_states,
                    image_sizes=image_sizes,
                    metas=metas,
                    input_ids=original_inputs,
                    attention_mask=attention_mask,
                    ori_bev_feats=images,
                    return_results=True,
                    hidden_states_last_feats=last_hidden_state if self.use_text_last_fetas else None,
                )
                # Add occupancy grid outputs
                if 'occ_results' in occ_output:
                    generation_output['occ_results'] = occ_output['occ_results']

            # Generate path planning
            if generate_plan and self.get_diff_anchor_planner_head() is not None:
                planning_output = self.generate_planning(
                    hidden_states=fused_hidden_states,
                    image_sizes=image_sizes,
                    metas=metas,
                    gt_planning=val_gt_plan,  # Won't be used during inference
                    input_ids=original_inputs,
                    attention_mask=attention_mask,
                    ori_bev_feats=images,
                    return_results=True,
                    hidden_states_last_feats=last_hidden_state if self.use_text_last_fetas else None,
                )
                # Add path planning outputs
                if planning_output is not None:
                    if isinstance(planning_output, dict) and 'results' in planning_output:
                        generation_output['planning_results'] = planning_output['results']
                    else:
                        generation_output['planning_results'] = planning_output
        return generation_output

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        generate_occ = kwargs.pop("generate_occ", False)
        generate_plan = kwargs.pop("generate_plan", False)
        metas = kwargs.pop("metas", None)
        val_gt_plan = kwargs.pop("val_gt_plan", None)

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if generate_occ:
            inputs['generate_occ'] = generate_occ
        if generate_plan:
            inputs['generate_plan'] = generate_plan
        if metas is not None:
            inputs['metas'] = metas
        if val_gt_plan:
            inputs['val_gt_plan'] = val_gt_plan

        return inputs

AutoConfig.register("emova_qwen2", EmovaQwen2Config)
AutoModelForCausalLM.register(EmovaQwen2Config, EmovaQwen2ForCausalLM)