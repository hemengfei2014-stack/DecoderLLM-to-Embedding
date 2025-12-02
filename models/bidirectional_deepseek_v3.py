import importlib
import torch
from torch import nn

from transformers import DeepseekV3Config, DeepseekV3Model, DeepseekV3ForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3Attention,
    DeepseekV3MLP,
    DeepseekV3RMSNorm,
    DeepseekV3PreTrainedModel,
)

from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import PeftModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache

logger = logging.get_logger(__name__)


class ModifiedDeepseekV3Attention(DeepseekV3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedDeepseekV3FlashAttention2(DeepseekV3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedDeepseekV3SdpaAttention(DeepseekV3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


DEEPSEEK_ATTENTION_CLASSES = {
    "eager": ModifiedDeepseekV3Attention,
    "flash_attention_2": ModifiedDeepseekV3FlashAttention2,
    "sdpa": ModifiedDeepseekV3SdpaAttention,
}


class ModifiedDeepseekV3DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        impl = getattr(config, "_attn_implementation")
        if impl not in DEEPSEEK_ATTENTION_CLASSES:
            raise ValueError(f"Unsupported _attn_implementation for DeepSeek: {impl}")
        self.self_attn = DEEPSEEK_ATTENTION_CLASSES[impl](
            config=config, layer_idx=layer_idx
        )


class DeepseekV3BiModel(DeepseekV3Model):
    _no_split_modules = ["ModifiedDeepseekV3DecoderLayer"]

    def __init__(self, config: DeepseekV3Config):
        DeepseekV3PreTrainedModel.__init__(self, config)
        if not hasattr(config, "_attn_implementation"):
            raise ValueError("DeepseekV3BiModel requires config._attn_implementation to be set explicitly")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedDeepseekV3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
        self.rotary_emb = DeepseekV3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

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

        causal_mask = torch.zeros(
            (sequence_length, target_length), dtype=dtype, device=device
        )
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :, None, None, :
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=kwargs.get("output_attentions", False),
        )

        hidden_states = inputs_embeds
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class DeepseekV3BiForMNTP(DeepseekV3ForCausalLM):
    def __init__(self, config):
        DeepseekV3PreTrainedModel.__init__(self, config)
        self.model = DeepseekV3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    def save_peft_model(self, path):
        self.model.save_pretrained(path)
