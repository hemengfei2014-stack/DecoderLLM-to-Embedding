import torch
from torch import nn

from transformers import Qwen3Model, Qwen3ForCausalLM, Qwen3PreTrainedModel, Qwen3Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
)
from peft import PeftModel


class ModifiedQwen3Attention(Qwen3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = ModifiedQwen3Attention(config=config, layer_idx=layer_idx)


class Qwen3BiModel(Qwen3Model):
    _no_split_modules = ["ModifiedQwen3DecoderLayer"]

    def __init__(self, config: Qwen3Config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            ModifiedQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        def _build_pad_mask(attn_mask, seq_len, dtype, device):
            if attn_mask is None:
                return None
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.ne(0)
            bs = attn_mask.size(0)
            min_dtype = torch.finfo(dtype).min
            out = torch.zeros((bs, 1, seq_len, seq_len), dtype=dtype, device=device)
            pad_cols = (~attn_mask).to(dtype=dtype) * min_dtype
            out[:, 0, :, :] = pad_cols[:, None, :].expand(bs, seq_len, seq_len)
            return out

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        impl = getattr(self.config, 'attn_implementation', 'sdpa')
        if impl == 'flash_attention_2':
            # FlashAttention v2 需要 2D 变长 mask（bs, seqlen），由内部构建 cu_seqlens
            mask_for_layer = attention_mask if attention_mask is None else (attention_mask.ne(0) if attention_mask.dtype != torch.bool else attention_mask)
        else:
            # SDPA/Eager 路径使用 4D pad 掩码（bs, 1, L, L）
            mask_for_layer = _build_pad_mask(attention_mask, hidden_states.size(1), hidden_states.dtype, hidden_states.device)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_for_layer,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3BiForMNTP(Qwen3ForCausalLM):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    def save_peft_model(self, path):
        self.model.save_pretrained(path)
