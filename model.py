# model.py
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from functools import partial
import math
from attnres_ops import (
    attention_residual_average_read,
    attention_residual_phase1_from_logits,
    attention_residual_phase2,
    attention_residual_phase2_torch,
    attention_residual_phase2_from_logit,
    attention_residual_read,
    lrid_attention_residual_phase2,
    lrid_attention_residual_phase2_torch,
    lrid_attention_residual_read,
)


def rms_norm_eps(x: torch.Tensor, eps: float = None) -> float:
    if eps is not None:
        return eps
    return torch.finfo(x.dtype).eps


def norm(x: Tensor, eps: float = None):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),), eps=eps)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps(x, eps))


def _norm_lrid_key(x: Tensor, num_heads: int):
    x_shape = x.shape
    x = x.reshape(*x_shape[:-1], num_heads, x_shape[-1] // num_heads)
    return norm(x).reshape(*x_shape)


class TokenEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, idx):
        return F.embedding(idx, self.weight)


@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 100277
    n_layer: int = 12 
    n_head: int = 12
    n_embd: int = 768
    mlp_hidden_dim: int = None
    mlp_ratio: float = 4.0
    weight_tying: bool = False
    rope_theta: float = 500000.0
    norm_pos: str = "after"
    qk_norm: bool = True
    clip_qkv: float = None
    flash_attention: bool = False
    init_std: float = 0.02
    init_cutoff_factor: float = None
    attnres_type: str = None
    use_attnres: bool = False
    use_fused_attnres: bool = False
    attnres_num_blocks: int = 8
    attnres_block_average: bool = True
    attnres_block_average_mode: str = "count"
    attnres_block_count_prior: bool = True
    attnres_block_learned_scale: bool = False
    attnres_block_learned_scale_init: str = "count"
    attnres_block_value_norm: bool = False
    attnres_key_norm: bool = True
    attn_res_query_norm: bool = False
    attn_res_query_init: str = "zero"
    attnres_training_cache_phase1: bool = True
    attnres_training_torch_phase2: bool = True
    attnres_fuse_read_norm: bool = True
    use_lrid: bool = False
    lrid_rank: int = 64
    lrid_projection_rank: int = None
    lrid_num_heads: int = 1
    lrid_input_dependent_query: bool = False
    lrid_static_embedding_key: bool = False
    lrid_add_static_embedding_key: bool = False
    lrid_add_static_source_key: bool = False
    lrid_key_from_value: bool = False
    lrid_key_from_value_shared: bool = False
    lrid_key_from_output_tail: bool = False
    lrid_key_value_norm: bool = True
    lrid_query_from_value: bool = False
    lrid_query_from_value_shared: bool = False
    lrid_use_logit_scale: bool = True
    lrid_logit_scale: float = None

    def __post_init__(self):
        self.attnres_type = (self.attnres_type or "block")
        self.attnres_type = self.attnres_type.lower()
        self.attnres_block_average_mode = (self.attnres_block_average_mode or "count").lower()
        if self.attnres_block_average_mode not in {"count", "sqrt"}:
            raise ValueError("attnres_block_average_mode must be one of: count, sqrt")
        self.attnres_block_learned_scale_init = self._normalize_block_scale_init(
            self.attnres_block_learned_scale_init
        )
        if self.attnres_block_learned_scale and self.attnres_type != "block":
            raise ValueError("attnres_block_learned_scale requires attnres_type='block'")
        if self.attnres_block_value_norm and self.attnres_type != "block":
            raise ValueError("attnres_block_value_norm requires attnres_type='block'")
        if self.attnres_block_value_norm and self.attnres_block_learned_scale:
            raise ValueError("attnres_block_value_norm and attnres_block_learned_scale are mutually exclusive")
        if (
            self.attnres_type == "block"
            and self.attnres_block_count_prior
            and (
                not self.attnres_block_average
                or self.attnres_block_average_mode != "count"
                or self.attnres_block_learned_scale
                or self.attnres_block_value_norm
            )
        ):
            raise ValueError(
                "attnres_block_count_prior requires count-mean block summaries: "
                "attnres_block_average=True, attnres_block_average_mode='count', "
                "attnres_block_learned_scale=False, and attnres_block_value_norm=False"
            )
        self.attn_res_query_init = (self.attn_res_query_init or "zero").lower()
        if self.attn_res_query_init not in {"zero", "normal", "trunc_normal"}:
            raise ValueError("attn_res_query_init must be one of: zero, normal, trunc_normal")
        if self.lrid_static_embedding_key and self.lrid_add_static_embedding_key:
            raise ValueError("lrid_static_embedding_key and lrid_add_static_embedding_key are mutually exclusive")
        if self.lrid_key_from_value_shared:
            self.lrid_key_from_value = True
        if self.lrid_key_from_output_tail:
            if self.lrid_key_from_value:
                raise ValueError("lrid_key_from_output_tail and lrid_key_from_value are mutually exclusive")
            if self.lrid_static_embedding_key or self.lrid_add_static_embedding_key or self.lrid_add_static_source_key:
                raise ValueError(
                    "lrid_key_from_output_tail cannot be combined with static LRID key additions"
                )
        if self.lrid_query_from_value_shared:
            self.lrid_query_from_value = True
        if self.lrid_rank < 1:
            raise ValueError("lrid_rank must be >= 1")
        if self.lrid_key_from_output_tail and self.lrid_rank > self.n_embd:
            raise ValueError("lrid_rank must be <= n_embd when lrid_key_from_output_tail=True")
        if self.lrid_projection_rank is None:
            self.lrid_projection_rank = self.lrid_rank
        if self.lrid_projection_rank < self.lrid_rank:
            raise ValueError("lrid_projection_rank must be >= lrid_rank")
        if self.lrid_num_heads < 1:
            raise ValueError("lrid_num_heads must be >= 1")
        if self.lrid_rank % self.lrid_num_heads != 0:
            raise ValueError("lrid_rank must be divisible by lrid_num_heads")
        if self.n_embd % self.lrid_num_heads != 0:
            raise ValueError("n_embd must be divisible by lrid_num_heads")
        if not self.lrid_use_logit_scale:
            self.lrid_logit_scale = 1.0
        elif self.lrid_logit_scale is None:
            self.lrid_logit_scale = 1.0 / math.sqrt(self.lrid_rank // self.lrid_num_heads)
        elif self.lrid_logit_scale <= 0.0:
            raise ValueError("lrid_logit_scale must be positive")
        if self.use_lrid:
            self.use_attnres = True

    @staticmethod
    def _normalize_block_scale_init(value):
        value = str(value or "count").lower().replace(" ", "")
        if value in {"count", "1/c", "inv_count", "inverse_count"}:
            return "count"
        if value in {"sqrt", "1/sqrtc", "1/sqrt(c)", "inv_sqrt", "inverse_sqrt"}:
            return "sqrt"
        if value in {"one", "1"}:
            return "one"
        raise ValueError(
            "attnres_block_learned_scale_init must be one of: count, sqrt, one"
        )


class LRIDStaticKey(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Parameter(torch.empty(config.lrid_rank))

    def reset_parameters(self, std=0.02, init_cutoff_factor=None):
        if init_cutoff_factor is not None:
            cutoff = init_cutoff_factor * std
            nn.init.trunc_normal_(self.key, mean=0.0, std=std, a=-cutoff, b=cutoff)
        else:
            nn.init.normal_(self.key, mean=0.0, std=std)

    def forward(self, reference):
        return self.key.to(reference.dtype).view(1, 1, -1).expand(reference.size(0), reference.size(1), -1)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.n_embd // config.n_head
        max_seq_len = config.block_size
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        freq = torch.outer(torch.arange(max_seq_len), inv_freq)
        self.register_buffer("sin", freq.sin()[None, None])
        self.register_buffer("cos", freq.cos()[None, None])

    def _forward_single(self, x, offset=0):
        T = x.size(-2)
        sin = self.sin[:, :, offset:offset + T]
        cos = self.cos[:, :, offset:offset + T]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1).flatten(-2)

    def forward(self, q, k=None, offset=0):
        if k is None:
            return self._forward_single(q, offset=offset)

        return self._forward_single(q, offset=offset), self._forward_single(k, offset=offset)


class LRIDFusedProjection(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.rank = config.lrid_rank
        self.projection_rank = config.lrid_projection_rank
        self.use_query = config.lrid_input_dependent_query
        self.use_output_tail_key = config.lrid_key_from_output_tail
        self.use_value_key = config.lrid_key_from_value
        self.use_shared_value_key = config.lrid_key_from_value_shared
        self.use_local_value_key = self.use_value_key and not self.use_shared_value_key
        self.use_key = not (self.use_value_key or self.use_output_tail_key)
        self.use_value_query = self.use_query and config.lrid_query_from_value
        self.use_shared_value_query = self.use_query and config.lrid_query_from_value_shared
        self.use_local_value_query = self.use_value_query and not self.use_shared_value_query
        self.use_fused_query = self.use_query and not self.use_value_query
        self.key_offset = output_dim
        self.query_offset = output_dim + (self.projection_rank if self.use_key else 0)
        extra_dim = (self.projection_rank if self.use_key else 0) + (self.rank if self.use_fused_query else 0)
        self.proj = nn.Linear(input_dim, output_dim + extra_dim, bias=False)
        self.use_key_norm = self.use_key and config.attnres_key_norm
        self.use_value_norm = (self.use_local_value_key or self.use_local_value_query) and config.lrid_key_value_norm
        self.num_heads = config.lrid_num_heads
        if self.use_local_value_key:
            self.value_key_proj = nn.Linear(output_dim, config.lrid_rank, bias=False)
        else:
            self.value_key_proj = None
        if self.use_local_value_query:
            self.value_query_proj = nn.Linear(output_dim, config.lrid_rank, bias=False)
        else:
            self.value_query_proj = None

    def _prepare_value_projection_input(self, value):
        if self.use_value_norm:
            return norm(value)
        return value

    def project_key_from_value(self, value):
        if not self.use_local_value_key:
            raise RuntimeError("Local value-key projection is only available in unshared lrid_key_from_value mode")
        return self.value_key_proj(self._prepare_value_projection_input(value))

    def project_query_from_value(self, value):
        if not self.use_local_value_query:
            raise RuntimeError("Local value-query projection is only available in unshared lrid_query_from_value mode")
        return self.value_query_proj(self._prepare_value_projection_input(value))

    def forward(self, x, emit_lrid_key=True):
        if not emit_lrid_key:
            output = F.linear(x, self.proj.weight[:self.output_dim], None)
            if self.use_query:
                return output, None, None
            return output, None

        projected = self.proj(x)
        output = projected[..., :self.output_dim].contiguous()
        key = None
        query = None
        if self.use_key:
            key = projected[..., self.key_offset:self.key_offset + self.rank]
            if self.use_key_norm:
                key = _norm_lrid_key(key, self.num_heads)
        elif self.use_output_tail_key:
            key = output[..., -self.rank:].contiguous()
        elif self.use_local_value_key:
            key = self.project_key_from_value(output)
        if self.use_fused_query:
            query = projected[..., self.query_offset:self.query_offset + self.rank]
        elif self.use_local_value_query:
            query = self.project_query_from_value(output)
        if self.use_query:
            return output, key, query
        return output, key


class LRIDSourceKeyProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = config.lrid_rank
        self.num_heads = config.lrid_num_heads
        self.use_output_tail_key = config.lrid_key_from_output_tail
        uses_value_projection = (
            config.lrid_key_from_value
            or (config.lrid_input_dependent_query and config.lrid_query_from_value_shared)
        )
        self.use_value_norm = uses_value_projection and config.lrid_key_value_norm
        self.proj = None if self.use_output_tail_key else nn.Linear(config.n_embd, config.lrid_rank, bias=False)
        if config.lrid_input_dependent_query and config.lrid_query_from_value_shared:
            self.query_proj = nn.Linear(config.n_embd, config.lrid_rank, bias=False)
        else:
            self.query_proj = None
        self.use_key_norm = config.attnres_key_norm and not config.lrid_key_from_value

    def _prepare_value_projection_input(self, x):
        if self.use_value_norm:
            return norm(x)
        return x

    def forward(self, x):
        if self.use_output_tail_key:
            return x[..., -self.rank:].contiguous()
        if self.config.lrid_key_from_value:
            x = self._prepare_value_projection_input(x)
        key = self.proj(x)
        if self.use_key_norm:
            key = _norm_lrid_key(key, self.num_heads)
        return key

    def project_query_from_value(self, x):
        if self.query_proj is None:
            raise RuntimeError("Shared value-query projection is only available when lrid_query_from_value_shared=True")
        return self.query_proj(self._prepare_value_projection_input(x))

class MultiHeadAttention(nn.Module):
    flash_attn_func = None
    flash_attn_varlen_func = None
    flash_tried = False
    
    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rope = RotaryEmbedding(config)
        self.layer_idx = layer_idx
        self.config = config
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        if config.use_lrid:
            self.c_proj = LRIDFusedProjection(config, config.n_embd, config.n_embd)
        else:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.use_qk_norm = config.qk_norm
        
        self.clip_qkv = config.clip_qkv
        
        if config.flash_attention and not MultiHeadAttention.flash_tried:
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                MultiHeadAttention.flash_attn_func = flash_attn_func
                MultiHeadAttention.flash_attn_varlen_func = flash_attn_varlen_func
                MultiHeadAttention.flash_tried = True
            except Exception as e:
                print(f"Error with flash-attn {e}.")
                MultiHeadAttention.flash_tried = True

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, is_causal=True, 
                                       cu_doc_len=None, max_doc_len=None):
        B, H, T, D = q.size()
        
        if cu_doc_len is not None and max_doc_len is not None and MultiHeadAttention.flash_attn_varlen_func is not None:
            q_flat = q.transpose(1, 2).reshape(B * T, H, D)
            k_flat = k.transpose(1, 2).reshape(B * T, H, D)
            v_flat = v.transpose(1, 2).reshape(B * T, H, D)

            cu_doc_len = cu_doc_len.to(device=q.device, dtype=torch.int32)
            x = MultiHeadAttention.flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_doc_len,
                cu_seqlens_k=cu_doc_len,
                max_seqlen_q=max_doc_len,
                max_seqlen_k=max_doc_len,
                causal=is_causal,
            )
            return x.view(B, T, H, D).contiguous().view(B, T, self.n_embd)

        elif cu_doc_len is not None or max_doc_len is not None:
            raise RuntimeError(
                "Document masking requires flash-attn varlen support. "
                "Install flash-attn or disable use_doc_masking."
            )
        
        elif MultiHeadAttention.flash_attn_func is not None and attn_mask is None:
            x = MultiHeadAttention.flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=is_causal,
            )
            return x.contiguous().view(B, T, self.n_embd)
        
        else:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            return x.transpose(1, 2).contiguous().view(B, T, self.n_embd)

    def forward(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None, emit_lrid_key=True):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        if self.clip_qkv is not None:
            q.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            k.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            v.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if self.use_qk_norm:
            q = norm(q)
            k = norm(k)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            pos_offset = past_k.size(-2)
        else:
            pos_offset = 0
        
        q, k = self.rope(q, k, offset=pos_offset)
        
        if past_kv is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        is_causal = past_kv is None

        attention_output = self._scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            cu_doc_len=cu_doc_len,
            max_doc_len=max_doc_len,
        )
        
        if self.config.use_lrid:
            projected = self.c_proj(attention_output, emit_lrid_key=emit_lrid_key)
        else:
            projected = self.c_proj(attention_output)
        if self.config.use_lrid:
            if self.config.lrid_input_dependent_query:
                x, lrid_key, lrid_query = projected
            else:
                x, lrid_key = projected
        else:
            x = projected
        
        if use_cache:
            if self.config.use_lrid:
                if self.config.lrid_input_dependent_query:
                    return x, (k, v), lrid_key, lrid_query
                return x, (k, v), lrid_key
            return x, (k, v)
        if self.config.use_lrid:
            if self.config.lrid_input_dependent_query:
                return x, lrid_key, lrid_query
            return x, lrid_key
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.mlp_hidden_dim if config.mlp_hidden_dim is not None else int(config.n_embd * config.mlp_ratio)
        self.fc1 = nn.Linear(config.n_embd, self.hidden_dim * 2, bias=False)
        if config.use_lrid:
            self.fc2 = LRIDFusedProjection(config, self.hidden_dim, config.n_embd)
        else:
            self.fc2 = nn.Linear(self.hidden_dim, config.n_embd, bias=False)

    def forward(self, x, emit_lrid_key=True):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        if self.config.use_lrid:
            return self.fc2(x, emit_lrid_key=emit_lrid_key)
        return self.fc2(x)


class AttentionResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_key_norm = config.attnres_key_norm
        self.query = nn.Parameter(torch.empty(config.n_embd))

    def _query(self, dtype):
        query = self.query
        if self.config.attn_res_query_norm:
            query = norm(query.float())
        return query.to(dtype)

    def forward(self, values, source_counts=None):
        keys = norm(values) if self.use_key_norm else values
        logits = torch.einsum("d,sbtd->sbt", self._query(keys.dtype), keys)
        if source_counts is not None:
            log_counts = torch.as_tensor(source_counts, device=logits.device, dtype=torch.float32).log()
            logits = logits + log_counts.view(-1, 1, 1)
        weights = F.softmax(logits.float(), dim=0).to(values.dtype)
        return torch.einsum("sbt,sbtd->btd", weights, values)


class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.norm_pos = config.norm_pos
        self.attn = MultiHeadAttention(config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx
        self.config = config

    def forward_attention(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None, x_is_normalized=False, emit_lrid_key=True):
        if self.norm_pos in {"before", "both"} and not x_is_normalized:
            x = norm(x)

        attn_out = self.attn(
            x,
            past_kv=past_kv,
            use_cache=use_cache,
            cu_doc_len=cu_doc_len,
            max_doc_len=max_doc_len,
            emit_lrid_key=emit_lrid_key,
        )

        if self.config.use_lrid:
            if self.config.lrid_input_dependent_query:
                if use_cache:
                    x, new_kv, lrid_key, lrid_query = attn_out
                else:
                    x, lrid_key, lrid_query = attn_out
                    new_kv = None
            else:
                if use_cache:
                    x, new_kv, lrid_key = attn_out
                else:
                    x, lrid_key = attn_out
                    new_kv = None
        elif use_cache:
            x, new_kv = attn_out
        else:
            x = attn_out
            new_kv = None

        if self.norm_pos in {"after", "both"}:
            x = norm(x)

        if use_cache:
            if self.config.use_lrid:
                if self.config.lrid_input_dependent_query:
                    return x, new_kv, lrid_key, lrid_query
                return x, new_kv, lrid_key
            return x, new_kv
        if self.config.use_lrid:
            if self.config.lrid_input_dependent_query:
                return x, lrid_key, lrid_query
            return x, lrid_key
        return x

    def forward_mlp(self, x, x_is_normalized=False, emit_lrid_key=True):
        if self.norm_pos in {"before", "both"} and not x_is_normalized:
            x = norm(x)

        mlp_out = self.mlp(x, emit_lrid_key=emit_lrid_key)
        if self.config.use_lrid:
            if self.config.lrid_input_dependent_query:
                x, lrid_key, lrid_query = mlp_out
            else:
                x, lrid_key = mlp_out
        else:
            x = mlp_out

        if self.norm_pos in {"after", "both"}:
            x = norm(x)

        if self.config.use_lrid:
            if self.config.lrid_input_dependent_query:
                return x, lrid_key, lrid_query
            return x, lrid_key
        return x

    def forward(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None):
        residual = x

        attn_out = self.forward_attention(
            x,
            past_kv=past_kv,
            use_cache=use_cache,
            cu_doc_len=cu_doc_len,
            max_doc_len=max_doc_len,
        )

        if use_cache:
            x, new_kv = attn_out
        else:
            x = attn_out
            new_kv = None

        x = residual + x

        residual = x

        x = self.forward_mlp(x)

        x = residual + x

        if use_cache:
            return x, new_kv
        else:
            return x


class OBPM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_attnres = config.use_attnres
        self.use_fused_attnres = config.use_fused_attnres
        self.attnres_type = config.attnres_type
        self.use_lrid = config.use_lrid
        self.attnres_block_ends = self._make_attnres_block_ends()
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        
        transformer_modules = dict(
            wte=TokenEmbedding(config.vocab_size, config.n_embd),
            layers=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)])
        )

        if self.use_attnres:
            if self.attnres_type not in {"full", "block"}:
                raise ValueError("attnres_type must be 'full' or 'block'")
            if self.attnres_type == "block" and config.attnres_num_blocks < 1:
                raise ValueError("attnres_num_blocks must be >= 1 when using block AttnRes")
            if self.use_lrid:
                needs_lrid_source_projection = (
                    (not config.lrid_static_embedding_key and not config.lrid_key_from_output_tail)
                    or config.lrid_key_from_value_shared
                    or (config.lrid_input_dependent_query and config.lrid_query_from_value_shared)
                )
                if needs_lrid_source_projection:
                    transformer_modules["lrid_embedding_key"] = LRIDSourceKeyProjection(config)
                if config.lrid_static_embedding_key or config.lrid_add_static_embedding_key:
                    transformer_modules["lrid_static_embedding_key"] = LRIDStaticKey(config)
                if config.lrid_add_static_source_key:
                    transformer_modules["lrid_static_source_key"] = LRIDStaticKey(config)
                key_head_dim = config.lrid_rank // config.lrid_num_heads
                transformer_modules["lrid_queries"] = nn.ParameterList(
                    [
                        nn.Parameter(torch.empty(config.lrid_num_heads, key_head_dim))
                        for _ in range(2 * config.n_layer)
                    ]
                )
                if config.lrid_input_dependent_query:
                    transformer_modules["lrid_query_gates"] = nn.ParameterList(
                        [
                            nn.Parameter(torch.zeros(config.lrid_num_heads))
                            for _ in range(2 * config.n_layer)
                        ]
                    )
            else:
                transformer_modules["attn_residuals"] = nn.ModuleList(
                    [AttentionResidual(config) for _ in range(2 * config.n_layer)]
                )

        self.transformer = nn.ModuleDict(transformer_modules)
        if self.use_attnres and self.attnres_type == "block" and config.attnres_block_learned_scale:
            self.transformer.register_parameter(
                "attnres_block_scales",
                nn.Parameter(self._make_attnres_block_scale_init()),
            )
        
        if not config.weight_tying:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(partial(self._init_weights, std=config.init_std, init_cutoff_factor=config.init_cutoff_factor))
        if self.use_lrid:
            for query in self.transformer.lrid_queries:
                self._init_attnres_query(query, config.init_std, config.init_cutoff_factor)
        if self.use_lrid and config.lrid_input_dependent_query:
            for module in self.modules():
                if isinstance(module, LRIDFusedProjection):
                    self._init_lrid_dynamic_query_projection(module, config.init_std, config.init_cutoff_factor)
    def to_mixed_precision(self, dtype=torch.bfloat16):
        self.to(dtype=dtype)
        return self
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _make_attnres_block_ends(self):
        if not self.use_attnres or self.attnres_type != "block":
            return None
        total_sublayers = 2 * self.config.n_layer
        num_blocks = min(self.config.attnres_num_blocks, total_sublayers)
        return frozenset(
            math.ceil(total_sublayers * i / num_blocks)
            for i in range(1, num_blocks + 1)
        )

    def _attnres_block_partial_count(self, summary_idx):
        if summary_idx < 1 or summary_idx > 2 * self.config.n_layer:
            raise RuntimeError("Block summary index is out of range")
        previous_end = 0
        for block_end in sorted(self.attnres_block_ends):
            if summary_idx <= block_end:
                return summary_idx - previous_end
            previous_end = block_end
        raise RuntimeError("Block summary index does not belong to any block")

    def _make_attnres_block_scale_init(self):
        values = []
        for summary_idx in range(1, 2 * self.config.n_layer + 1):
            count = self._attnres_block_partial_count(summary_idx)
            if self.config.attnres_block_learned_scale_init == "count":
                scale = 1.0 / count
            elif self.config.attnres_block_learned_scale_init == "sqrt":
                scale = 1.0 / math.sqrt(count)
            else:
                scale = 1.0
            values.append(scale)
        return torch.tensor(values, dtype=torch.float32)

    def _attnres_query_idx(self, residual_idx):
        if residual_idx < 1:
            raise RuntimeError("Residual site 0 has no query because it only reads the embedding source")
        return residual_idx - 1

    def _attnres_block_source_scale(self, count, summary_idx=None, dtype=None, device=None):
        if self.config.attnres_block_learned_scale:
            if summary_idx is None:
                raise RuntimeError("Learned block scaling requires a block summary index")
            scale = self.transformer.attnres_block_scales[summary_idx - 1]
            if dtype is not None or device is not None:
                scale = scale.to(dtype=dtype or scale.dtype, device=device or scale.device)
            return scale
        if self.config.attnres_block_average:
            return 1.0 / self._attnres_block_average_denominator(count)
        return None

    def _attnres_block_summary(self, value, count, summary_idx=None):
        if self.config.attnres_block_value_norm:
            return norm(value)
        scale = self._attnres_block_source_scale(
            count,
            summary_idx=summary_idx,
            dtype=value.dtype,
            device=value.device,
        )
        if scale is not None:
            return value * scale
        return value

    def _lrid_block_source(self, value, key=None, query=None, count=1, summary_idx=None):
        if self.config.attnres_block_value_norm:
            value = norm(value)
        else:
            scale = self._attnres_block_source_scale(
                count,
                summary_idx=summary_idx,
                dtype=value.dtype,
                device=value.device,
            )
            if scale is not None:
                value = value * scale
                if key is not None and not self.config.lrid_key_from_value and not self.config.attnres_key_norm:
                    if torch.is_tensor(scale):
                        scale = scale.to(dtype=key.dtype, device=key.device)
                    key = key * scale
        key_value = value if self.config.lrid_key_from_value_shared else None
        query_value = value if self.config.lrid_query_from_value_shared else None
        return self._lrid_source(
            value,
            key,
            query,
            key_value=key_value,
            query_value=query_value,
            add_static_key=True,
        )

    def _attnres_block_average_denominator(self, count):
        if self.config.attnres_block_average_mode == "sqrt":
            return math.sqrt(count)
        return count

    def _use_attnres_block_count_prior(self):
        return self.attnres_type == "block" and self.config.attnres_block_count_prior

    def _attnres_block_count_logit_bias(self, count):
        if not self._use_attnres_block_count_prior():
            return 0.0
        return math.log(float(count))

    def _attnres_block_source_counts(self, completed_counts, partial_count=None):
        if not self._use_attnres_block_count_prior():
            return None
        if partial_count is None:
            return completed_counts
        return completed_counts + [partial_count]

    def _apply_attnres(self, residual_idx, sources, normalize_output=False, average_read=False, source_counts=None):
        if len(sources) == 1:
            return norm(sources[0]) if normalize_output else sources[0]
        if average_read:
            return attention_residual_average_read(sources, normalize_output=normalize_output)
        residual = self.transformer.attn_residuals[self._attnres_query_idx(residual_idx)]
        if self.config.use_fused_attnres:
            query = residual._query(sources[0].dtype)
            return attention_residual_read(
                sources,
                query,
                residual.use_key_norm,
                normalize_output=normalize_output,
                source_counts=source_counts,
            )
        values = torch.stack(sources, dim=0)
        output = residual(values, source_counts=source_counts)
        return norm(output) if normalize_output else output

    def _attnres_query(self, residual_idx, dtype):
        residual = self.transformer.attn_residuals[self._attnres_query_idx(residual_idx)]
        return residual._query(dtype)

    def _apply_training_phase2(
        self,
        partial_source,
        query,
        interblock_output,
        interblock_lse,
        normalize_output=False,
        partial_count=1,
    ):
        phase2 = attention_residual_phase2_torch if self.config.attnres_training_torch_phase2 else attention_residual_phase2
        return phase2(
            partial_source,
            query,
            interblock_output,
            interblock_lse,
            self.config.attnres_key_norm,
            normalize_output=normalize_output,
            logit_bias=self._attnres_block_count_logit_bias(partial_count),
        )

    def _apply_lrid_training_phase2(
        self,
        partial_value,
        partial_key,
        query,
        interblock_output,
        interblock_lse,
        normalize_output=False,
        partial_count=1,
    ):
        phase2 = lrid_attention_residual_phase2_torch if self.config.attnres_training_torch_phase2 else lrid_attention_residual_phase2
        return phase2(
            partial_value,
            partial_key,
            query,
            interblock_output,
            interblock_lse,
            self.config.lrid_logit_scale,
            self.config.attnres_key_norm,
            normalize_output=normalize_output,
            logit_bias=self._attnres_block_count_logit_bias(partial_count),
        )

    def _forward_block_attnres_fused_training(
        self,
        x,
        past_kv=None,
        use_cache=False,
        cu_doc_len=None,
        max_doc_len=None,
        return_hidden=False,
        fused_read_norm=False,
    ):
        if past_kv is not None or use_cache:
            raise NotImplementedError("KV-cache generation is not supported with attention residuals yet.")

        total_sublayers = 2 * self.config.n_layer
        block_ends = sorted(self.attnres_block_ends)
        block_end_set = self.attnres_block_ends
        query_bank = torch.stack(
            [
                self._attnres_query(read_idx, x.dtype)
                for read_idx in range(1, total_sublayers + 1)
            ],
            dim=0,
        ).contiguous()

        def make_source_logits(source, first_read, source_count=1):
            key = norm(source) if self.config.attnres_key_norm else source
            queries = query_bank[first_read - 1:].to(key.dtype)
            logits = F.linear(key, queries).permute(2, 0, 1).contiguous()
            if self._use_attnres_block_count_prior() and source_count != 1:
                logits = logits + math.log(float(source_count))
            return logits

        completed_blocks = [x]
        completed_logits = [make_source_logits(x, 1, 1)]
        completed_logit_first_reads = [1]
        partial_block = None
        partial_count = 0
        phase_first_read = None
        phase_end = None
        phase_outputs = None
        phase_lses = None

        def next_phase_end(residual_idx):
            for block_end in block_ends:
                if block_end > residual_idx:
                    return block_end
            return residual_idx + 1

        def invalidate_phase():
            nonlocal phase_first_read, phase_end, phase_outputs, phase_lses
            phase_first_read = None
            phase_end = None
            phase_outputs = None
            phase_lses = None

        def ensure_phase(residual_idx):
            nonlocal phase_first_read, phase_end, phase_outputs, phase_lses
            if residual_idx < 1:
                return
            if phase_outputs is not None and phase_first_read <= residual_idx < phase_end:
                return
            phase_first_read = residual_idx
            phase_end = next_phase_end(residual_idx)
            if len(completed_blocks) == 1:
                logit_offset = phase_first_read - completed_logit_first_reads[0]
                phase_lse = completed_logits[0][logit_offset:phase_end - completed_logit_first_reads[0]].float()
                phase_outputs = completed_blocks[0].unsqueeze(0).expand(phase_lse.size(0), -1, -1, -1).unbind(0)
                phase_lses = phase_lse.unbind(0)
                return
            phase_logits = torch.stack(
                [
                    source_logits[phase_first_read - first_read:phase_end - first_read]
                    for source_logits, first_read in zip(completed_logits, completed_logit_first_reads)
                ],
                dim=1,
            ).contiguous()
            phase_output, phase_lse = attention_residual_phase1_from_logits(
                completed_blocks,
                phase_logits,
            )
            phase_outputs = phase_output.unbind(0)
            phase_lses = phase_lse.unbind(0)

        def read_residual(residual_idx):
            if residual_idx == 0:
                output = completed_blocks[0]
            else:
                ensure_phase(residual_idx)
                phase_idx = residual_idx - phase_first_read
                interblock_output = phase_outputs[phase_idx]
                if partial_block is None:
                    output = interblock_output
                else:
                    partial_source = self._attnres_block_summary(partial_block, partial_count, residual_idx)
                    query = query_bank[self._attnres_query_idx(residual_idx)].to(partial_source.dtype)
                    output = self._apply_training_phase2(
                        partial_source,
                        query,
                        interblock_output,
                        phase_lses[phase_idx],
                        normalize_output=fused_read_norm,
                        partial_count=partial_count,
                    )
            return norm(output) if fused_read_norm and (residual_idx == 0 or partial_block is None) else output

        def append_partial_if_block_end(residual_idx):
            nonlocal completed_blocks, completed_logit_first_reads, partial_block, partial_count
            if residual_idx not in block_end_set:
                return
            completed_source = self._attnres_block_summary(partial_block, partial_count, residual_idx)
            completed_blocks.append(completed_source)
            completed_logits.append(make_source_logits(completed_source, residual_idx, partial_count))
            completed_logit_first_reads.append(residual_idx)
            partial_block = None
            partial_count = 0
            invalidate_phase()

        def add_partial(layer_output):
            nonlocal partial_block, partial_count
            if partial_block is None:
                partial_block = layer_output
                partial_count = 1
            else:
                partial_block = partial_block + layer_output
                partial_count += 1

        for layer_idx, block in enumerate(self.transformer.layers):
            x = read_residual(2 * layer_idx)
            attn_out = block.forward_attention(
                x,
                past_kv=None,
                use_cache=False,
                cu_doc_len=cu_doc_len,
                max_doc_len=max_doc_len,
                x_is_normalized=fused_read_norm,
            )
            add_partial(attn_out)
            append_partial_if_block_end(2 * layer_idx + 1)

            x = read_residual(2 * layer_idx + 1)
            mlp_out = block.forward_mlp(x, x_is_normalized=fused_read_norm)
            add_partial(mlp_out)
            append_partial_if_block_end(2 * layer_idx + 2)

        x = read_residual(total_sublayers)
        if not fused_read_norm:
            x = norm(x)

        if return_hidden:
            return x

        if self.config.weight_tying:
            return F.linear(x, self.transformer.wte.weight, None)
        return self.lm_head(x)

    def _forward_block_lrid_attnres_fused_training(
        self,
        x,
        past_kv=None,
        use_cache=False,
        cu_doc_len=None,
        max_doc_len=None,
        return_hidden=False,
        fused_read_norm=False,
    ):
        if past_kv is not None or use_cache:
            raise NotImplementedError("KV-cache generation is not supported with attention residuals yet.")

        total_sublayers = 2 * self.config.n_layer
        block_ends = sorted(self.attnres_block_ends)
        block_end_set = self.attnres_block_ends
        key_dim = self.config.lrid_rank
        query_bank = torch.stack(
            [
                self.transformer.lrid_queries[self._attnres_query_idx(read_idx)].reshape(key_dim)
                for read_idx in range(1, total_sublayers + 1)
            ],
            dim=0,
        )
        if self.config.attn_res_query_norm:
            query_bank = norm(query_bank.float()).to(x.dtype)
        else:
            query_bank = query_bank.to(x.dtype)
        query_bank = query_bank.contiguous()

        def source_key(source):
            key = source[1].reshape(source[1].size(0), source[1].size(1), key_dim)
            if self.config.attnres_key_norm:
                key = norm(key.float()).to(source[0].dtype)
            return key

        def make_source_logits(source, first_read, source_count=1):
            key = source_key(source)
            queries = query_bank[first_read - 1:].to(key.dtype)
            logits = (
                F.linear(key, queries)
                .permute(2, 0, 1)
                .contiguous()
                * self.config.lrid_logit_scale
            )
            if self._use_attnres_block_count_prior() and source_count != 1:
                logits = logits + math.log(float(source_count))
            return logits

        embedding_source = self._embedding_lrid_source(x)
        completed_values = [embedding_source[0]]
        completed_logits = [make_source_logits(embedding_source, 1, 1)]
        completed_logit_first_reads = [1]
        partial_block = None
        partial_key = None
        partial_count = 0
        phase_first_read = None
        phase_end = None
        phase_outputs = None
        phase_lses = None

        def next_phase_end(residual_idx):
            for block_end in block_ends:
                if block_end > residual_idx:
                    return block_end
            return residual_idx + 1

        def invalidate_phase():
            nonlocal phase_first_read, phase_end, phase_outputs, phase_lses
            phase_first_read = None
            phase_end = None
            phase_outputs = None
            phase_lses = None

        def ensure_phase(residual_idx):
            nonlocal phase_first_read, phase_end, phase_outputs, phase_lses
            if residual_idx < 1:
                return
            if phase_outputs is not None and phase_first_read <= residual_idx < phase_end:
                return
            phase_first_read = residual_idx
            phase_end = next_phase_end(residual_idx)
            if len(completed_values) == 1:
                logit_offset = phase_first_read - completed_logit_first_reads[0]
                phase_lse = completed_logits[0][logit_offset:phase_end - completed_logit_first_reads[0]].float()
                phase_outputs = completed_values[0].unsqueeze(0).expand(phase_lse.size(0), -1, -1, -1).unbind(0)
                phase_lses = phase_lse.unbind(0)
                return
            phase_logits = torch.stack(
                [
                    source_logits[phase_first_read - first_read:phase_end - first_read]
                    for source_logits, first_read in zip(completed_logits, completed_logit_first_reads)
                ],
                dim=1,
            ).contiguous()
            phase_output, phase_lse = attention_residual_phase1_from_logits(
                completed_values,
                phase_logits,
            )
            phase_outputs = phase_output.unbind(0)
            phase_lses = phase_lse.unbind(0)

        def current_partial_source(summary_idx):
            return self._lrid_block_source(partial_block, partial_key, None, partial_count, summary_idx)

        def read_residual(residual_idx):
            if residual_idx == 0:
                output = completed_values[0]
            else:
                ensure_phase(residual_idx)
                phase_idx = residual_idx - phase_first_read
                interblock_output = phase_outputs[phase_idx]
                if partial_block is None:
                    output = interblock_output
                else:
                    partial_source = current_partial_source(residual_idx)
                    query = query_bank[self._attnres_query_idx(residual_idx)].to(partial_source[1].dtype)
                    output = self._apply_lrid_training_phase2(
                        partial_source[0],
                        partial_source[1].reshape(partial_source[1].size(0), partial_source[1].size(1), key_dim),
                        query,
                        interblock_output,
                        phase_lses[phase_idx],
                        normalize_output=fused_read_norm,
                        partial_count=partial_count,
                    )
            return norm(output) if fused_read_norm and (residual_idx == 0 or partial_block is None) else output

        def append_partial_if_block_end(residual_idx):
            nonlocal completed_logit_first_reads, partial_block, partial_key, partial_count
            if residual_idx not in block_end_set:
                return
            completed_source = current_partial_source(residual_idx)
            completed_values.append(completed_source[0])
            completed_logits.append(make_source_logits(completed_source, residual_idx, partial_count))
            completed_logit_first_reads.append(residual_idx)
            partial_block = None
            partial_key = None
            partial_count = 0
            invalidate_phase()

        def add_partial(layer_output, lrid_key):
            nonlocal partial_block, partial_key, partial_count
            if partial_block is None:
                partial_block = layer_output
                partial_key = lrid_key
                partial_count = 1
            else:
                partial_block = partial_block + layer_output
                partial_key = partial_key + lrid_key
                partial_count += 1

        for layer_idx, block in enumerate(self.transformer.layers):
            x = read_residual(2 * layer_idx)
            attn_out, lrid_key = block.forward_attention(
                x,
                past_kv=None,
                use_cache=False,
                cu_doc_len=cu_doc_len,
                max_doc_len=max_doc_len,
                x_is_normalized=fused_read_norm,
                emit_lrid_key=True,
            )
            add_partial(attn_out, lrid_key)
            append_partial_if_block_end(2 * layer_idx + 1)

            x = read_residual(2 * layer_idx + 1)
            mlp_out, lrid_key = block.forward_mlp(
                x,
                x_is_normalized=fused_read_norm,
                emit_lrid_key=True,
            )
            add_partial(mlp_out, lrid_key)
            append_partial_if_block_end(2 * layer_idx + 2)

        x = read_residual(total_sublayers)
        if not fused_read_norm:
            x = norm(x)

        if return_hidden:
            return x

        if self.config.weight_tying:
            return F.linear(x, self.transformer.wte.weight, None)
        return self.lm_head(x)

    def _use_block_attnres_fused_training_path(self, past_kv, use_cache):
        return (
            self.use_attnres
            and self.config.use_fused_attnres
            and self.config.attnres_training_cache_phase1
            and not self.use_lrid
            and self.attnres_type == "block"
            and past_kv is None
            and not use_cache
        )

    def _use_block_lrid_attnres_fused_training_path(self, past_kv, use_cache):
        return (
            self.use_attnres
            and self.config.use_fused_attnres
            and self.config.attnres_training_cache_phase1
            and self.use_lrid
            and self.attnres_type == "block"
            and past_kv is None
            and not use_cache
            and self.config.lrid_num_heads == 1
            and not self.config.lrid_input_dependent_query
            and not self.config.lrid_key_from_value
            and not self.config.lrid_key_from_value_shared
            and not self.config.lrid_query_from_value
            and not self.config.lrid_query_from_value_shared
        )

    def _project_lrid_source_key(self, value):
        if self.config.lrid_key_from_output_tail:
            return self._lrid_output_tail_key(value)
        return self.transformer.lrid_embedding_key(value)

    def _project_lrid_source_query(self, value):
        return self.transformer.lrid_embedding_key.project_query_from_value(value)

    def _static_lrid_embedding_key(self, embedding):
        return self.transformer.lrid_static_embedding_key(embedding)

    def _add_static_lrid_embedding_key(self, key):
        if self.config.lrid_add_static_embedding_key:
            key = key + self.transformer.lrid_static_embedding_key(key)
        return key

    def _add_static_lrid_source_key(self, key):
        if self.config.lrid_add_static_source_key:
            key = key + self.transformer.lrid_static_source_key(key)
        return key

    def _lrid_output_tail_key(self, value):
        return value[..., -self.config.lrid_rank:].contiguous()

    def _lrid_source(self, value, key=None, query=None, key_value=None, query_value=None, add_static_key=False):
        if key is None:
            if self.config.lrid_key_from_output_tail:
                key = self._lrid_output_tail_key(value)
            elif self.config.lrid_key_from_value_shared:
                key = self._project_lrid_source_key(value if key_value is None else key_value)
            else:
                raise RuntimeError("LR AttnRes source key is missing")
        if add_static_key:
            key = self._add_static_lrid_source_key(key)
        if self.config.lrid_input_dependent_query:
            if self.config.lrid_query_from_value_shared:
                query = self._project_lrid_source_query(value if query_value is None else query_value)
            return value, key, query
        return value, key

    def _embedding_lrid_source(self, embedding):
        if self.config.lrid_key_from_output_tail:
            key = self._lrid_output_tail_key(embedding)
        elif self.config.lrid_static_embedding_key:
            key = self._static_lrid_embedding_key(embedding)
        else:
            key = self._project_lrid_source_key(embedding)
            key = self._add_static_lrid_embedding_key(key)
        return self._lrid_source(embedding, key)

    def _apply_lrid_attnres(
        self,
        residual_idx,
        sources,
        query_override=None,
        normalize_output=False,
        average_read=False,
        source_counts=None,
    ):
        if len(sources) == 1:
            output = sources[0][0]
            return norm(output) if normalize_output else output

        value_sources = [source[0] for source in sources]
        if average_read:
            return attention_residual_average_read(value_sources, normalize_output=normalize_output)
        key_sources = [source[1] for source in sources]
        num_heads = self.config.lrid_num_heads
        key_head_dim = self.config.lrid_rank // num_heads
        value_head_dim = self.config.n_embd // num_heads

        query_idx = self._attnres_query_idx(residual_idx)
        static_query = self.transformer.lrid_queries[query_idx]
        if self.config.lrid_input_dependent_query:
            dynamic_query = query_override if query_override is not None else sources[-1][2]
            if dynamic_query is None:
                raise RuntimeError("Input-dependent LR AttnRes query is missing for the active source")
            dynamic_query = dynamic_query.reshape(*dynamic_query.shape[:-1], num_heads, key_head_dim)
            gate = self.transformer.lrid_query_gates[query_idx].view(1, 1, num_heads, 1)
            query = static_query.unsqueeze(0).unsqueeze(0) + gate * dynamic_query
        else:
            query = static_query

        if self.config.attn_res_query_norm:
            query = norm(query.float())

        if self.config.use_fused_attnres:
            return lrid_attention_residual_read(
                value_sources,
                key_sources,
                query,
                num_heads,
                self.config.lrid_logit_scale,
                self.config.attnres_key_norm,
                normalize_output=normalize_output,
                source_counts=source_counts,
            )

        values = torch.stack(value_sources, dim=0)
        keys = torch.stack(key_sources, dim=0)
        keys = keys.reshape(*keys.shape[:-1], num_heads, key_head_dim)
        values = values.reshape(*values.shape[:-1], num_heads, value_head_dim)
        if self.config.attnres_key_norm:
            keys = norm(keys.float()).to(values.dtype)

        query = query.to(keys.dtype)
        if self.config.lrid_input_dependent_query:
            logits = torch.einsum("sbthr,bthr->sbth", keys, query) * self.config.lrid_logit_scale
        else:
            logits = torch.einsum("sbthr,hr->sbth", keys, query) * self.config.lrid_logit_scale
        if source_counts is not None:
            log_counts = torch.as_tensor(source_counts, device=logits.device, dtype=torch.float32).log()
            logits = logits + log_counts.view(-1, 1, 1, 1)
        weights = F.softmax(logits.float(), dim=0).to(values.dtype)
        output = torch.einsum("sbth,sbthd->bthd", weights, values)
        output = output.reshape(output.size(0), output.size(1), self.config.n_embd)
        return norm(output) if normalize_output else output
    
    def _init_attnres_query(self, query, std=0.02, init_cutoff_factor=None):
        if self.config.attn_res_query_init == "zero":
            nn.init.zeros_(query)
        elif self.config.attn_res_query_init == "normal":
            nn.init.normal_(query, mean=0.0, std=std)
        elif self.config.attn_res_query_init == "trunc_normal":
            cutoff = (init_cutoff_factor if init_cutoff_factor is not None else 3.0) * std
            nn.init.trunc_normal_(query, mean=0.0, std=std, a=-cutoff, b=cutoff)

    def _init_lrid_dynamic_query_projection(self, module, std=0.02, init_cutoff_factor=None):
        if not module.use_fused_query:
            return
        query_weight = module.proj.weight[module.query_offset:]
        if init_cutoff_factor is not None:
            cutoff = init_cutoff_factor * std
            nn.init.trunc_normal_(query_weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
        else:
            nn.init.normal_(query_weight, mean=0.0, std=std)

    def _init_weights(self, module, std=0.02, init_cutoff_factor=None):
        if isinstance(module, nn.Linear):
            if init_cutoff_factor is not None:
                cutoff = init_cutoff_factor * std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, (nn.Embedding, TokenEmbedding)):
            if init_cutoff_factor is not None:
                cutoff = init_cutoff_factor * std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, LRIDStaticKey):
            module.reset_parameters(std, init_cutoff_factor)
        elif isinstance(module, AttentionResidual):
            self._init_attnres_query(module.query, std, init_cutoff_factor)

    def _sample_next_token(self, logits, temperature=1.0, top_k=None):
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1 when set")

        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def get_lm_head_weight(self):
        if self.config.weight_tying:
            return self.transformer.wte.weight
        return self.lm_head.weight

    def get_lm_head_bias(self):
        return None

    def forward(
        self,
        idx,
        past_kv=None,
        use_cache=False,
        cu_doc_len=None,
        max_doc_len=None,
        return_hidden=False,
    ):
        _, T = idx.size()
        assert T <= self.config.block_size, f"Token length {T} exceeds max sequence length {self.config.block_size}"
        
        x = self.transformer.wte(idx)
        fused_read_norm = (
            self.use_attnres
            and self.config.use_fused_attnres
            and self.config.norm_pos == "before"
            and self.config.attnres_fuse_read_norm
            and (not self.use_lrid or self.config.lrid_num_heads == 1)
        )
        if self._use_block_attnres_fused_training_path(past_kv, use_cache):
            return self._forward_block_attnres_fused_training(
                x,
                past_kv=past_kv,
                use_cache=use_cache,
                cu_doc_len=cu_doc_len,
                max_doc_len=max_doc_len,
                return_hidden=return_hidden,
                fused_read_norm=fused_read_norm,
            )
        if self._use_block_lrid_attnres_fused_training_path(past_kv, use_cache):
            return self._forward_block_lrid_attnres_fused_training(
                x,
                past_kv=past_kv,
                use_cache=use_cache,
                cu_doc_len=cu_doc_len,
                max_doc_len=max_doc_len,
                return_hidden=return_hidden,
                fused_read_norm=fused_read_norm,
            )
        if self.use_attnres:
            if past_kv is not None or use_cache:
                raise NotImplementedError("KV-cache generation is not supported with attention residuals yet.")
            embedding = x
            if self.use_lrid:
                embedding_source = self._embedding_lrid_source(embedding)
            if self.attnres_type == "full":
                residual_sources = [embedding_source] if self.use_lrid else [embedding]
            else:
                block_ends = self.attnres_block_ends
                completed_blocks = [embedding_source] if self.use_lrid else [embedding]
                completed_block_counts = [1]
                partial_block = None
                partial_count = 0
                if self.use_lrid:
                    partial_key = None
                    if self.config.lrid_input_dependent_query:
                        partial_query = None
        
        if past_kv is None:
            past_kv = [None] * len(self.transformer.layers)
        new_kv = [] if use_cache else None
        
        for layer_idx, block in enumerate(self.transformer.layers):
            if self.use_attnres:
                if self.use_lrid:
                    if self.attnres_type == "full":
                        x = self._apply_lrid_attnres(
                            2 * layer_idx,
                            residual_sources,
                            normalize_output=fused_read_norm,
                            average_read=False,
                        )
                    else:
                        attn_res_idx = 2 * layer_idx
                        sources = completed_blocks if partial_block is None else completed_blocks + [
                            self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count, attn_res_idx)
                        ]
                        source_counts = self._attnres_block_source_counts(
                            completed_block_counts,
                            None if partial_block is None else partial_count,
                        )
                        x = self._apply_lrid_attnres(
                            attn_res_idx,
                            sources,
                            normalize_output=fused_read_norm,
                            average_read=False,
                            source_counts=source_counts,
                        )

                    attn_out = block.forward_attention(
                        x,
                        past_kv=past_kv[layer_idx],
                        use_cache=False,
                        cu_doc_len=cu_doc_len,
                        max_doc_len=max_doc_len,
                        x_is_normalized=fused_read_norm,
                        emit_lrid_key=True,
                    )
                    if self.config.lrid_input_dependent_query:
                        attn_out, lrid_key, lrid_query = attn_out
                    else:
                        attn_out, lrid_key = attn_out
                    layer_output = attn_out

                    if self.attnres_type == "full":
                        residual_sources.append(
                            self._lrid_source(
                                layer_output,
                                lrid_key,
                                lrid_query if self.config.lrid_input_dependent_query else None,
                                add_static_key=True,
                            )
                        )
                        x = self._apply_lrid_attnres(
                            2 * layer_idx + 1,
                            residual_sources,
                            normalize_output=fused_read_norm,
                            average_read=False,
                        )
                    else:
                        after_attn_idx = 2 * layer_idx + 1
                        if partial_block is None:
                            partial_block = layer_output
                            partial_count = 1
                        else:
                            partial_block = partial_block + layer_output
                            partial_count += 1
                        block_source_value = self._attnres_block_summary(partial_block, partial_count, after_attn_idx)
                        if self.config.lrid_key_from_value_shared:
                            partial_key = None
                        elif self.config.lrid_key_from_value:
                            partial_key = block.attn.c_proj.project_key_from_value(block_source_value)
                        else:
                            partial_key = lrid_key if partial_count == 1 else partial_key + lrid_key
                        if self.config.lrid_input_dependent_query:
                            if self.config.lrid_query_from_value_shared:
                                partial_query = None
                            elif self.config.lrid_query_from_value:
                                partial_query = block.attn.c_proj.project_query_from_value(block_source_value)
                            else:
                                partial_query = lrid_query
                        is_block_end = after_attn_idx in block_ends
                        if is_block_end:
                            completed_blocks.append(
                                self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count, after_attn_idx)
                            )
                            completed_block_counts.append(partial_count)
                            partial_block = None
                            partial_key = None
                            partial_count = 0
                            if self.config.lrid_input_dependent_query:
                                partial_query = None
                        sources = completed_blocks if partial_block is None else completed_blocks + [
                            self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count, after_attn_idx)
                        ]
                        source_counts = self._attnres_block_source_counts(
                            completed_block_counts,
                            None if partial_block is None else partial_count,
                        )
                        x = self._apply_lrid_attnres(
                            after_attn_idx,
                            sources,
                            query_override=partial_query if self.config.lrid_input_dependent_query else None,
                            normalize_output=fused_read_norm,
                            average_read=False,
                            source_counts=source_counts,
                        )

                    mlp_out = block.forward_mlp(
                        x,
                        x_is_normalized=fused_read_norm,
                        emit_lrid_key=True,
                    )
                    if self.config.lrid_input_dependent_query:
                        mlp_out, lrid_key, lrid_query = mlp_out
                    else:
                        mlp_out, lrid_key = mlp_out
                    layer_output = mlp_out
                    x = mlp_out

                    if self.attnres_type == "full":
                        residual_sources.append(
                            self._lrid_source(
                                layer_output,
                                lrid_key,
                                lrid_query if self.config.lrid_input_dependent_query else None,
                                add_static_key=True,
                            )
                        )
                    else:
                        after_mlp_idx = 2 * layer_idx + 2
                        if partial_block is None:
                            partial_block = layer_output
                            partial_count = 1
                        else:
                            partial_block = partial_block + layer_output
                            partial_count += 1
                        block_source_value = self._attnres_block_summary(partial_block, partial_count, after_mlp_idx)
                        if self.config.lrid_key_from_value_shared:
                            partial_key = None
                        elif self.config.lrid_key_from_value:
                            partial_key = block.mlp.fc2.project_key_from_value(block_source_value)
                        else:
                            partial_key = lrid_key if partial_count == 1 else partial_key + lrid_key
                        if self.config.lrid_input_dependent_query:
                            if self.config.lrid_query_from_value_shared:
                                partial_query = None
                            elif self.config.lrid_query_from_value:
                                partial_query = block.mlp.fc2.project_query_from_value(block_source_value)
                            else:
                                partial_query = lrid_query
                        is_block_end = after_mlp_idx in block_ends
                        if is_block_end:
                            completed_blocks.append(
                                self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count, after_mlp_idx)
                            )
                            completed_block_counts.append(partial_count)
                            partial_block = None
                            partial_key = None
                            partial_count = 0
                            if self.config.lrid_input_dependent_query:
                                partial_query = None
                    continue
                elif self.attnres_type == "full":
                    x = self._apply_attnres(
                        2 * layer_idx,
                        residual_sources,
                        normalize_output=fused_read_norm,
                        average_read=False,
                    )
                else:
                    attn_res_idx = 2 * layer_idx
                    sources = completed_blocks if partial_block is None else completed_blocks + [self._attnres_block_summary(partial_block, partial_count, attn_res_idx)]
                    source_counts = self._attnres_block_source_counts(
                        completed_block_counts,
                        None if partial_block is None else partial_count,
                    )
                    x = self._apply_attnres(
                        2 * layer_idx,
                        sources,
                        normalize_output=fused_read_norm,
                        average_read=False,
                        source_counts=source_counts,
                    )

                attn_out = block.forward_attention(
                    x,
                    past_kv=past_kv[layer_idx],
                    use_cache=False,
                    cu_doc_len=cu_doc_len,
                    max_doc_len=max_doc_len,
                    x_is_normalized=fused_read_norm,
                )
                layer_output = attn_out

                if self.attnres_type == "full":
                    residual_sources.append(layer_output)
                    x = self._apply_attnres(
                        2 * layer_idx + 1,
                        residual_sources,
                        normalize_output=fused_read_norm,
                        average_read=False,
                    )
                else:
                    after_attn_idx = 2 * layer_idx + 1
                    if partial_block is None:
                        partial_block = layer_output
                        partial_count = 1
                    else:
                        partial_block = partial_block + layer_output
                        partial_count += 1
                    is_block_end = after_attn_idx in block_ends
                    if is_block_end:
                        completed_blocks.append(self._attnres_block_summary(partial_block, partial_count, after_attn_idx))
                        completed_block_counts.append(partial_count)
                        partial_block = None
                        partial_count = 0
                    sources = completed_blocks if partial_block is None else completed_blocks + [self._attnres_block_summary(partial_block, partial_count, after_attn_idx)]
                    source_counts = self._attnres_block_source_counts(
                        completed_block_counts,
                        None if partial_block is None else partial_count,
                    )
                    x = self._apply_attnres(
                        after_attn_idx,
                        sources,
                        normalize_output=fused_read_norm,
                        average_read=False,
                        source_counts=source_counts,
                    )

                mlp_out = block.forward_mlp(x, x_is_normalized=fused_read_norm)
                layer_output = mlp_out
                x = mlp_out

                if self.attnres_type == "full":
                    residual_sources.append(layer_output)
                else:
                    after_mlp_idx = 2 * layer_idx + 2
                    if partial_block is None:
                        partial_block = layer_output
                        partial_count = 1
                    else:
                        partial_block = partial_block + layer_output
                        partial_count += 1
                    is_block_end = after_mlp_idx in block_ends
                    if is_block_end:
                        completed_blocks.append(self._attnres_block_summary(partial_block, partial_count, after_mlp_idx))
                        completed_block_counts.append(partial_count)
                        partial_block = None
                        partial_count = 0
            else:
                block_out = block(
                    x,
                    past_kv=past_kv[layer_idx],
                    use_cache=use_cache,
                    cu_doc_len=cu_doc_len,
                    max_doc_len=max_doc_len,
                )

                if use_cache:
                    x, present_kv = block_out
                    new_kv.append(present_kv)
                else:
                    x = block_out

        if self.use_attnres:
            if self.use_lrid:
                if self.attnres_type == "full":
                    x = self._apply_lrid_attnres(
                        2 * self.config.n_layer,
                        residual_sources,
                        normalize_output=fused_read_norm,
                        average_read=False,
                    )
                else:
                    sources = completed_blocks if partial_block is None else completed_blocks + [
                        self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count, 2 * self.config.n_layer)
                    ]
                    source_counts = self._attnres_block_source_counts(
                        completed_block_counts,
                        None if partial_block is None else partial_count,
                    )
                    x = self._apply_lrid_attnres(
                        2 * self.config.n_layer,
                        sources,
                        normalize_output=fused_read_norm,
                        average_read=False,
                        source_counts=source_counts,
                    )
            elif self.attnres_type == "full":
                x = self._apply_attnres(
                    2 * self.config.n_layer,
                    residual_sources,
                    normalize_output=fused_read_norm,
                    average_read=False,
                )
            else:
                sources = completed_blocks if partial_block is None else completed_blocks + [self._attnres_block_summary(partial_block, partial_count, 2 * self.config.n_layer)]
                source_counts = self._attnres_block_source_counts(
                    completed_block_counts,
                    None if partial_block is None else partial_count,
                )
                x = self._apply_attnres(
                    2 * self.config.n_layer,
                    sources,
                    normalize_output=fused_read_norm,
                    average_read=False,
                    source_counts=source_counts,
                )
        
        if not fused_read_norm:
            x = norm(x)

        if return_hidden:
            if use_cache:
                return x, new_kv
            return x
        
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_kv
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, max_context=None):
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)
        _, T = idx.size()

        if max_context is None:
            max_context = self.config.block_size

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_context < 1:
            raise ValueError("max_context must be >= 1")

        if T > max_context:
            idx = idx[:, -max_context:]
            T = idx.size(1)

        if self.use_attnres or idx.size(1) + max_new_tokens > max_context:
            generated = idx
            for _ in range(max_new_tokens):
                idx_cond = generated[:, -max_context:]
                logits = self(idx_cond)
                logits = logits[:, -1, :]
                next_token = self._sample_next_token(logits, temperature=temperature, top_k=top_k)
                generated = torch.cat((generated, next_token), dim=1)

            return generated

        past_kv = None

        if T > 0:
            start = 0
            while start < T:
                end = min(start + self.config.block_size, T)
                idx_cond = idx[:, start:end]
                logits, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)
                start = end

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if idx.size(1) > 0 else idx
            logits, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)
            logits = logits[:, -1, :]
            next_token = self._sample_next_token(logits, temperature=temperature, top_k=top_k)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
