# model.py
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from functools import partial
import math


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
    attnres_num_blocks: int = 8
    attnres_block_average: bool = False
    attnres_key_norm: bool = True
    attn_res_query_norm: bool = False
    attn_res_query_init: str = "zero"
    use_lrid: bool = False
    lrid_rank: int = 64
    lrid_num_heads: int = 1
    lrid_input_dependent_query: bool = False
    lrid_static_embedding_key: bool = False
    lrid_add_static_embedding_key: bool = False
    lrid_add_static_source_key: bool = False
    lrid_key_from_value: bool = False
    lrid_key_from_value_shared: bool = False
    lrid_key_value_norm: bool = True
    lrid_query_from_value: bool = False
    lrid_query_from_value_shared: bool = False
    lrid_use_logit_scale: bool = True
    lrid_logit_scale: float = None

    def __post_init__(self):
        self.attnres_type = (self.attnres_type or "block")
        self.attnres_type = self.attnres_type.lower()
        self.attn_res_query_init = (self.attn_res_query_init or "zero").lower()
        if self.attn_res_query_init not in {"zero", "normal", "trunc_normal"}:
            raise ValueError("attn_res_query_init must be one of: zero, normal, trunc_normal")
        if self.lrid_static_embedding_key and self.lrid_add_static_embedding_key:
            raise ValueError("lrid_static_embedding_key and lrid_add_static_embedding_key are mutually exclusive")
        if self.lrid_key_from_value_shared:
            self.lrid_key_from_value = True
        if self.lrid_query_from_value_shared:
            self.lrid_query_from_value = True
        if self.lrid_rank < 1:
            raise ValueError("lrid_rank must be >= 1")
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
        self.use_query = config.lrid_input_dependent_query
        self.use_value_key = config.lrid_key_from_value
        self.use_shared_value_key = config.lrid_key_from_value_shared
        self.use_local_value_key = self.use_value_key and not self.use_shared_value_key
        self.use_key = not self.use_value_key
        self.use_value_query = self.use_query and config.lrid_query_from_value
        self.use_shared_value_query = self.use_query and config.lrid_query_from_value_shared
        self.use_local_value_query = self.use_value_query and not self.use_shared_value_query
        self.use_fused_query = self.use_query and not self.use_value_query
        self.key_offset = output_dim
        self.query_offset = output_dim + (self.rank if self.use_key else 0)
        extra_dim = (self.rank if self.use_key else 0) + (self.rank if self.use_fused_query else 0)
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

    def forward(self, x):
        projected = self.proj(x)
        output = projected[..., :self.output_dim]
        key = None
        query = None
        if self.use_key:
            key = projected[..., self.key_offset:self.key_offset + self.rank]
            if self.use_key_norm:
                key = _norm_lrid_key(key, self.num_heads)
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
        uses_value_projection = (
            config.lrid_key_from_value
            or (config.lrid_input_dependent_query and config.lrid_query_from_value_shared)
        )
        self.use_value_norm = uses_value_projection and config.lrid_key_value_norm
        self.proj = nn.Linear(config.n_embd, config.lrid_rank, bias=False)
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

    def forward(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None):
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
        self.hidden_dim = config.mlp_hidden_dim if config.mlp_hidden_dim is not None else int(config.n_embd * config.mlp_ratio)
        self.fc1 = nn.Linear(config.n_embd, self.hidden_dim * 2, bias=False)
        if config.use_lrid:
            self.fc2 = LRIDFusedProjection(config, self.hidden_dim, config.n_embd)
        else:
            self.fc2 = nn.Linear(self.hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
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

    def forward(self, values):
        keys = norm(values) if self.use_key_norm else values
        logits = torch.einsum("d,sbtd->sbt", self._query(keys.dtype), keys)
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

    def forward_attention(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None):
        if self.norm_pos in {"before", "both"}:
            x = norm(x)

        attn_out = self.attn(x, past_kv=past_kv, use_cache=use_cache, cu_doc_len=cu_doc_len, max_doc_len=max_doc_len)

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

    def forward_mlp(self, x):
        if self.norm_pos in {"before", "both"}:
            x = norm(x)

        mlp_out = self.mlp(x)
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
                    not config.lrid_static_embedding_key
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

    def _attnres_query_idx(self, residual_idx):
        if residual_idx < 1:
            raise RuntimeError("Residual site 0 has no query because it only reads the embedding source")
        return residual_idx - 1

    def _attnres_block_summary(self, value, count):
        if self.config.attnres_block_average:
            return value / count
        return value

    def _lrid_block_source(self, value, key=None, query=None, count=1):
        if self.config.attnres_block_average:
            value = value / count
            if not self.config.lrid_key_from_value and not self.config.attnres_key_norm:
                key = key / count
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

    def _apply_attnres(self, residual_idx, sources):
        if len(sources) == 1:
            return sources[0]
        values = torch.stack(sources, dim=0)
        return self.transformer.attn_residuals[self._attnres_query_idx(residual_idx)](values)

    def _project_lrid_source_key(self, value):
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

    def _lrid_source(self, value, key=None, query=None, key_value=None, query_value=None, add_static_key=False):
        if key is None:
            if self.config.lrid_key_from_value_shared:
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
        if self.config.lrid_static_embedding_key:
            key = self._static_lrid_embedding_key(embedding)
        else:
            key = self._project_lrid_source_key(embedding)
            key = self._add_static_lrid_embedding_key(key)
        return self._lrid_source(embedding, key)

    def _apply_lrid_attnres(self, residual_idx, sources, query_override=None):
        if len(sources) == 1:
            return sources[0][0]

        values = torch.stack([source[0] for source in sources], dim=0)
        keys = torch.stack([source[1] for source in sources], dim=0)
        num_heads = self.config.lrid_num_heads
        key_head_dim = self.config.lrid_rank // num_heads
        value_head_dim = self.config.n_embd // num_heads

        keys = keys.reshape(*keys.shape[:-1], num_heads, key_head_dim)
        values = values.reshape(*values.shape[:-1], num_heads, value_head_dim)
        if self.config.attnres_key_norm:
            keys = norm(keys.float()).to(values.dtype)

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
        query = query.to(keys.dtype)
        if self.config.lrid_input_dependent_query:
            logits = torch.einsum("sbthr,bthr->sbth", keys, query) * self.config.lrid_logit_scale
        else:
            logits = torch.einsum("sbthr,hr->sbth", keys, query) * self.config.lrid_logit_scale
        weights = F.softmax(logits.float(), dim=0).to(values.dtype)
        output = torch.einsum("sbth,sbthd->bthd", weights, values)
        return output.reshape(output.size(0), output.size(1), self.config.n_embd)
    
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
                        x = self._apply_lrid_attnres(2 * layer_idx, residual_sources)
                    else:
                        attn_res_idx = 2 * layer_idx
                        sources = completed_blocks if partial_block is None else completed_blocks + [self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count)]
                        x = self._apply_lrid_attnres(attn_res_idx, sources)

                    attn_out = block.forward_attention(
                        x,
                        past_kv=past_kv[layer_idx],
                        use_cache=False,
                        cu_doc_len=cu_doc_len,
                        max_doc_len=max_doc_len,
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
                        x = self._apply_lrid_attnres(2 * layer_idx + 1, residual_sources)
                    else:
                        after_attn_idx = 2 * layer_idx + 1
                        if partial_block is None:
                            partial_block = layer_output
                            partial_count = 1
                        else:
                            partial_block = partial_block + layer_output
                            partial_count += 1
                        block_source_value = self._attnres_block_summary(partial_block, partial_count)
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
                            completed_blocks.append(self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count))
                            partial_block = None
                            partial_key = None
                            partial_count = 0
                            if self.config.lrid_input_dependent_query:
                                partial_query = None
                        sources = completed_blocks if partial_block is None else completed_blocks + [self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count)]
                        x = self._apply_lrid_attnres(
                            after_attn_idx,
                            sources,
                            query_override=partial_query if self.config.lrid_input_dependent_query else None,
                        )

                    mlp_out = block.forward_mlp(x)
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
                        block_source_value = self._attnres_block_summary(partial_block, partial_count)
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
                            completed_blocks.append(self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count))
                            partial_block = None
                            partial_key = None
                            partial_count = 0
                            if self.config.lrid_input_dependent_query:
                                partial_query = None
                    continue
                elif self.attnres_type == "full":
                    x = self._apply_attnres(2 * layer_idx, residual_sources)
                else:
                    attn_res_idx = 2 * layer_idx
                    sources = completed_blocks if partial_block is None else completed_blocks + [self._attnres_block_summary(partial_block, partial_count)]
                    x = self._apply_attnres(2 * layer_idx, sources)

                attn_out = block.forward_attention(
                    x,
                    past_kv=past_kv[layer_idx],
                    use_cache=False,
                    cu_doc_len=cu_doc_len,
                    max_doc_len=max_doc_len,
                )
                layer_output = attn_out

                if self.attnres_type == "full":
                    residual_sources.append(layer_output)
                    x = self._apply_attnres(2 * layer_idx + 1, residual_sources)
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
                        completed_blocks.append(self._attnres_block_summary(partial_block, partial_count))
                        partial_block = None
                        partial_count = 0
                    sources = completed_blocks if partial_block is None else completed_blocks + [self._attnres_block_summary(partial_block, partial_count)]
                    x = self._apply_attnres(after_attn_idx, sources)

                mlp_out = block.forward_mlp(x)
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
                        completed_blocks.append(self._attnres_block_summary(partial_block, partial_count))
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
                    x = self._apply_lrid_attnres(2 * self.config.n_layer, residual_sources)
                else:
                    sources = completed_blocks if partial_block is None else completed_blocks + [self._lrid_block_source(partial_block, partial_key, partial_query if self.config.lrid_input_dependent_query else None, partial_count)]
                    x = self._apply_lrid_attnres(2 * self.config.n_layer, sources)
            elif self.attnres_type == "full":
                x = self._apply_attnres(2 * self.config.n_layer, residual_sources)
            else:
                sources = completed_blocks if partial_block is None else completed_blocks + [self._attnres_block_summary(partial_block, partial_count)]
                x = self._apply_attnres(2 * self.config.n_layer, sources)
        
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
