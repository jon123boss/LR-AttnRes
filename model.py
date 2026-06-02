# model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from functools import partial
import math

@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 57601 
    n_layer: int = 12 
    n_head: int = 12
    n_embd: int = 768
    mlp_hidden_dim: int = None
    mlp_ratio: float = 4.0
    weight_tying: bool = False
    rope_theta: float = 500000.0
    rmsnorm_eps: float = 1e-6
    rmsnorm_use_weight: bool = True
    rmsnorm_use_bias: bool = False
    norm_pos: str = "after"
    qk_norm: bool = True
    clip_qkv: float = None
    flash_attention: bool = False
    init_std: float = 0.02
    init_cutoff_factor: float = None
    attnres_type: str = None
    use_attnres: bool = False
    attnres_num_blocks: int = 8
    use_lrid: bool = False
    lrid_rank: int = 64
    lrid_init: str = "zero_query"
    lrid_logit_scale: float = None

    def __post_init__(self):
        self.attnres_type = (self.attnres_type or "block")
        self.attnres_type = self.attnres_type.lower()
        self.lrid_init = (self.lrid_init or "zero_query").lower()
        if self.lrid_rank < 1:
            raise ValueError("lrid_rank must be >= 1")
        if self.lrid_init not in {"zero_query", "normal"}:
            raise ValueError("lrid_init must be one of: zero_query, normal")
        if self.lrid_logit_scale is None:
            self.lrid_logit_scale = 1.0 / math.sqrt(self.lrid_rank)
        elif self.lrid_logit_scale <= 0.0:
            raise ValueError("lrid_logit_scale must be positive")
        if self.use_lrid:
            self.use_attnres = True

class RMSNorm(nn.Module):
    def __init__(self, config, dim=None):
        super().__init__()
        self.eps = config.rmsnorm_eps
        dim = dim if dim is not None else config.n_embd

        if config.rmsnorm_use_weight:
            self.weight = nn.Parameter(torch.ones(dim))
            if config.rmsnorm_use_bias:
                self.bias = nn.Parameter(torch.zeros(dim))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x_norm.to(orig_dtype)

        if self.weight is not None:
            x_norm = x_norm * self.weight.to(x_norm.dtype)
        if self.bias is not None:
            x_norm = x_norm + self.bias.to(x_norm.dtype)

        return x_norm


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

    def forward(self, x, offset=0):
        T = x.size(-2)
        sin = self.sin[:, :, offset:offset + T]
        cos = self.cos[:, :, offset:offset + T]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1).flatten(-2)


class LRIDFusedProjection(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.rank = config.lrid_rank
        self.proj = nn.Linear(input_dim, output_dim + 2 * self.rank, bias=False)
        self.query_norm = RMSNorm(config, dim=self.rank)
        self.key_norm = RMSNorm(config, dim=self.rank)

    def forward(self, x):
        output, query, key = self.proj(x).split(
            (self.output_dim, self.rank, self.rank),
            dim=-1,
        )
        return output, self.query_norm(query), self.key_norm(key)

    @torch.no_grad()
    def init_lrid(self, mode):
        if mode == "zero_query":
            self.proj.weight[self.output_dim:self.output_dim + self.rank].zero_()


class LRIDSourceKeyProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.n_embd, config.lrid_rank, bias=False)
        self.key_norm = RMSNorm(config, dim=config.lrid_rank)

    def forward(self, x):
        return self.key_norm(self.proj(x))

    @torch.no_grad()
    def init_lrid(self, mode):
        pass


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
        
        self.q_norm = RMSNorm(config, dim=self.head_dim) if config.qk_norm else None
        self.k_norm = RMSNorm(config, dim=self.head_dim) if config.qk_norm else None
        
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
        
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            pos_offset = past_k.size(-2)
        else:
            pos_offset = 0
        
        q = self.rope(q, offset=pos_offset)
        k = self.rope(k, offset=pos_offset)
        
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
            x, lrid_query, lrid_key = projected
        else:
            x = projected
        
        if use_cache:
            if self.config.use_lrid:
                return x, (k, v), lrid_query, lrid_key
            return x, (k, v)
        if self.config.use_lrid:
            return x, lrid_query, lrid_key
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
        self.norm = RMSNorm(config)
        self.query = nn.Parameter(torch.empty(config.n_embd))

    def forward(self, values):
        keys = self.norm(values)
        logits = torch.einsum("d,sbtd->sbt", self.query.to(keys.dtype), keys)
        weights = F.softmax(logits, dim=0).to(values.dtype)
        return torch.einsum("sbt,sbtd->btd", weights, values)


class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.norm_pos = config.norm_pos
        self.attn_norm = RMSNorm(config)
        self.attn = MultiHeadAttention(config, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx
        self.config = config

    def forward_attention(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None):
        if self.norm_pos in {"before", "both"}:
            x = self.attn_norm(x)

        attn_out = self.attn(x, past_kv=past_kv, use_cache=use_cache, cu_doc_len=cu_doc_len, max_doc_len=max_doc_len)

        if self.config.use_lrid:
            if use_cache:
                x, new_kv, lrid_query, lrid_key = attn_out
            else:
                x, lrid_query, lrid_key = attn_out
                new_kv = None
        elif use_cache:
            x, new_kv = attn_out
        else:
            x = attn_out
            new_kv = None

        if self.norm_pos in {"after", "both"}:
            x = self.attn_norm(x)

        if use_cache:
            if self.config.use_lrid:
                return x, new_kv, lrid_query, lrid_key
            return x, new_kv
        if self.config.use_lrid:
            return x, lrid_query, lrid_key
        return x

    def forward_mlp(self, x):
        if self.norm_pos in {"before", "both"}:
            x = self.mlp_norm(x)

        mlp_out = self.mlp(x)
        if self.config.use_lrid:
            x, lrid_query, lrid_key = mlp_out
        else:
            x = mlp_out

        if self.norm_pos in {"after", "both"}:
            x = self.mlp_norm(x)

        if self.config.use_lrid:
            return x, lrid_query, lrid_key
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
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        
        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            layers=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            final_norm=RMSNorm(config)
        )

        if self.use_attnres:
            if self.attnres_type not in {"full", "block"}:
                raise ValueError("attnres_type must be 'full' or 'block'")
            if self.attnres_type == "block" and config.attnres_num_blocks < 1:
                raise ValueError("attnres_num_blocks must be >= 1 when using block AttnRes")
            if self.use_lrid:
                transformer_modules["lrid_embedding_key"] = LRIDSourceKeyProjection(config)
            else:
                transformer_modules["attn_residuals"] = nn.ModuleList(
                    [AttentionResidual(config) for _ in range(2 * config.n_layer + 1)]
                )

        self.transformer = nn.ModuleDict(transformer_modules)
        
        if not config.weight_tying:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(partial(self._init_weights, std=config.init_std, init_cutoff_factor=config.init_cutoff_factor))
        if self.use_lrid:
            self._init_lrid_weights()
    
    def to_mixed_precision(self, dtype=torch.bfloat16):
        self.to(dtype=dtype)
        return self
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _attnres_block_size(self):
        if not self.use_attnres or self.attnres_type != "block":
            return None
        return math.ceil((2 * self.config.n_layer) / self.config.attnres_num_blocks)

    def _apply_attnres(self, residual_idx, sources):
        values = torch.stack(sources, dim=0)
        return self.transformer.attn_residuals[residual_idx](values)

    def _embedding_lrid_source(self, embedding):
        return embedding, self.transformer.lrid_embedding_key(embedding)

    def _apply_lrid_attnres(self, sources, query):
        if len(sources) == 1 or query is None:
            return sources[0][0]

        values = torch.stack([value for value, _ in sources], dim=0)
        keys = torch.stack([key for _, key in sources], dim=0)
        keys_float = keys.float()
        keys = (
            keys_float
            * torch.rsqrt(keys_float.pow(2).mean(dim=-1, keepdim=True) + self.config.rmsnorm_eps)
        ).to(values.dtype)
        logits = torch.einsum("sbtr,btr->sbt", keys, query.to(keys.dtype)) * self.config.lrid_logit_scale
        weights = F.softmax(logits.float(), dim=0).to(values.dtype)
        return torch.einsum("sbt,sbtd->btd", weights, values)

    def _init_lrid_weights(self):
        for module in self.modules():
            if isinstance(module, (LRIDFusedProjection, LRIDSourceKeyProjection)):
                module.init_lrid(self.config.lrid_init)
    
    def _init_weights(self, module, std=0.02, init_cutoff_factor=None):
        if isinstance(module, nn.Linear):
            if init_cutoff_factor is not None:
                cutoff = init_cutoff_factor * std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            if init_cutoff_factor is not None:
                cutoff = init_cutoff_factor * std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, AttentionResidual):
            nn.init.zeros_(module.query)

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
    
    def forward(self, idx, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None):
        _, T = idx.size()
        assert T <= self.config.block_size, f"Token length {T} exceeds max sequence length {self.config.block_size}"
        
        x = self.transformer.wte(idx)
        if self.use_attnres:
            if past_kv is not None or use_cache:
                raise NotImplementedError("KV-cache generation is not supported with attention residuals yet.")
            embedding = x
            if self.use_lrid:
                embedding_source = self._embedding_lrid_source(embedding)
                prev_lrid_query = None
            if self.attnres_type == "full":
                residual_sources = [embedding_source] if self.use_lrid else [embedding]
            else:
                block_size = self._attnres_block_size()
                completed_blocks = [embedding_source] if self.use_lrid else [embedding]
                partial_block = None
                if self.use_lrid:
                    partial_key = None
        
        if past_kv is None:
            past_kv = [None] * len(self.transformer.layers)
        new_kv = [] if use_cache else None
        
        for layer_idx, block in enumerate(self.transformer.layers):
            if self.use_attnres:
                if self.use_lrid:
                    if self.attnres_type == "full":
                        x = embedding if layer_idx == 0 else self._apply_lrid_attnres(residual_sources, prev_lrid_query)
                    else:
                        if layer_idx == 0:
                            x = embedding
                        else:
                            attn_res_idx = 2 * layer_idx
                            in_block_idx = attn_res_idx % block_size
                            sources = completed_blocks if in_block_idx == 0 else completed_blocks + [(partial_block, partial_key)]
                            x = self._apply_lrid_attnres(sources, prev_lrid_query)

                    attn_out, lrid_query, lrid_key = block.forward_attention(
                        x,
                        past_kv=past_kv[layer_idx],
                        use_cache=False,
                        cu_doc_len=cu_doc_len,
                        max_doc_len=max_doc_len,
                    )
                    layer_output = attn_out

                    if self.attnres_type == "full":
                        residual_sources.append((layer_output, lrid_key))
                        x = self._apply_lrid_attnres(residual_sources, lrid_query)
                    else:
                        if partial_block is None:
                            partial_block = layer_output
                            partial_key = lrid_key
                        else:
                            partial_block = partial_block + layer_output
                            partial_key = partial_key + lrid_key
                        sources = completed_blocks if (2 * layer_idx + 1) % block_size == 0 else completed_blocks + [(partial_block, partial_key)]
                        x = self._apply_lrid_attnres(sources, lrid_query)

                    mlp_out, lrid_query, lrid_key = block.forward_mlp(x)
                    layer_output = mlp_out
                    x = mlp_out
                    prev_lrid_query = lrid_query

                    if self.attnres_type == "full":
                        residual_sources.append((layer_output, lrid_key))
                    else:
                        partial_block = layer_output if partial_block is None else partial_block + layer_output
                        partial_key = lrid_key if partial_key is None else partial_key + lrid_key
                        is_block_end = ((2 * layer_idx + 2) % block_size == 0) or (layer_idx + 1 == self.config.n_layer)
                        if is_block_end:
                            completed_blocks.append((partial_block, partial_key))
                            partial_block = None
                            partial_key = None
                    continue
                elif self.attnres_type == "full":
                    x = self._apply_attnres(2 * layer_idx, residual_sources)
                else:
                    attn_res_idx = 2 * layer_idx
                    in_block_idx = attn_res_idx % block_size
                    sources = completed_blocks if in_block_idx == 0 else completed_blocks + [partial_block]
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
                    partial_block = layer_output if partial_block is None else partial_block + layer_output
                    sources = completed_blocks if (2 * layer_idx + 1) % block_size == 0 else completed_blocks + [partial_block]
                    x = self._apply_attnres(2 * layer_idx + 1, sources)

                mlp_out = block.forward_mlp(x)
                layer_output = mlp_out
                x = mlp_out

                if self.attnres_type == "full":
                    residual_sources.append(layer_output)
                else:
                    partial_block = layer_output if partial_block is None else partial_block + layer_output
                    is_block_end = ((2 * layer_idx + 2) % block_size == 0) or (layer_idx + 1 == self.config.n_layer)
                    if is_block_end:
                        completed_blocks.append(partial_block)
                        partial_block = None
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
                    x = self._apply_lrid_attnres(residual_sources, prev_lrid_query)
                else:
                    sources = completed_blocks if partial_block is None else completed_blocks + [(partial_block, partial_key)]
                    x = self._apply_lrid_attnres(sources, prev_lrid_query)
            elif self.attnres_type == "full":
                x = self._apply_attnres(2 * self.config.n_layer, residual_sources)
            else:
                sources = completed_blocks if partial_block is None else completed_blocks + [partial_block]
                x = self._apply_attnres(2 * self.config.n_layer, sources)
        
        x = self.transformer.final_norm(x)
        
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
