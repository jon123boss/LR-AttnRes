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
    act_type: str = "gelu"
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
    use_attnres: bool = False
    attnres_mode: str = "block"
    attnres_num_blocks: int = 6
    track_attnres: bool = False


class ZeroInitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=False)
    
    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()


class ActivationFunction(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        self.act_type = act_type.lower()
        if self.act_type == "relu":
            self.activation = nn.ReLU()
        elif self.act_type == "gelu":
            self.activation = nn.GELU()
        elif self.act_type == "silu":
            self.activation = nn.SiLU()
        elif self.act_type == "swiglu":
            self.activation = SwiGLU()
        elif self.act_type == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {act_type}")
    
    def forward(self, x):
        return self.activation(x)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


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
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
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
        self.c_proj = ZeroInitLinear(config.n_embd, config.n_embd)
        
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
        
        x = self.c_proj(attention_output)
        
        if use_cache:
            return x, (k, v)
        else:
            return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.mlp_hidden_dim if config.mlp_hidden_dim is not None else int(config.n_embd * config.mlp_ratio)
        self.act = ActivationFunction(config.act_type)
        self.fc1 = nn.Linear(config.n_embd, self.hidden_dim, bias=False)
        self.fc2 = ZeroInitLinear(self.hidden_dim // 2 if config.act_type.lower() == "swiglu" else self.hidden_dim, config.n_embd)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


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

        if config.use_attnres:
            # Zero-initialized pseudo-query vectors (paper: "Crucially, all pseudo-query
            # vectors must be initialized to zero" to ensure uniform attention at start)
            self.attn_res_query = nn.Parameter(torch.zeros(config.n_embd))
            self.mlp_res_query = nn.Parameter(torch.zeros(config.n_embd))
            # RMSNorm on keys to prevent magnitude differences from biasing softmax
            self.attn_res_norm = RMSNorm(config)
            self.mlp_res_norm = RMSNorm(config)

    def _compute_attnres(self, blocks, partial_block, query, norm, tracker=None, layer_idx=None, sublayer="attn"):
        """
        Compute Attention Residual (AttnRes) over block representations.
        blocks: list of [B, T, D] tensors (completed block reps + embedding b_0)
        partial_block: [B, T, D] or None (intra-block partial sum b_n^{i-1})
        query: [D] learned pseudo-query vector
        norm: RMSNorm applied to keys
        tracker: optional list to store tracked attention weights
        layer_idx: layer index for tracking
        sublayer: "attn" or "mlp" for tracking
        Returns: [B, T, D] AttnRes output
        """
        if partial_block is None:
            V = torch.stack(blocks)  # [N, B, T, D]
        else:
            V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
        K = norm(V)
        logits = torch.einsum("d, n b t d -> n b t", query, K)
        weights = logits.softmax(0)
        h = torch.einsum("n b t, n b t d -> b t d", weights, V)
        
        if tracker is not None:
            # Store detached mean weights: [num_sources] averaged over batch and sequence
            with torch.no_grad():
                tracker.append({
                    "layer_idx": layer_idx,
                    "sublayer": sublayer,
                    "weights": weights.mean(dim=(1, 2)).detach().cpu(),  # [num_sources]
                    "num_sources": weights.size(0),
                })
        
        return h

    def forward(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None):
        residual = x
        
        if self.norm_pos in {"before", "both"}:
            x = self.attn_norm(x)
        
        attn_out = self.attn(
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
        
        if self.norm_pos in {"after", "both"}:
            x = self.attn_norm(x)
        
        x = residual + x
        
        residual = x
        
        if self.norm_pos in {"before", "both"}:
            x = self.mlp_norm(x)
        
        x = self.mlp(x)
        
        if self.norm_pos in {"after", "both"}:
            x = self.mlp_norm(x)
        
        x = residual + x
        
        if use_cache:
            return x, new_kv
        else:
            return x


class OBPM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            layers=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            final_norm=RMSNorm(config)
        ))
        
        if not config.weight_tying:
            self.lm_head = ZeroInitLinear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(partial(self._init_weights, std=config.init_std, init_cutoff_factor=config.init_cutoff_factor))
    
    def to_mixed_precision(self, dtype=torch.bfloat16):
        self.to(dtype=dtype)
        return self
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module, std=0.02, init_cutoff_factor=None):
        if isinstance(module, ZeroInitLinear):
            return
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
    
    def forward(self, idx, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Token length {T} exceeds max sequence length {self.config.block_size}"
        
        x = self.transformer.wte(idx)
        
        if past_kv is None:
            past_kv = [None] * len(self.transformer.layers)
        new_kv = [] if use_cache else None
        
        if not self.config.use_attnres:
            # Standard residual path
            for layer_idx, block in enumerate(self.transformer.layers):
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
        else:
            # Attention Residuals (AttnRes) path
            # Follows paper Fig. 2 pseudocode exactly.
            # blocks already includes token embedding as b_0.
            blocks = [x]
            partial_block = None

            # Tracking buffer for attention weights
            attnres_tracker = [] if self.config.track_attnres else None

            # Compute block boundaries (Transformer layer indices where new blocks start)
            if self.config.attnres_mode == "block":
                total_layers = self.config.n_layer
                num_blocks = self.config.attnres_num_blocks
                base = total_layers // num_blocks
                rem = total_layers % num_blocks
                block_sizes = [base + 1] * rem + [base] * (num_blocks - rem)
                block_starts = set()
                cumsum = 0
                for size in block_sizes:
                    if cumsum > 0:
                        block_starts.add(cumsum)
                    cumsum += size
            else:
                # Full mode: every layer after 0 starts a new block
                block_starts = set(range(1, self.config.n_layer))
            
            for layer_idx, block in enumerate(self.transformer.layers):
                # ---- Attention sub-layer ----
                # Compute AttnRes input BEFORE boundary check (as in pseudocode)
                h = block._compute_attnres(
                    blocks, partial_block,
                    block.attn_res_query, block.attn_res_norm,
                    tracker=attnres_tracker, layer_idx=layer_idx, sublayer="attn"
                )
                
                # If reaches block boundary, start new block
                if layer_idx in block_starts:
                    blocks.append(partial_block)
                    partial_block = None
                
                if block.norm_pos in {"before", "both"}:
                    h = block.attn_norm(h)
                
                attn_out = block.attn(
                    h,
                    past_kv=past_kv[layer_idx],
                    use_cache=use_cache,
                    cu_doc_len=cu_doc_len,
                    max_doc_len=max_doc_len,
                )
                
                if use_cache:
                    attn_out, present_kv = attn_out
                    new_kv.append(present_kv)
                
                if block.norm_pos in {"after", "both"}:
                    attn_out = block.attn_norm(attn_out)
                
                partial_block = attn_out if partial_block is None else partial_block + attn_out
                
                # ---- MLP sub-layer ----
                h = block._compute_attnres(
                    blocks, partial_block,
                    block.mlp_res_query, block.mlp_res_norm,
                    tracker=attnres_tracker, layer_idx=layer_idx, sublayer="mlp"
                )
                
                if block.norm_pos in {"before", "both"}:
                    h = block.mlp_norm(h)
                
                mlp_out = block.mlp(h)
                
                if block.norm_pos in {"after", "both"}:
                    mlp_out = block.mlp_norm(mlp_out)
                
                partial_block = partial_block + mlp_out
            
            x = partial_block
            
            # Store tracker on model for external access
            if attnres_tracker is not None:
                self._last_attnres_tracker = attnres_tracker
        
        x = self.transformer.final_norm(x)
        
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_kv
        return logits
    
    def get_attnres_weights(self):
        """
        Retrieve tracked AttnRes attention weights from the last forward pass.
        Returns a dict mapping wandb-friendly keys to weight values, or None if not tracking.
        Format: {"attnres/layer_{l}_{sublayer}/source_{s}": weight, ...}
        """
        if not hasattr(self, "_last_attnres_tracker") or self._last_attnres_tracker is None:
            return None
        
        result = {}
        for entry in self._last_attnres_tracker:
            layer_idx = entry["layer_idx"]
            sublayer = entry["sublayer"]
            weights = entry["weights"]  # [num_sources]
            for src_idx in range(weights.size(0)):
                key = f"attnres/layer_{layer_idx}_{sublayer}/source_{src_idx}"
                result[key] = weights[src_idx].item()
        
        return result
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, max_context=None):
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)
        B, T = idx.size()

        if max_context is None:
            max_context = self.config.block_size

        if T > max_context:
            idx = idx[:, -max_context:]
            T = idx.size(1)

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
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if idx.size(1) > max_context:
                idx = idx[:, -max_context:]

        return idx
