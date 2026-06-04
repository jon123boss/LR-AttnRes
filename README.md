## Download Repository

```bash
git clone https://github.com/jon123boss/LR-AttnRes
cd LR-AttnRes
```

## Prerequisites

Install required dependencies via pip:

```bash
pip install flash-attn --no-build-isolation
pip install tiktoken
pip install huggingface-hub
pip install datasets
pip install lm_eval
pip install hf_transfer
pip install wandb  # Optional, for experiment tracking
pip install liger-kernel  # Optional, for Liger kernel speed/memory ablations
```

## Data Preparation

This repo now defaults to GPT-4-tokenized Ultra-FineWeb-en shards:

- tokenizer: `tiktoken.encoding_for_model("gpt-4")` / `cl100k_base`
- vocab size: `100277`
- document separator token: `100257`
- shard dtype: `uint32`

Prepare 20B total tokens once and optionally upload them to your Hugging Face
dataset repo:

```bash
python prepare_ultrafineweb.py \
  --hf-repo-id <your-hf-username>/Ultra-FineWeb-en-20B-gpt4 \
  --upload
```

After the shards are uploaded, download them for training:

```bash
python prepdata.py --repo-id <your-hf-username>/Ultra-FineWeb-en-20B-gpt4
```

## LR AttnRes

LR AttnRes can be enabled as a block Attention Residuals variant:

```bash
python train.py --use_lrid --attnres_type block --lrid_rank 64
```

`--use_lrid` automatically enables `use_attnres`. LR AttnRes uses the same learned,
input-independent depth queries as normal Attention Residuals, but routes over
low-rank input-dependent source keys. An optional ablation can also emit
input-dependent depth queries with `--lrid_input_dependent_query`, changing LR
output projections from `d + k` to `d + 2k`; this uses a gated hybrid query
`static_query + gate * dynamic_query`. Depth routing can be split into multiple
heads with `--lrid_num_heads`; `lrid_rank` remains the total low-rank width.
Use `--lrid_static_embedding_key` to make the embedding source key a learned,
input-independent LR key instead of projecting it from token embeddings.
Use `--lrid_add_static_embedding_key` or `--lrid_add_static_source_key` to add
a learned static key to the computed embedding key or computed non-embedding
source keys.
Use `--lrid_key_from_value` to project LR keys from the source value or block
summary instead of fusing them into each output projection. This is unshared by
default, keeping a separate value-key projector per LR output module;
`--lrid_key_from_value_shared` uses one shared source-key projection. Use
`--lrid_query_from_value` to do the same for dynamic queries, with
`--lrid_query_from_value_shared` for the shared variant. Outside key/query
projections use stateless `rms_norm(source_value)` by default; disable it with
`--no-lrid_key_value_norm`.
Logit scaling defaults to `1 / sqrt(lrid_rank / lrid_num_heads)`;
disable it with `--no-lrid_logit_scale` or set it explicitly with `--lrid_logit_scale`.
Attention Residual key normalization, query normalization, and query initialization
are configurable via `--attnres_key_norm`, `--attn_res_query_norm`, and
`--attn_res_query_init`. Block summaries can be averaged by their sublayer count
with `--attnres_block_average`.

See [LR_ATTNRES.md](LR_ATTNRES.md) for the full design note, parameter cost,
stability rationale, and experiment matrix.

## Liger Kernel Toggles

Liger kernels are optional and default off. Enable all implemented kernels with:

```bash
python train.py --use_liger_kernels
```

Each kernel can also be toggled independently:

```bash
python train.py \
  --liger_rms_norm \
  --liger_rope \
  --liger_swiglu \
  --liger_cross_entropy \
  --liger_fused_linear_cross_entropy \
  --liger_embedding \
  --liger_attnres
```

Use `--no-liger_<name>` to subtract one from the master switch, for example
`--use_liger_kernels --no-liger_rope`. `--liger_strict` raises instead of
falling back when a requested Liger kernel is unavailable.
