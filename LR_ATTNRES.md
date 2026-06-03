# LR AttnRes: Low-Rank Attention Residuals

Date: 2026-06-03
Repo: `/Users/jonathansu/Documents/GitHub/LRID`

## Summary

LR AttnRes is the new name for the stabilized LRID line of experiments. It is a low-rank variant of Attention Residuals that keeps normal AttnRes-style learned depth queries, but replaces hidden-size source keys with low-rank, input-dependent source keys.

The current behavior is:

```text
depth query: learned input-independent parameter, one per AttnRes depth site
source key: low-rank input-dependent key emitted by each sublayer output projection
source value: normal hidden-size sublayer output
```

Recommended first run:

```bash
python train.py --use_lrid --attnres_type block --lrid_rank 64
```

To disable LR AttnRes logit scaling:

```bash
python train.py --use_lrid --attnres_type block --lrid_rank 64 --no-lrid_logit_scale
```

The code still uses `use_lrid` flag names for compatibility with the repo, but the architecture name should now be LR AttnRes.

## Motivation

Attention Residuals replace fixed residual accumulation with attention over prior depth sources. A later sublayer can choose how much to read from the embedding, previous attention outputs, and previous MLP outputs.

Standard residuals are fixed. Static Attention Residuals are learned, but their depth queries are input-independent. The original LRID idea tried to make both the query and key input-dependent through low-rank projections.

In practice, the input-dependent query path was unstable. LR AttnRes removes that path and keeps the lower-risk part:

- low-rank input-dependent source keys
- learned static depth queries
- normal hidden-size source values

This keeps the model’s ability to route differently based on token/source content, while avoiding a moving low-rank query projection at every sublayer.

## Architecture

When `use_lrid=True`, LR AttnRes replaces the normal output projection wrappers with low-rank key-emitting wrappers.

Attention output projection:

```text
c_proj: d -> d + k
```

MLP output projection:

```text
fc2: hidden -> d + k
```

The projection output is split into:

```text
output_d: normal sublayer output
key_k:    low-rank source key
```

The token embedding is also a depth source, so it gets its own low-rank key projection:

```text
embedding_key = Linear(d, k)(embedding)
```

For every Attention Residual depth site, LR AttnRes has a learned query:

```text
q_r in R^k
```

There are:

```text
2 * n_layer + 1
```

query parameters, matching the number of depth-aggregation sites in the current AttnRes implementation.

## Routing Formula

For depth site `r`:

```text
values = stack(source_value_i)
keys = stack(source_key_i)
keys = RMSNorm(keys)
query = q_r
logits_i = scale * dot(keys_i, query)
weights_i = softmax_i(logits)
output = sum_i weights_i * values_i
```

The low-rank source keys are input-dependent. The query is learned and input-independent.

## Logit Scale Toggle

LR AttnRes has an optional logit scale.

Default:

```text
lrid_use_logit_scale = True
lrid_logit_scale = 1 / sqrt(lrid_rank)
```

For `lrid_rank=64`, the default scale is:

```text
0.125
```

Disable scaling:

```bash
--no-lrid_logit_scale
```

When disabled, the effective scale is:

```text
1.0
```

Set a custom scale:

```bash
--lrid_logit_scale 0.0625
```

Use the toggle because it is not yet obvious whether the scale is necessary once the query path is static. The scale is useful for conservative stability; the unscaled path may be worth testing because static zero-initialized queries start with zero logits anyway.

## Initialization

LR AttnRes queries are zero-initialized:

```text
q_r = 0
```

At step 0, all depth logits are zero, so depth routing is uniform over available sources. This mirrors normal Attention Residuals and avoids the instability from computed low-rank query projections.

There is no `lrid_init` setting anymore because LR AttnRes no longer has a computed query branch to initialize.

Attention Residual query initialization is configurable:

```bash
--attn_res_query_init zero
--attn_res_query_init normal
--attn_res_query_init trunc_normal
```

`zero` preserves uniform depth routing at step 0. `normal` and `trunc_normal`
start with non-uniform routing and are useful ablations.

Key normalization and query normalization are also configurable:

```bash
--attnres_key_norm
--no-attnres_key_norm
--attn_res_query_norm
--no-attn_res_query_norm
```

These toggles apply to both static Attention Residuals and LR AttnRes. Key
normalization controls whether source keys are RMS-normalized before depth
attention. Query normalization controls whether the learned depth query is
RMS-normalized before scoring.

## Full vs Block

Full LR AttnRes attends over all previous sublayer sources. It is the most expressive and the most expensive.

Block LR AttnRes compresses prior sublayer outputs into block summaries. It is the recommended default.

Recommended:

```bash
python train.py \
  --use_lrid \
  --attnres_type block \
  --attnres_num_blocks 8 \
  --lrid_rank 64
```

Full-path experiment:

```bash
python train.py \
  --use_lrid \
  --attnres_type full \
  --lrid_rank 64
```

Unscaled experiment:

```bash
python train.py \
  --use_lrid \
  --attnres_type block \
  --lrid_rank 64 \
  --no-lrid_logit_scale
```

## Parameter Cost

Let:

```text
d = model hidden size
h = MLP hidden size
k = lrid_rank
L = number of transformer layers
```

Per layer, LR AttnRes adds:

```text
attention key overhead = k * d
MLP key overhead       = k * h
```

Once per model, it adds:

```text
embedding key overhead = k * d
depth query overhead   = (2L + 1) * k
```

For the current default:

```text
d = 768
h = 2048
k = 64
L = 12
```

Approximate added parameters:

```text
per layer = 64 * 768 + 64 * 2048
          = 49,152 + 131,072
          = 180,224

12 layers = 2,162,688
embedding key = 49,152
queries = 25 * 64 = 1,600

total extra = 2,213,440
```

This is about half the old LRID overhead because the old computed-query projection branch was removed.

## Stability Notes

The unstable LRID path computed a query from every sublayer output. That created large gradient norms in training. Static-query LR AttnRes removes that computed query branch.

The last smoke diagnostic for static-query LR AttnRes reported:

```text
static-query lrid grad_norm = 3.8204
query_grad_norm             = 0.0303
```

The query gradient being nonzero confirms that the learned static depth queries train.

The printed `grad_norm` in training is the value returned by `clip_grad_norm_`, which is the pre-clipping norm. Large printed values do not mean the update is that large, but extreme values are still an instability signal.

## Current Config Surface

Model config:

```text
use_lrid: bool
attnres_key_norm: bool
attn_res_query_norm: bool
attn_res_query_init: "zero" | "normal" | "trunc_normal"
lrid_rank: int
lrid_use_logit_scale: bool
lrid_logit_scale: float | None
```

Training CLI:

```bash
--use_lrid
--no-use_lrid
--attnres_key_norm
--no-attnres_key_norm
--attn_res_query_norm
--no-attn_res_query_norm
--attn_res_query_init
--lrid_rank
--lrid_use_logit_scale
--no-lrid_use_logit_scale
--no-lrid_logit_scale
--lrid_logit_scale
```

`--no-lrid_logit_scale` is an alias for disabling `lrid_use_logit_scale`.

## What To Log

Already logged:

```text
grad_norm
train/step_loss
train/loss
val/loss
tokens_per_s
ms_per_step
model/num_params
```

Recommended future LR AttnRes-specific logs:

```text
depth attention entropy
mean embedding-source weight
mean completed-block weight
mean partial-block weight
LR query grad norm
LR key grad norm
effective lrid_logit_scale
```

## Experiment Matrix

Start with:

```text
baseline
static block Attention Residuals
LR AttnRes block, rank 32, scaled
LR AttnRes block, rank 64, scaled
LR AttnRes block, rank 64, unscaled
LR AttnRes full, rank 64, scaled
```

Then sweep:

```text
lrid_rank = 16, 32, 64, 128
lrid_logit_scale = off, 1/sqrt(k), 0.5/sqrt(k)
attnres_type = block, full
```

Suggested first comparison:

```bash
python train.py --use_lrid --attnres_type block --lrid_rank 64
python train.py --use_lrid --attnres_type block --lrid_rank 64 --no-lrid_logit_scale
```

If unscaled converges cleanly and improves loss, it may become the preferred default. Until then, scaled remains the conservative setting.

## Known Limitations

LR AttnRes still uses the repo’s `use_lrid` naming in code and CLI.

KV-cache generation is not supported for AttnRes/LR AttnRes. Generation uses a no-cache sliding-window fallback.

Document masking requires FlashAttention varlen support.

The current implementation is ready for training experiments, but not yet optimized for inference.
