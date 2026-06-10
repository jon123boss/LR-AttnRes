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

In practice, the input-dependent query path was unstable. LR AttnRes removes
that path by default and keeps the lower-risk part:

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

An alternate source-key mode is available:

```bash
--lrid_key_from_value
```

In this mode the attention and MLP output projections do not fuse source keys
into the same linear projection. Instead, each LR key is produced by a separate
value-key projection over the source value itself. With the default
`--lrid_key_value_norm`, that projection sees stateless
`rms_norm(source_value)`.

By default this is unshared: each attention and MLP output module keeps its own
value-key projection. Use `--lrid_key_from_value_shared` to instead share one
source-value key projection across all LR AttnRes sources.

For block LR AttnRes, the source value is the live partial block or completed
block summary. This means block keys are projected from the block value instead
of being sums or averages of per-sublayer keys.

The embedding source key can be made input-independent:

```bash
--lrid_static_embedding_key
```

With this flag, the embedding depth source uses one learned low-rank key
parameter broadcast over batch and token positions instead of a key projected
from the token embedding value. Shared source-key/query projections are still
created when requested for non-embedding sources.

Static keys can also be added to computed keys:

```bash
--lrid_add_static_embedding_key
--lrid_add_static_source_key
```

`--lrid_add_static_embedding_key` uses the normal computed embedding key plus a
learned input-independent key. It is mutually exclusive with
`--lrid_static_embedding_key`, which replaces the computed embedding key.
`--lrid_add_static_source_key` adds one learned input-independent key to each
computed non-embedding LR source key after the fused, value-from-key, shared,
full, or block key path has produced that source key.

An input-dependent query ablation is available:

```bash
--lrid_input_dependent_query
```

When enabled, attention and MLP output projections emit both a source key and a
source query:

```text
c_proj: d -> d + 2k
fc2: hidden -> d + 2k
```

The split becomes:

```text
output_d: normal sublayer output
key_k:    low-rank source key
query_k:  low-rank query for future depth routing
```

If `--lrid_key_from_value` and `--lrid_input_dependent_query` are both enabled,
the fused output projection emits only the normal output and the dynamic query:

```text
c_proj: d -> d + k
fc2: hidden -> d + k
```

The dynamic query can also be moved outside the fused output projection:

```bash
--lrid_query_from_value
```

In this mode, each dynamic query is produced from the source value directly,
using the same value normalization switch as outside keys:
`--lrid_key_value_norm` / `--no-lrid_key_value_norm`. By default this is
unshared, so each LR output module has its own value-query projection. Use
`--lrid_query_from_value_shared` to share one source-value query projection.

For block LR AttnRes, outside dynamic queries are projected from the live
partial block or completed block source value, mirroring outside source keys.
By default, `--attnres_block_average` is enabled and that source value is scaled
by the selected block-average denominator: the count by default, or the square
root of the count with `--attnres_block_average_mode sqrt`.
When `--attnres_block_learned_scale` is enabled, this fixed denominator is
replaced by one learned scalar per partial/completed block source. The scalar is
indexed by the residual source position: a live partial source after sublayer
`i` uses the same scalar that the completed block ending at `i` will keep. The
initializer is selected by `--attnres_block_learned_scale_init`: `count` starts
from `1/c`, `sqrt` starts from `1/sqrt(c)`, and `one` starts from the raw sum.
When `--attnres_block_value_norm` is enabled, the block source value is instead
stateless RMS-normalized to unit RMS scale. This overrides fixed averaging and
is mutually exclusive with learned block scaling.

The token embedding is also a depth source, so it gets its own low-rank key projection:

```text
embedding_key = Linear(d, k)(embedding)
```

By default, every query-bearing Attention Residual depth site has a learned static query:

```text
q_r in R^(lrid_num_heads x lrid_rank / lrid_num_heads)
```

There are:

```text
2 * n_layer
```

query parameters. The first depth-aggregation site has only the embedding as a
source, so it is an identity and has no query parameter.

When `lrid_input_dependent_query=True`, the static query parameters are still
created. The depth site uses a gated hybrid of the static query and the latest
available source query emitted by an attention or MLP output projection.

LR AttnRes depth routing can be multi-head:

```text
m = lrid_num_heads
key head dim   = k / m
value head dim = d / m
```

`lrid_rank` remains the total low-rank key width. The learned query is stored as
`R^(m x k/m)` per depth site. Source values keep their normal hidden width and
are reshaped to `R^(m x d/m)` only for the depth-attention weighted sum.

## Routing Formula

For depth site `r`:

```text
values = stack(source_value_i)
keys = stack(source_key_i)
values = reshape(values, sources, batch, time, m, d/m)
keys = reshape(keys, sources, batch, time, m, k/m)
keys = rms_norm(keys over k/m)
query = q_r                         # static-query mode
logits_i,h = scale * dot(keys_i,h, query_h)
weights_i,h = softmax_i(logits_h)
output_h = sum_i weights_i,h * values_i,h
output = reshape(output, batch, time, d)
```

For input-dependent query mode:

```text
dynamic_query = latest_source_query
dynamic_query = reshape(dynamic_query, batch, time, m, k/m)
gate_r in R^m
query = q_r + gate_r * dynamic_query
logits_i,h = scale * dot(keys_i,h, query_h)
```

The low-rank source keys are always input-dependent. The query is learned and
input-independent by default, and input-dependent only when
`lrid_input_dependent_query=True`.

## Logit Scale Toggle

LR AttnRes has an optional logit scale.

Default:

```text
lrid_use_logit_scale = True
lrid_logit_scale = 1 / sqrt(lrid_rank / lrid_num_heads)
```

For `lrid_rank=64` and `lrid_num_heads=1`, the default scale is:

```text
0.125
```

For `lrid_rank=64` and `lrid_num_heads=8`, the default scale is:

```text
0.353553...
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

Use the toggle because it is not yet obvious whether the scale is necessary once
the query path is static. The scale is useful for conservative stability; the
unscaled path may be worth testing because zero-initialized queries start with
zero logits anyway.

## Initialization

Static LR AttnRes queries are zero-initialized:

```text
q_r = 0
```

At step 0, all depth logits are zero, so depth routing is uniform over available sources. This mirrors normal Attention Residuals and avoids the instability from computed low-rank query projections.

When `lrid_input_dependent_query=True`, each depth site also has a learned
per-head gate:

```text
gate_r = 0
```

The dynamic query projection rows are initialized normally, but the zero gate
makes the effective query exactly static at step 0. This avoids a dead branch:
the gate can receive gradients immediately, and dynamic query projection rows
begin receiving gradients once the gate opens.

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

By default, block summaries are means of the sublayer outputs in that block:
`--attnres_block_average` divides partial and completed block summaries by the
selected denominator before using them as depth sources. The default denominator
is the number of accumulated sublayers; set `--attnres_block_average_mode sqrt`
to divide by the square root of that count. Each compressed block source also
adds `log(count)` to its depth-routing logit by default so it receives the
softmax prior mass of the sublayers it represents. Disable this prior for
ablations with `--no-attnres_block_count_prior`. It requires averaged block
summaries; both `count` and `sqrt` averaging modes are supported, while
learned-scale, raw-sum, and value-norm block summaries are rejected.
In LR AttnRes, emitted block source keys are scaled only when
`--no-attnres_key_norm` is used; when key normalization is enabled, dividing a
key by the denominator would be removed by the later RMSNorm. The fused dynamic
query, when enabled, remains the latest emitted query rather than an average.
The same rule applies to learned scales: source values are always scaled, while
precomputed LRID source keys are scaled only when key normalization is disabled.
With block value normalization, precomputed LRID keys are left unchanged; keys
projected from block values see the normalized value because the projection is
applied after the block value normalization.

When `--lrid_key_from_value` is enabled, block keys are not averaged separately.
The key is projected from the current block source value, so any configured
block value averaging applies before the key projection.

When `--lrid_query_from_value` is enabled, the same rule applies to dynamic
queries projected from block source values.

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
m = lrid_num_heads
L = number of transformer layers
```

Per layer, LR AttnRes adds:

```text
attention key overhead = k * d
MLP key overhead       = k * h
```

With unshared `lrid_key_from_value=True`, each LR-producing module still owns a
key projection, but that projection is applied to the `d`-wide source value:

```text
attention value-key overhead = k * d
MLP value-key overhead       = k * d
```

With `lrid_key_from_value_shared=True`, these per-layer key overheads are
removed and replaced by the once-per-model source key projection below.

When `lrid_input_dependent_query=True`, it also adds:

```text
attention query overhead = k * d
MLP query overhead       = k * h
```

With unshared `lrid_query_from_value=True`, each LR-producing module still owns
a dynamic-query projection, but that projection is applied to the `d`-wide
source value:

```text
attention value-query overhead = k * d
MLP value-query overhead       = k * d
```

With `lrid_query_from_value_shared=True`, these per-layer dynamic-query
overheads are removed and replaced by a once-per-model source query projection.

Once per model, it adds:

```text
embedding key projection overhead = k * d
depth query overhead              = (2L) * k
```

With `lrid_static_embedding_key=True`, the embedding key overhead is `k`
instead of `k * d`, unless shared source-key/query projections still require
the embedding/source projection module.

Additive static key modes add:

```text
static embedding key overhead = k
static source key overhead    = k
```

In shared source-key mode, the embedding key projection is reused as the shared
source-value key projection.

Shared source-query mode adds:

```text
source query overhead = k * d
```

In input-dependent query mode, static depth queries remain and gates are added:

```text
depth query overhead = (2L) * k
depth gate overhead  = (2L) * m
```

When `k` is fixed, increasing `m` does not change these projection or query
parameter counts. It only splits depth routing into `m` independent source
weight distributions, with the constraints `k % m == 0` and `d % m == 0`.

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
queries = 24 * 64 = 1,536

total extra = 2,213,376
```

This is about half the old LRID overhead because the computed-query projection
branch is disabled by default.

## Stability Notes

The unstable LRID path computed a query from every sublayer output. That created large gradient norms in training. Static-query LR AttnRes removes that computed query branch by default.

`--lrid_input_dependent_query` re-enables this family of behavior as an explicit
ablation. It should be treated as higher risk than static-query LR AttnRes,
especially with nonzero query projection initialization.

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
attnres_block_average: bool
attnres_block_average_mode: "count" | "sqrt"
attnres_block_count_prior: bool
attnres_block_learned_scale: bool
attnres_block_learned_scale_init: "count" | "sqrt" | "one"
attnres_block_value_norm: bool
attnres_key_norm: bool
attn_res_query_norm: bool
attn_res_query_init: "zero" | "normal" | "trunc_normal"
lrid_rank: int
lrid_num_heads: int
lrid_input_dependent_query: bool
lrid_static_embedding_key: bool
lrid_add_static_embedding_key: bool
lrid_add_static_source_key: bool
lrid_key_from_value: bool
lrid_key_from_value_shared: bool
lrid_key_value_norm: bool
lrid_query_from_value: bool
lrid_query_from_value_shared: bool
lrid_use_logit_scale: bool
lrid_logit_scale: float | None
```

Training CLI:

```bash
--use_lrid
--no-use_lrid
--attnres_block_average
--no-attnres_block_average
--attnres_block_average_mode {count,sqrt}
--attnres_block_count_prior
--no-attnres_block_count_prior
--attnres_block_learned_scale
--no-attnres_block_learned_scale
--attnres_block_learned_scale_init {count,sqrt,one}
--attnres_block_value_norm
--no-attnres_block_value_norm
--attnres_key_norm
--no-attnres_key_norm
--attn_res_query_norm
--no-attn_res_query_norm
--attn_res_query_init
--lrid_rank
--lrid_num_heads
--lrid_input_dependent_query
--no-lrid_input_dependent_query
--lrid_static_embedding_key
--no-lrid_static_embedding_key
--lrid_add_static_embedding_key
--no-lrid_add_static_embedding_key
--lrid_add_static_source_key
--no-lrid_add_static_source_key
--lrid_key_from_value
--no-lrid_key_from_value
--lrid_key_from_value_shared
--no-lrid_key_from_value_shared
--lrid_key_value_norm
--no-lrid_key_value_norm
--lrid_query_from_value
--no-lrid_query_from_value
--lrid_query_from_value_shared
--no-lrid_query_from_value_shared
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
LR AttnRes block, rank 64, 8 depth heads, scaled
LR AttnRes block, rank 64, input-dependent query, scaled
LR AttnRes block, rank 64, unscaled
LR AttnRes full, rank 64, scaled
```

Then sweep:

```text
lrid_rank = 16, 32, 64, 128
lrid_num_heads = 1, 2, 4, 8
lrid_input_dependent_query = false, true
lrid_logit_scale = off, 1/sqrt(k / lrid_num_heads), 0.5/sqrt(k / lrid_num_heads)
attnres_type = block, full
```

Suggested first comparison:

```bash
python train.py --use_lrid --attnres_type block --lrid_rank 64
python train.py --use_lrid --attnres_type block --lrid_rank 64 --lrid_num_heads 8
python train.py --use_lrid --attnres_type block --lrid_rank 64 --lrid_input_dependent_query
python train.py --use_lrid --attnres_type block --lrid_rank 64 --no-lrid_logit_scale
```

If unscaled converges cleanly and improves loss, it may become the preferred default. Until then, scaled remains the conservative setting.

## Known Limitations

LR AttnRes still uses the repo’s `use_lrid` naming in code and CLI.

KV-cache generation is not supported for AttnRes/LR AttnRes. Generation uses a no-cache sliding-window fallback.

Document masking requires FlashAttention varlen support.

The current implementation is ready for training experiments, but not yet optimized for inference.
