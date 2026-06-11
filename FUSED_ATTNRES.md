# Fused Attention Residual Kernels

This repo now has an optional fused Attention Residual read path in
`attnres_ops.py`. The model keeps the original PyTorch implementation as the
reference path and only uses the fused path when `use_fused_attnres=True`.

The current implementation is correct against the PyTorch path, but it has not
met the requested `<3%` compiled BF16 training-overhead target yet. The best
current measured block-mode training result is still `+8.95%` for base AttnRes
and `+12.75%` for LR-AttnRes on the representative B1/N=4 benchmark with
`torch.compile(mode="max-autotune")`.

## Files

- `attnres_ops.py`: packaged Triton/PyTorch Attention Residual read operators.
- `model.py`: model integration and block-training cached-logit schedule.
- `bench_attnres.py`: benchmark harness for baseline, PyTorch AttnRes, fused
  AttnRes, PyTorch LRID, and fused LRID.
- `tests/test_attnres_ops.py`: parity tests against the PyTorch reference path.

## Public Kernel API

Base AttnRes read:

```python
from attnres_ops import attention_residual_read

out = attention_residual_read(
    values,                 # Tensor [S, B, T, D] or list of [B, T, D]
    query,                  # Tensor [D]
    key_norm=True,
    normalize_output=False,
    source_counts=None,        # optional beta=1 wrapper: logits += log(count)
    source_logit_biases=None,  # optional explicit source biases, e.g. beta * log(count)
)
# out: [B, T, D]
```

LR-AttnRes/LRID read:

```python
from attnres_ops import lrid_attention_residual_read

out = lrid_attention_residual_read(
    values,                 # Tensor/list of value sources [B, T, D]
    keys,                   # Tensor/list of key sources [B, T, R]
    query,                  # [H, R/H] static or [B, T, H, R/H] dynamic
    num_heads=1,
    logit_scale=1.0 / (32 ** 0.5),
    key_norm=True,
    normalize_output=False,
    source_counts=None,        # optional beta=1 wrapper: logits += log(count)
    source_logit_biases=None,  # optional explicit source biases, e.g. beta * log(count)
)
# out: [B, T, D]
```

Model block mode now computes explicit `source_logit_biases` for
`beta * log(count)` so fixed and learned beta share the same fused path.
The public ops still accept `source_counts` as the beta=1 compatibility wrapper.
Count-biased direct reads and cached block-training reads use the cached-logit
fused phase1 path, so generalized block priors stay on fused AttnRes.

Block training uses a cached-logit two-part read internally:

- `attention_residual_phase1_from_logits(values, logits)` computes the
  completed-block read without materializing full attention weights as a public
  tensor.
- Phase 2 merges the current partial block with the completed-block read. The
  default path currently uses the PyTorch/Inductor phase2 expression because the
  custom BF16 Triton phase2 path was slower and did not match the full-model
  BF16 path tightly enough. Base AttnRes uses an `addcmul` blend expression;
  LRID uses `torch.lerp`, which measured better for the low-rank phase2 path.

## Model Flags

Training CLI flags:

```bash
--use_fused_attnres true
--attnres_training_cache_phase1 true
--attnres_training_torch_phase2 true
--attnres_fuse_read_norm true
```

LRID-specific:

```bash
--use_lrid true
--lrid_rank 32
--lrid_projection_rank 64   # optional physical padding; logical rank stays 32
--lrid_key_from_output_tail true  # optional: use value[..., -r:] as the LRID key
--lrid_num_heads 1
```

`lrid_projection_rank=64` can improve GEMM shape efficiency for logical
`lrid_rank=32`, but it does not change the logical key rank used by the residual
read.

`lrid_key_from_output_tail=True` is an alternative LRID key-source mode. Instead
of learning extra rank-`r` key rows in the attention/MLP output projections, the
model uses the last `r` dimensions of each residual value as the key. The
residual read is still the LRID read, so the fused LRID kernels consume the
ordinary full-width values and the sliced rank-`r` tail keys.

## Recommended Commands

Start with the standalone parity and compatibility script:

```bash
python test_attnres_fused_parity.py
```

This checks direct base AttnRes and LRID fused reads against the PyTorch
reference for both forward and backward, then checks small full-model
fused-versus-PyTorch parity and old-style checkpoint/config compatibility.

For a focused CUDA/Triton smoke test:

```bash
python test_attnres_fused_parity.py --mode direct --device cuda --dtype fp32
python test_attnres_fused_parity.py --mode model --device cuda --dtype fp32
python test_attnres_fused_parity.py --mode compat --device cuda
```

Run the pytest coverage when `pytest` is available:

```bash
PYTHONPATH=. pytest -q tests/test_attnres_ops.py
```

Train with the fused path enabled:

```bash
python train.py \
  --use_attnres true \
  --use_fused_attnres true \
  --attnres_type block \
  --attnres_num_blocks 8
```

Train LRID/LR-AttnRes with the fused path enabled:

```bash
python train.py \
  --use_lrid true \
  --use_fused_attnres true \
  --attnres_type block \
  --attnres_num_blocks 8 \
  --lrid_rank 32
```

Train LRID/LR-AttnRes with output-tail keys:

```bash
python train.py \
  --use_lrid true \
  --use_fused_attnres true \
  --lrid_key_from_output_tail true \
  --attnres_type block \
  --attnres_num_blocks 8 \
  --lrid_rank 32
```

For the current best LRID projection-shape benchmark path:

```bash
python train.py \
  --use_lrid true \
  --use_fused_attnres true \
  --attnres_type block \
  --attnres_num_blocks 8 \
  --lrid_rank 32 \
  --lrid_projection_rank 64
```

Benchmark the representative compiled training path:

```bash
python bench_attnres.py \
  --dtype bf16 \
  --phase train \
  --batch_sizes 1 \
  --cases baseline,attnres_fused,lrid_fused \
  --lrid_ranks 32 \
  --n_layer 24 \
  --n_embd 1024 \
  --n_head 16 \
  --block_size 2048 \
  --attnres_type block \
  --attnres_num_blocks 8 \
  --torch_compile true \
  --torch_compile_mode max-autotune \
  --torch_compile_dynamic false \
  --torch_compile_cudagraphs true \
  --inductor_coordinate_descent true \
  --inductor_max_autotune_gemm true \
  --nonzero_attnres_queries true \
  --iters 8 \
  --warmup 4 \
  --loss hidden_mse \
  --attnres_fuse_read_norm true
```

## Backward Compatibility

The unfused PyTorch path remains the default. Existing training/evaluation
commands and existing checkpoints use `use_fused_attnres=False` unless the new
flag is explicitly set.

Old checkpoint `model_args` dictionaries that do not contain the new fused
fields still instantiate because `ModelConfig` supplies defaults for:

- `use_fused_attnres=False`
- `attnres_training_cache_phase1=True`
- `attnres_training_torch_phase2=True`
- `attnres_fuse_read_norm=True`
- `lrid_projection_rank=lrid_rank`
- `lrid_key_from_output_tail=False`

Old LRID checkpoints remain shape-compatible because `lrid_projection_rank`
defaults to the logical `lrid_rank`; extra projection rows only exist when a new
run explicitly sets `--lrid_projection_rank` greater than `--lrid_rank`.
Output-tail key mode is opt-in, so existing checkpoints keep using the learned
LRID key projection unless `--lrid_key_from_output_tail true` is set for a new
run.

## Correctness

Run:

```bash
python -m py_compile attnres_ops.py model.py train.py utils.py bench_attnres.py tests/test_attnres_ops.py
python test_attnres_fused_parity.py
PYTHONPATH=. pytest -q tests/test_attnres_ops.py
```

On a CUDA/Triton host, the expected result is:

```text
25 passed
```

On a non-CUDA host, these pytest cases skip by design; use
`test_attnres_fused_parity.py --device cpu` for CPU/fallback logic checks.

The tests cover:

- Base AttnRes fused read parity.
- LRID fused read parity.
- Static and dynamic LRID query shapes.
- BF16 LRID read parity.
- Phase1 cached-logit backward parity.
- Phase2 backward parity in FP32.
- Full-model fused-vs-PyTorch path parity for small full and block configs.
- Cached block training path parity in BF16 for base AttnRes and LRID.

## Benchmarks

Representative command for compiled BF16 block training:

```bash
python bench_attnres.py \
  --dtype bf16 \
  --phase train \
  --batch_sizes 1 \
  --cases baseline,attnres_fused,lrid_fused \
  --lrid_ranks 32 \
  --n_layer 24 \
  --n_embd 1024 \
  --n_head 16 \
  --block_size 2048 \
  --attnres_type block \
  --attnres_num_blocks 4 \
  --torch_compile true \
  --torch_compile_mode reduce-overhead \
  --torch_compile_dynamic false \
  --torch_compile_cudagraphs true \
  --nonzero_attnres_queries true \
  --iters 8 \
  --warmup 4 \
  --loss hidden_mse \
  --attnres_fuse_read_norm true
```

For the best measured LRID projection shape:

```bash
python bench_attnres.py \
  --dtype bf16 \
  --phase train \
  --batch_sizes 1 \
  --cases baseline,attnres_fused,lrid_fused \
  --lrid_ranks 32 \
  --lrid_projection_rank 64 \
  --n_layer 24 \
  --n_embd 1024 \
  --n_head 16 \
  --block_size 2048 \
  --attnres_type block \
  --attnres_num_blocks 4 \
  --torch_compile true \
  --torch_compile_mode max-autotune \
  --torch_compile_dynamic false \
  --torch_compile_cudagraphs true \
  --inductor_coordinate_descent true \
  --inductor_max_autotune_gemm true \
  --nonzero_attnres_queries true \
  --iters 8 \
  --warmup 4 \
  --loss hidden_mse \
  --attnres_fuse_read_norm true
```

## Current Measured Results

Environment for these rows:

- H100 80GB
- PyTorch `2.8.0+cu128`
- Triton `3.4.0`
- BF16 training, hidden-MSE loss, batch size 1
- `n_layer=24`, `n_embd=1024`, `n_head=16`, `block_size=2048`
- nonzero static AttnRes/LRID queries
- `torch.compile(dynamic=False)` with CUDA graphs

| setting | compile mode | case | latency | overhead |
|---|---|---:|---:|---:|
| N=8, r=32 | reduce-overhead | baseline | `14.071 ms` | `0.00%` |
| N=8, r=32 | reduce-overhead | base AttnRes fused | `16.032 ms` | `+13.94%` |
| N=8, r=32 | reduce-overhead | LRID fused | `16.773 ms` | `+19.20%` |
| N=8, r=32, projection_rank=64 | reduce-overhead | baseline | `14.053 ms` | `0.00%` |
| N=8, r=32, projection_rank=64 | reduce-overhead | LRID fused | `16.584 ms` | `+18.00%` |
| N=4, r=32 | reduce-overhead | baseline | `14.046 ms` | `0.00%` |
| N=4, r=32 | reduce-overhead | base AttnRes fused | `15.674 ms` | `+11.59%` |
| N=4, r=32 | reduce-overhead | LRID fused | `16.137 ms` | `+14.89%` |
| N=4, r=32, base post-cleanup smoke | reduce-overhead | baseline | `13.986 ms` | `0.00%` |
| N=4, r=32, base post-cleanup smoke | reduce-overhead | base AttnRes fused | `15.588 ms` | `+11.46%` |
| N=4, r=32, LRID post-cleanup smoke | reduce-overhead | baseline | `14.006 ms` | `0.00%` |
| N=4, r=32, LRID post-cleanup smoke | reduce-overhead | LRID fused | `16.171 ms` | `+15.46%` |
| N=4, r=32, projection_rank=64 | reduce-overhead | baseline | `14.087 ms` | `0.00%` |
| N=4, r=32, projection_rank=64 | reduce-overhead | LRID fused | `16.053 ms` | `+13.96%` |
| N=4, r=32, projection_rank=64 | max-autotune | baseline | `13.563 ms` | `0.00%` |
| N=4, r=32, projection_rank=64 | max-autotune | base AttnRes fused | `14.777 ms` | `+8.95%` |
| N=4, r=32, projection_rank=64 | max-autotune | LRID fused | `15.292 ms` | `+12.75%` |

Full-mode representative 24-layer training is not yet a production-optimized
path in this implementation. The generic full read supports correctness tests
and small configs, but the high-source-count 24-layer full path needs a tiled
source kernel or a different training schedule before it should be compared to
the block-mode path.

## Theoretical Overhead

For the 24-layer, 1024-wide model:

- Full base AttnRes read cost: about `0.813%` of non-embedding Transformer core.
- Full LRID `r=32` read cost: about `0.419%`; extra LRID projection rows add
  about `0.967%`.
- Block N=8 base read cost: about `0.175%`.
- Block N=8 LRID `r=32` read cost: about `0.090%`; projection rows still add
  about `0.967%`.
- Block N=4 base read cost: about `0.112%`.
- Block N=4 LRID `r=32` read cost: about `0.058%`; projection rows still add
  about `0.967%`.

The measured wall-clock overhead is therefore dominated by runtime structure,
not arithmetic count.

## Current Bottlenecks

Profiler results point to:

- Phase2 partial-block merge and its backward graph as the largest explicit
  residual-specific cost.
- LRID projection shape overhead: physical projection rank `64` improves some
  runs, showing the `+32` logical rows are not the whole story.
- Source-logit preparation and block source bookkeeping in the compiled graph.
- Full mode source count: the current generic list kernels are not enough for
  48 residual-writing sublayers at representative depth.

## Notes

- The fused path is optional and does not remove the PyTorch reference path.
- Static query and single-depth-head LRID are the optimized training target.
- Input-dependent LRID queries and multi-head depth routing still fall back to
  the general fused/read path where supported, but they are not the current
  performance target.
- No public Liger Kernel depth-wise AttnRes kernel was found in the current
  public Liger docs/repo surface. Liger provides general LLM kernels such as
  RMSNorm, RoPE, SwiGLU, CrossEntropy, fused linear CE, and multi-token
  attention, so there is not yet a direct apples-to-apples Liger AttnRes
  benchmark included here.
