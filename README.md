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

## Training

Single-process training still works with:

```bash
python train.py
```

For DDP, launch with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

```bash
torchrun --standalone --nproc_per_node=2 train.py --torch-max-autotune --full_run
```


For 8-GPU DDP with PyTorch compile max-autotune:

```bash
torchrun --standalone --nproc_per_node=8 train.py --torch-max-autotune
```

By default, DDP preserves the configured global batch size by dividing
`grad_accum_steps` across ranks, so the default `8` accumulation steps becomes
`1` local accumulation step on 8 GPUs. Use `--no-ddp_preserve_global_batch` if
you want global batch size to scale with `WORLD_SIZE`.

Enable PyTorch compile max-autotune with:

```bash
python train.py --torch-max-autotune
```

Max-autotune writes TorchInductor/Triton autotune caches. By default this uses
PyTorch/Triton's normal cache locations. To force a specific large/persistent
disk:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --torch-max-autotune \
  --torch_compile_cache_dir /workspace/LR-AttnRes/out/torchinductor_cache
```

For a full automated run that prompts for Hugging Face sign-in/repo setup at
startup, trains, saves the final checkpoint, uploads it to Hugging Face as
`final_model.pt`, and then runs evaluation:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --torch-max-autotune \
  --full_run \
  --full_run_hf_repo_id <your-hf-username>/<model-repo>
```

If `--full_run_hf_repo_id` is omitted, `full_run` prompts for it at startup.
Evaluation results from the automatic eval are saved to
`out/full_run_eval_step:<step>.txt`.

To resume after an interrupted run, leave `--ckpt_file_name` empty to pick the
highest-numbered `ckpt_step:<step>.pt` in `out_dir`:

```bash
torchrun --standalone --nproc_per_node=2 train.py --init_from resume --ckpt_file_name ""
```

## Evaluation

`run_eval.py` can load checkpoints produced by DDP training because checkpoints
save the unwrapped model state on rank 0. A normal eval run is single-process:

```bash
python run_eval.py --ckpts out/ckpt_step:1000.pt
```

Every `run_eval.py` invocation saves a text report by default:

```bash
python run_eval.py --ckpts out/ckpt_step:1000.pt --results-file out/eval_results.txt
```

For multi-GPU validation loss, launch with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 run_eval.py --validation-only
```

Validation loss is sharded across ranks and reduced exactly. Downstream lm-eval
tasks run on rank 0 only; when both validation and downstream tasks are enabled,
`run_eval.py` tears down the process group after validation before rank 0 starts
the long task pass. Eval compile is opt-in:

```bash
python run_eval.py --ckpts out/ckpt_step:1000.pt --torch-max-autotune
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
`--attn_res_query_init`. Block summaries are averaged by default with
`--attnres_block_average`; this divides by the sublayer count, and
`--attnres_block_average_mode sqrt` divides by the square root of the count.
Block AttnRes also adds a `log(count)` depth-logit prior for each compressed
block source so a block receives the softmax mass of the sublayers it represents.
For a learnable alternative, `--attnres_block_learned_scale` gives each live
partial/completed block source its own scalar, initialized with
`--attnres_block_learned_scale_init {count,sqrt,one}`. To remove block value
scale directly, `--attnres_block_value_norm` applies stateless RMSNorm to each
block source value instead of scalar scaling.

See [LR_ATTNRES.md](LR_ATTNRES.md) for the full design note, parameter cost,
stability rationale, and experiment matrix.
