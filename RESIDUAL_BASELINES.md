# Residual baselines

The existing trainer supports three additional residual operators on the same
attention/MLP backbone. The defaults retain the existing PreNorm/AttnRes paths.
Select one baseline with `--residual_type gated`, `ihc`, or `mhc`; these options
require `norm_pos="before"` and cannot be combined with AttnRes/LRID.

| Baseline | Default residual state | Read and write | Final readout |
| --- | --- | --- | --- |
| Gated Residuals (GR) | Four width-D streams | Per-stream affine RMSNorm, elementwise gated mean, scalar gated writes | Separate normalized gated mean; replaces final RMSNorm |
| iHC | Four width-D streams | Joint normalized controller, sigmoid read/write gates, identity bypass | Learned sigmoid collapse, then backbone final RMSNorm |
| mHC | Four width-D streams | Joint normalized controller, sigmoid read/write gates, Sinkhorn stream mixing | Learned sigmoid collapse, then backbone final RMSNorm |

Embeddings are replicated across streams. Each attention and MLP sublayer has
its own controller. The new controllers are initialized after the backbone,
so the same seed retains identical shared backbone parameters. Their extra
parameters and stream activations must be included in experiment accounting:
equal backbone size does not imply equal total parameters or training cost.

## Training

From the repository root, using the existing prepared data and 0.5B backbone
defaults:

```bash
python train.py --residual_type gated --out_dir out/gr --wandb_run_name gr --no-full_run
python train.py --residual_type ihc --out_dir out/ihc --wandb_run_name ihc --no-full_run
python train.py --residual_type mhc --out_dir out/mhc --wandb_run_name mhc --no-full_run
```

`--residual_num_streams` defaults to 4. GR's low-rank gate defaults to D/8;
`--gated_residual_rank` overrides it. `--mhc_sinkhorn_iters` defaults to 20.
All settings are stored in `model_args`. Loading older checkpoints selects the
existing path; resuming a baseline restores its saved settings and rejects
explicit conflicting baseline flags. The existing `return_hidden` loss path,
weight tying, cached decoding and document-mask arguments are supported.

All residual-controller/head parameters use the existing AdamW group while
backbone matrices retain Muon. GR norm gains and all iHC/mHC controllers remain
FP32 through `to_mixed_precision`; the GR gate matrices and backbone use BF16.
This is a matched-backbone experiment harness, not a reproduction of the full
Qwen, Hy4 or DeepSeek model/training recipes.

## Initialization and sources

- GR follows [Qwen3.8-Flash-Next, Eqs. 30–34](https://arxiv.org/abs/2608.30320)
  and [Transformers v5.16.0](https://github.com/huggingface/transformers/blob/v5.16.0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py):
  gate matrices have normal std 0.02, zero-centered RMS gains start at zero,
  and the read/injection projections use the paper's 1/N scaling.
- iHC follows [Hy4's published implementation](https://github.com/huggingface/transformers/blob/0f53828e83f2212060233d2ea552cc76617b8712/src/transformers/models/hy_v4/modeling_hy_v4.py):
  controller std 0.006, dynamic scale 0.01, read bias -log(N-1), write bias zero,
  write magnitude 2, norm epsilon 1e-5 and gate epsilon 1e-6.
- mHC follows [the paper](https://arxiv.org/abs/2512.24880) and
  [DeepSeek-V4's operator conventions](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py):
  FP32 controller and merge arithmetic, input-to-output stream orientation,
  and column-ending, epsilon-stabilized Sinkhorn. Scale 0.01 is the paper's
  choice; projection std 0.02 and zero static biases are explicit local
  fresh-training choices because the original training initialization is not
  fully specified by those sources. Finite Sinkhorn iterations approximate
  double stochasticity.

## Optional online kernels

The default `torch` implementations are differentiable and support the
trainer's existing whole-model `torch.compile` path. They require no new
packages. Two explicit alternatives use published training kernels and their
upstream backward implementations:

| Selector | Operation | Pinned dependency | Runtime |
| --- | --- | --- | --- |
| `--gated_residual_norm_backend fla` | Grouped per-stream affine RMSNorm | [FLA fbe2c38](https://github.com/fla-org/flash-linear-attention/tree/fbe2c383c191cdbdd7656864305c9eadee1a6756) | CUDA/Triton |
| `--mhc_sinkhorn_backend tile` | Fused Sinkhorn forward/backward | [TileKernels 79461d7](https://github.com/deepseek-ai/TileKernels/tree/79461d72d6f97f91b89403705a841c9d4c1eddd6) | Four streams, SM90/SM100, CUDA toolkit >=13.1 |

Install only the dependency you intend to test in a suitable CUDA environment:

```bash
python -m pip install 'flash-linear-attention[cuda] @ git+https://github.com/fla-org/flash-linear-attention.git@fbe2c383c191cdbdd7656864305c9eadee1a6756'
python -m pip install 'tile-kernels @ git+https://github.com/deepseek-ai/TileKernels.git@79461d72d6f97f91b89403705a841c9d4c1eddd6'
```

The TileKernels pin requires Python >=3.10, PyTorch >=2.10 and TileLang >=0.1.9.
Its toolkit requirement is newer than the repository's qualified CUDA 13.0
environment; installing the package alone does not supply that toolkit.
The pins matter: newer TileKernels revisions removed the imported autograd API.

Both pinned wrappers require eager execution, so add
`--no-torch_compile --no-torch_compile_max_autotune` when choosing either
alternative. Unsupported devices/dependencies raise an error; they do not
silently switch implementations. The pinned vLLM GR/iHC kernels inspected
were inference-only and were not connected to training.

These optional CUDA paths have conditional output/gradient parity tests but
have not been executed or benchmarked on H100 in this change. Compare complete
training steps against the default compiled `torch` implementation before
selecting a backend; an isolated fused operation does not establish an
end-to-end speedup. Backend changes are rejected during trajectory-exact resume.

## Checks

```bash
python -m pytest -q tests/test_residual_gated.py tests/test_residual_ihc.py tests/test_residual_mhc.py tests/test_residual_baselines.py
```

Tests cover independent equations and gradients, initialization, mixed
precision, optimizer coverage, checkpoint loading, shared backbone weights,
cached decoding, CPU-reference packed-document isolation, and fullgraph capture.
CUDA-specific checks skip when their hardware or optional packages are absent.
Source attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
