import argparse
import gc

import torch
import torch._dynamo as dynamo
from torch.nn import functional as F

if hasattr(dynamo.config, "recompile_limit"):
    dynamo.config.recompile_limit = 64

from attnres_ops import is_fused_attnres_available
from model import ModelConfig, OBPM


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark fused Attention Residual reads.")
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_embd", type=int, default=1024)
    parser.add_argument("--mlp_hidden_dim", type=int, default=2816)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--attnres_num_blocks", type=int, default=8)
    parser.add_argument("--batch_sizes", type=str, default="1,2,4")
    parser.add_argument("--lrid_ranks", type=str, default="32,64")
    parser.add_argument("--lrid_projection_rank", type=int, default=None)
    parser.add_argument("--lrid_num_heads", type=int, default=1)
    parser.add_argument("--attnres_type", choices=("block", "full"), default="block")
    parser.add_argument("--attnres_block_average", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-attnres_block_average", dest="attnres_block_average", action="store_false")
    parser.add_argument("--attnres_key_norm", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-attnres_key_norm", dest="attnres_key_norm", action="store_false")
    parser.add_argument("--attnres_training_cache_phase1", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-attnres_training_cache_phase1", dest="attnres_training_cache_phase1", action="store_false")
    parser.add_argument("--attnres_training_torch_phase2", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-attnres_training_torch_phase2", dest="attnres_training_torch_phase2", action="store_false")
    parser.add_argument("--attnres_fuse_read_norm", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-attnres_fuse_read_norm", dest="attnres_fuse_read_norm", action="store_false")
    parser.add_argument("--flash_attention", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-flash_attention", dest="flash_attention", action="store_false")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--phase", choices=("forward", "train"), default="forward")
    parser.add_argument("--loss", choices=("hidden_mse", "lm"), default="hidden_mse")
    parser.add_argument("--optimizer_step", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-optimizer_step", dest="optimizer_step", action="store_false")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--torch_compile", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-torch_compile", dest="torch_compile", action="store_false")
    parser.add_argument("--torch_compile_mode", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"), default="default")
    parser.add_argument("--torch_compile_fullgraph", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-torch_compile_fullgraph", dest="torch_compile_fullgraph", action="store_false")
    parser.add_argument("--torch_compile_dynamic", type=_str_to_bool, default=None)
    parser.add_argument("--torch_compile_cudagraphs", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-torch_compile_cudagraphs", dest="torch_compile_cudagraphs", action="store_false")
    parser.add_argument("--inductor_coordinate_descent", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--inductor_max_autotune_gemm", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--cases",
        type=str,
        default="baseline,attnres_pytorch,attnres_fused,lrid_pytorch,lrid_fused",
        help="Comma-separated cases to run.",
    )
    parser.add_argument("--nonzero_attnres_queries", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-nonzero_attnres_queries", dest="nonzero_attnres_queries", action="store_false")
    return parser.parse_args()


def make_config(args, kind, fused=False, lrid_rank=None):
    use_attnres = kind in {"attnres", "lrid"}
    use_lrid = kind == "lrid"
    return ModelConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_hidden_dim=args.mlp_hidden_dim,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        norm_pos="before",
        flash_attention=args.flash_attention,
        use_attnres=use_attnres,
        use_fused_attnres=fused,
        attnres_type=args.attnres_type,
        attnres_num_blocks=args.attnres_num_blocks,
        attnres_block_average=args.attnres_block_average,
        attnres_key_norm=args.attnres_key_norm,
        attnres_training_cache_phase1=args.attnres_training_cache_phase1,
        attnres_training_torch_phase2=args.attnres_training_torch_phase2,
        attnres_fuse_read_norm=args.attnres_fuse_read_norm,
        use_lrid=use_lrid,
        lrid_rank=lrid_rank or 64,
        lrid_projection_rank=args.lrid_projection_rank if use_lrid else None,
        lrid_num_heads=args.lrid_num_heads,
        lrid_use_logit_scale=True,
    )


def resolve_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def zero_grad(model):
    for param in model.parameters():
        param.grad = None


def sgd_step(model, lr):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-lr)


def training_loss(args, model, idx, targets):
    if args.loss == "hidden_mse":
        hidden = model(idx, return_hidden=True)
        return hidden.float().square().mean()

    logits = model(idx)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))


def run_training_step(args, model, idx, targets):
    zero_grad(model)
    loss = training_loss(args, model, idx, targets)
    loss.backward()
    if args.optimizer_step:
        sgd_step(model, args.lr)


def measure_case(args, device, dtype, batch_size, label, kind, fused=False, lrid_rank=None):
    torch.manual_seed(1234)
    config = make_config(args, kind, fused=fused, lrid_rank=lrid_rank)
    model = OBPM(config).to(device)
    if args.phase == "train":
        model.train()
    else:
        model.eval()
    if args.nonzero_attnres_queries and config.use_attnres:
        with torch.no_grad():
            if config.use_lrid:
                for query in model.transformer.lrid_queries:
                    query.normal_(0.0, config.init_std)
            else:
                for residual in model.transformer.attn_residuals:
                    residual.query.normal_(0.0, config.init_std)
        model._refresh_zero_query_fastpath_state()
    if dtype != torch.float32:
        model.to(dtype=dtype)
    if args.torch_compile:
        dynamo.reset()
        if not args.torch_compile_cudagraphs:
            try:
                import torch._inductor.config as inductor_config

                if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                    inductor_config.triton.cudagraphs = False
                if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraph_trees"):
                    inductor_config.triton.cudagraph_trees = False
            except Exception:
                pass
        if args.inductor_coordinate_descent or args.inductor_max_autotune_gemm:
            try:
                import torch._inductor.config as inductor_config

                if args.inductor_coordinate_descent and hasattr(inductor_config, "coordinate_descent_tuning"):
                    inductor_config.coordinate_descent_tuning = True
                if args.inductor_max_autotune_gemm and hasattr(inductor_config, "max_autotune_gemm"):
                    inductor_config.max_autotune_gemm = True
            except Exception:
                pass
        compile_kwargs = {}
        if args.torch_compile_mode != "default":
            compile_kwargs["mode"] = args.torch_compile_mode
        if args.torch_compile_fullgraph:
            compile_kwargs["fullgraph"] = True
        if args.torch_compile_dynamic is not None:
            compile_kwargs["dynamic"] = args.torch_compile_dynamic
        model = torch.compile(model, **compile_kwargs)
    idx = torch.randint(0, args.vocab_size, (batch_size, args.block_size), device=device)
    targets = torch.randint(0, args.vocab_size, (batch_size, args.block_size), device=device)

    try:
        if args.phase == "forward":
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(idx, return_hidden=True)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    model(idx, return_hidden=True)
                end.record()
                torch.cuda.synchronize()
        else:
            for _ in range(args.warmup):
                run_training_step(args, model, idx, targets)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                run_training_step(args, model, idx, targets)
            end.record()
            torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / args.iters
    finally:
        del model
        del idx
        del targets
        gc.collect()
        torch.cuda.empty_cache()
    projection_rank = ""
    if kind == "lrid":
        projection_rank = args.lrid_projection_rank or lrid_rank or 64
    return {
        "label": label,
        "batch": batch_size,
        "rank": lrid_rank or "",
        "projection_rank": projection_rank,
        "latency_ms": latency_ms,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested, but this CUDA device does not support bf16")
    device = torch.device("cuda")
    dtype = resolve_dtype(args.dtype)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    lrid_ranks = [int(x) for x in args.lrid_ranks.split(",") if x.strip()]
    cases = {x.strip() for x in args.cases.split(",") if x.strip()}

    print(
        f"fused_available={is_fused_attnres_available()} dtype={args.dtype} "
        f"phase={args.phase} loss={args.loss} optimizer_step={args.optimizer_step} "
        f"torch_compile={args.torch_compile} mode={args.torch_compile_mode} "
        f"fullgraph={args.torch_compile_fullgraph} dynamic={args.torch_compile_dynamic} "
        f"cudagraphs={args.torch_compile_cudagraphs} "
        f"coordesc={args.inductor_coordinate_descent} maxautomm={args.inductor_max_autotune_gemm} "
        f"nonzero_queries={args.nonzero_attnres_queries} "
        f"lrid_projection_rank={args.lrid_projection_rank}"
    )
    print("case,batch,rank,projection_rank,latency_ms,overhead_vs_baseline_pct,speedup_vs_pytorch")
    for batch_size in batch_sizes:
        rows = []
        baseline = measure_case(args, device, dtype, batch_size, "baseline_prenorm", "baseline")
        rows.append(baseline)
        attnres_pt = None
        if "attnres_pytorch" in cases:
            attnres_pt = measure_case(args, device, dtype, batch_size, "attnres_pytorch", "attnres", fused=False)
            rows.append(attnres_pt)
        if "attnres_fused" in cases:
            attnres_fused = measure_case(args, device, dtype, batch_size, "attnres_fused", "attnres", fused=True)
            rows.append(attnres_fused)
        for rank in lrid_ranks:
            if "lrid_pytorch" in cases:
                lrid_pt = measure_case(args, device, dtype, batch_size, "lrid_pytorch", "lrid", fused=False, lrid_rank=rank)
                rows.append(lrid_pt)
            if "lrid_fused" in cases:
                lrid_fused = measure_case(args, device, dtype, batch_size, "lrid_fused", "lrid", fused=True, lrid_rank=rank)
                rows.append(lrid_fused)

        pt_by_label_rank = {(row["label"], row["rank"]): row for row in rows}
        for row in rows:
            overhead = 100.0 * (row["latency_ms"] / baseline["latency_ms"] - 1.0)
            speedup = ""
            if row["label"] == "attnres_fused" and attnres_pt is not None:
                speedup = attnres_pt["latency_ms"] / row["latency_ms"]
            elif row["label"] == "lrid_fused" and ("lrid_pytorch", row["rank"]) in pt_by_label_rank:
                pt = pt_by_label_rank[("lrid_pytorch", row["rank"])]
                speedup = pt["latency_ms"] / row["latency_ms"]
            speedup_text = "" if speedup == "" else f"{speedup:.3f}x"
            print(
                f"{row['label']},{row['batch']},{row['rank']},{row['projection_rank']},{row['latency_ms']:.3f},"
                f"{overhead:.2f},{speedup_text}",
                flush=True,
            )


if __name__ == "__main__":
    main()
