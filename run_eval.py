# run_eval.py
#   python run_eval.py
#   python run_eval.py --ckpts out/ckpt_step:38146.pt
#   python run_eval.py --validation-only
#   python run_eval.py --tasks-only

import argparse
import glob
import os
import re
from importlib.metadata import PackageNotFoundError, version
from typing import List, Tuple, Union

_DEFAULT_TORCH_COMPILE_CACHE_DIR = (
    os.environ.get("TORCH_COMPILE_CACHE_DIR")
    or os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    or ""
)
_EXPLICIT_TRITON_CACHE_DIR = os.environ.get("TRITON_CACHE_DIR")
if os.environ.get("TORCH_COMPILE_CACHE_DIR") and os.environ.get("TORCHINDUCTOR_CACHE_DIR") is None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.environ["TORCH_COMPILE_CACHE_DIR"]

import torch
from tqdm import tqdm

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import tiktoken
from tokenizer_utils import GPT4_TOKENIZER_MODEL as _GPT4_TOKENIZER_MODEL, get_tiktoken_encoding
import torch.distributed as dist

try:
    from lm_eval import simple_evaluate
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ModuleNotFoundError as exc:
    if exc.name != "lm_eval":
        raise

    simple_evaluate = None
    _LM_EVAL_IMPORT_ERROR = exc

    class LM:
        def __init__(self):
            self._rank = 0
            self._world_size = 1

        @property
        def rank(self):
            return self._rank

        @property
        def world_size(self):
            return self._world_size

    def register_model(*_names):
        def decorator(cls):
            return cls

        return decorator
else:
    _LM_EVAL_IMPORT_ERROR = None

from model import OBPM, ModelConfig
from criterion import get_criterion
from dataloader import warmup_boundaries
from utils import compute_validation_loss, get_validation_dataloader, unwrap_model

OLMES_DEFAULT_SHOTS = 5
DEFAULT_CKPT_DIR = "out"
CHECKPOINT_PATTERN = "ckpt_step:*.pt"
DEFAULT_RESULTS_FILE = os.path.join(DEFAULT_CKPT_DIR, "eval_results.txt")
DATASET_SCRIPT_ERROR = "Dataset scripts are no longer supported"
DATASET_SCRIPT_TASKS = {"siqa", "social_iqa"}

TASK_MAPPING = {
    "mmlu": "mmlu",
    "MMLU": "mmlu",
    "arc-c": "arc_challenge",
    "ARC-C": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "arc-e": "arc_easy",
    "ARC-E": "arc_easy",
    "arc_easy": "arc_easy",
    "boolq": "boolq",
    "CommonSenseQA": "commonsense_qa",
    "commonsense_qa": "commonsense_qa",
    "HellaSwag": "hellaswag",
    "hellaswag": "hellaswag",
    "OpenbookQA": "openbookqa",
    "openbookqa": "openbookqa",
    "PIQA": "piqa",
    "piqa": "piqa",
    "SIQA": "siqa",
    "siqa": "siqa",
    "Winogrande": "winogrande",
    "winogrande": "winogrande",
}


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def configure_torch_compile_cache(cache_dir: str) -> str:
    cache_dir = (cache_dir or "").strip()
    if not cache_dir:
        return ""
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    if _EXPLICIT_TRITON_CACHE_DIR is None:
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    return cache_dir


def setup_distributed():
    has_rank = "RANK" in os.environ
    has_world_size = "WORLD_SIZE" in os.environ
    if not has_rank and not has_world_size:
        return False, 0, 1, 0
    if not has_rank or not has_world_size:
        raise RuntimeError("Both RANK and WORLD_SIZE must be set for distributed evaluation.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("RANK must satisfy 0 <= RANK < WORLD_SIZE")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be >= 0")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(local_rank)
    if world_size == 1:
        return False, rank, world_size, local_rank

    backend = "nccl" if cuda_available else "gloo"
    if cuda_available:
        device = torch.device("cuda", local_rank)
        try:
            dist.init_process_group(backend=backend, device_id=device)
        except TypeError:
            dist.init_process_group(backend=backend)
        dist.barrier(device_ids=[local_rank])
    else:
        dist.init_process_group(backend=backend)
        dist.barrier()
    return True, rank, world_size, local_rank


def print0(master_process: bool, *args, **kwargs):
    if master_process:
        print(*args, **kwargs)


def _parse_batch_size(batch_size: Union[int, str, None]) -> int:
    if isinstance(batch_size, int):
        return max(1, batch_size)
    if isinstance(batch_size, str) and batch_size.isdigit():
        return max(1, int(batch_size))
    return 1


def _checkpoint_sort_key(path: str):
    match = re.search(r"ckpt_step:(\d+)\.pt$", os.path.basename(path))
    if match:
        return (0, int(match.group(1)))
    return (1, path)


def discover_checkpoints(ckpt_dir: str) -> List[str]:
    checkpoints = glob.glob(os.path.join(ckpt_dir, CHECKPOINT_PATTERN))
    return sorted(checkpoints, key=_checkpoint_sort_key)


def _datasets_disables_scripts() -> bool:
    try:
        datasets_version = version("datasets")
    except PackageNotFoundError:
        return False

    major = datasets_version.split(".", 1)[0]
    return major.isdigit() and int(major) >= 4


def _merge_eval_outputs(combined_output: dict, task_output: dict):
    for key, value in task_output.items():
        if isinstance(value, dict):
            combined_output.setdefault(key, {}).update(value)
        else:
            combined_output[key] = value


def run_downstream_tasks(lm_obj: "OBPMWrapper", valid_tasks: List[str], device: str):
    if simple_evaluate is None:
        raise RuntimeError(
            "run_eval.py requires lm_eval. Install it with `pip install lm_eval` "
            "before running downstream evaluations."
        ) from _LM_EVAL_IMPORT_ERROR

    combined_output = {"results": {}}
    skipped_tasks = {}
    datasets_disables_scripts = _datasets_disables_scripts()

    for task in valid_tasks:
        if datasets_disables_scripts and task in DATASET_SCRIPT_TASKS:
            reason = (
                "installed datasets package no longer supports dataset scripts; "
                "install datasets<4 or omit this task"
            )
            print(f"Skipping task {task}: {reason}.")
            skipped_tasks[task] = reason
            continue

        print(f"Running task: {task}")
        try:
            task_output = simple_evaluate(
                model=lm_obj,
                tasks=[task],
                num_fewshot=OLMES_DEFAULT_SHOTS,
                batch_size=1,
                device=device,
            )
        except RuntimeError as exc:
            if DATASET_SCRIPT_ERROR not in str(exc):
                raise

            reason = (
                "dataset script is incompatible with the installed datasets package; "
                "install datasets<4 or omit this task"
            )
            print(f"Skipping task {task}: {reason}.")
            skipped_tasks[task] = reason
            continue

        _merge_eval_outputs(combined_output, task_output)

    if skipped_tasks:
        combined_output["skipped_tasks"] = skipped_tasks

    return combined_output


def _find_split_token_index(enc: tiktoken.Encoding, full_ids: List[int], context: str) -> int:
    target_len = len(context)

    lo, hi = 0, len(full_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        dec_len = len(enc.decode(full_ids[:mid]))
        if dec_len < target_len:
            lo = mid + 1
        else:
            hi = mid

    k0 = lo
    for k in range(max(0, k0 - 8), min(len(full_ids) + 1, k0 + 9)):
        if enc.decode(full_ids[:k]) == context:
            return k

    return len(enc.encode(context))


@register_model("obpm")
class OBPMWrapper(LM):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: Union[int, str] = 1,
        max_batch_size: int = 64,
        torch_compile: bool = False,
        torch_compile_max_autotune: bool = False,
        torch_compile_cache_dir: str = _DEFAULT_TORCH_COMPILE_CACHE_DIR,
        verbose: bool = True,
    ):
        super().__init__()
        self._device = torch.device(device)

        self.batch_size_per_gpu = _parse_batch_size(batch_size)
        self.max_batch_size = max_batch_size
        self.torch_compile = bool(torch_compile or torch_compile_max_autotune)
        self.torch_compile_mode = "max-autotune" if torch_compile_max_autotune else None
        self.torch_compile_cache_dir = torch_compile_cache_dir
        self.verbose = bool(verbose)

        if self.verbose:
            print(f"Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

        model_args = checkpoint.get("model_args", {})
        if isinstance(model_args, dict):
            config = ModelConfig(**model_args)
        else:
            config = model_args
        self.checkpoint_config = checkpoint.get("config", {})

        self.model = OBPM(config)


        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self._device)

        if self._device.type == "cuda" and hasattr(self.model, "to_mixed_precision"):
            self.model.to_mixed_precision(dtype=torch.bfloat16)
        if self.torch_compile:
            self.torch_compile_cache_dir = configure_torch_compile_cache(self.torch_compile_cache_dir)
            if self.verbose:
                print(
                    f"Torch compile enabled for eval | mode: {self.torch_compile_mode or 'default'} | "
                    f"cache: {self.torch_compile_cache_dir or 'default'}"
                )
            self.model = torch.compile(self.model, mode=self.torch_compile_mode)

        self.model.eval()

        self.tokenizer_model = self.checkpoint_config.get("tokenizer_model", _GPT4_TOKENIZER_MODEL)
        self.tokenizer = get_tiktoken_encoding(self.tokenizer_model)
        self.eot_token_id = self.tokenizer.eot_token

        self.vocab_size = int(config.vocab_size)
        self.max_length = int(config.block_size)

    @property
    def device(self):
        return str(self._device)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def max_gen_toks(self):
        return 256

    @property
    def tokenizer_name(self):
        return f"tiktoken-{self.tokenizer.name}"

    def _truncate_left(self, ids: List[int], split: int) -> Tuple[List[int], int]:
        overflow = len(ids) - self.max_length
        if overflow <= 0:
            return ids, split

        ids = ids[overflow:]
        split = max(1, split - overflow) 
        return ids, split

    def _encode_pair(self, context: str, continuation: str):

        full_text = context + continuation
        full_ids = self.tokenizer.encode(full_text)

        split = _find_split_token_index(self.tokenizer, full_ids, context)

        if split == 0:
            full_ids = [self.eot_token_id] + full_ids
            split = 1

        full_ids, split = self._truncate_left(full_ids, split)

        cont_ids = full_ids[split:]
        return full_ids, split, cont_ids

    def loglikelihood(self, requests):
        res = []

        for instance in tqdm(requests, desc="Evaluating (loglikelihood)", leave=False):
            context, continuation = instance.args
            full_ids, ctx_len, cont_ids = self._encode_pair(context, continuation)
            cont_len = len(cont_ids)

            if cont_len == 0:
                res.append((0.0, True))
                continue

            x = torch.tensor([full_ids], dtype=torch.long, device=self._device)

            with torch.inference_mode():
                logits = self.model(x)
                log_probs = torch.log_softmax(logits, dim=-1)

            start_logit_idx = ctx_len - 1
            end_logit_idx = ctx_len + cont_len - 1

            target = torch.tensor(full_ids[ctx_len : ctx_len + cont_len], dtype=torch.long, device=self._device)

            token_log_probs = log_probs[0, start_logit_idx:end_logit_idx, :]

            greedy = token_log_probs.argmax(dim=-1)
            is_greedy = bool((greedy == target).all().item())

            gathered = torch.gather(token_log_probs, 1, target.unsqueeze(-1)).squeeze(-1)
            sum_ll = float(gathered.sum().item())

            res.append((sum_ll, is_greedy))

        return res

    def loglikelihood_rolling(self, requests):
        out = []
        for instance in tqdm(requests, desc="Evaluating (loglikelihood_rolling)", leave=False):
            (text,) = instance.args
            ids = self.tokenizer.encode(text)

            if len(ids) == 0:
                out.append(0.0)
                continue

            total = 0.0
            scored_ids = [self.eot_token_id] + ids
            for t in range(1, len(scored_ids)):
                start = max(0, (t + 1) - self.max_length)
                window = scored_ids[start : t + 1]
                x = torch.tensor([window], dtype=torch.long, device=self._device)

                with torch.inference_mode():
                    logits = self.model(x)
                    log_probs = torch.log_softmax(logits, dim=-1)

                target_id = window[-1]
                lp = log_probs[0, -2, target_id]
                total += float(lp.item())

            out.append(total)

        return out

    def generate_until(self, requests):
        res = []
        for instance in tqdm(requests, desc="Generating", leave=False):
            context, gen_kwargs = instance.args

            until = gen_kwargs.get("until", [])
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", 64))

            tokens = self.tokenizer.encode(context)
            if len(tokens) == 0:
                tokens = [self.eot_token_id]

            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length :]

            x = torch.tensor([tokens], dtype=torch.long, device=self._device)

            with torch.inference_mode():
                out_idx = unwrap_model(self.model).generate(x, max_new_tokens=max_gen_toks, temperature=0.0)

            out = out_idx[0].tolist()
            new_tokens = out[len(x[0]) :]
            text = self.tokenizer.decode(new_tokens)

            for term in until:
                if term and term in text:
                    text = text.split(term)[0]
                    break

            res.append(text)
        return res

    def _chunk_requests(self, requests, chunk_size: int):
        for i in range(0, len(requests), chunk_size):
            yield requests[i : i + chunk_size]


def run_validation_loss(
    lm_obj: OBPMWrapper,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    master_process: bool = True,
):
    config = getattr(lm_obj, "checkpoint_config", None)
    if not config:
        raise RuntimeError(
            "Checkpoint does not contain a training config, so run_eval.py cannot "
            "build the validation dataloader for validation loss."
        )

    eval_config = dict(config)
    eval_config["pin_memory"] = bool(lm_obj._device.type == "cuda" and eval_config.get("pin_memory", False))
    eval_config["rank"] = rank if distributed else 0
    eval_config["world_size"] = world_size if distributed else 1
    eval_config["master_process"] = master_process

    val_loader = get_validation_dataloader(eval_config)
    if eval_config.get("use_doc_masking", False):
        print0(master_process, "Warming up validation document boundary cache...")
        warmup_boundaries(val_loader.dataset, verbose=master_process)
        print0(master_process, "Validation boundary warmup complete.")

    criterion = get_criterion(eval_config)
    val_metrics = compute_validation_loss(
        lm_obj.model,
        criterion,
        val_loader,
        lm_obj._device,
        lm_obj.vocab_size,
        use_doc_masking=eval_config.get("use_doc_masking", False),
        distributed=distributed,
    )
    print0(
        master_process,
        f"Validation loss: {val_metrics['loss']:.4f} "
        f"({val_metrics['tokens']:,} tokens across {val_metrics['batches']:,} batches)"
    )
    print0(master_process, "-" * 80)
    return val_metrics


def evaluate_checkpoints(
    checkpoints: List[str],
    tasks_list: List[str],
    include_validation: bool = True,
    include_tasks: bool = True,
    torch_compile: bool = False,
    torch_compile_max_autotune: bool = False,
    torch_compile_cache_dir: str = _DEFAULT_TORCH_COMPILE_CACHE_DIR,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    if not include_validation and not include_tasks:
        raise ValueError("At least one evaluation mode must be enabled.")

    master_process = rank == 0
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    print0(master_process, f"Device: {device}")
    print0(master_process, f"Distributed eval: {distributed} | Rank: {rank}/{world_size} | Local rank: {local_rank}")
    print0(
        master_process,
        f"Torch compile: {torch_compile or torch_compile_max_autotune} | "
        f"mode: {'max-autotune' if torch_compile_max_autotune else 'default'} | "
        f"cache: {torch_compile_cache_dir or 'default'}",
    )

    valid_tasks = [TASK_MAPPING.get(t, t) for t in tasks_list] if include_tasks else []

    if include_validation and include_tasks:
        print0(master_process, "Evaluation mode: validation loss + downstream tasks")
    elif include_validation:
        print0(master_process, "Evaluation mode: validation loss only")
    else:
        print0(master_process, "Evaluation mode: downstream tasks only")

    if include_tasks:
        print0(master_process, f"Tasks to evaluate: {valid_tasks}")
        if distributed:
            print0(master_process, "Downstream tasks run on rank 0 only; validation loss is sharded across ranks.")
    print0(master_process, "-" * 80)

    results = {}

    if distributed and include_tasks:
        if include_validation:
            print0(
                master_process,
                "Running distributed validation first; non-rank0 processes will exit before downstream tasks.",
            )
            for ckpt in checkpoints:
                if not os.path.exists(ckpt):
                    print0(master_process, f"Skipping missing checkpoint: {ckpt}")
                    continue

                print0(master_process, f"\nEvaluating validation for checkpoint: {ckpt}")
                print0(master_process, "=" * 80)

                lm_obj = OBPMWrapper(
                    model_path=ckpt,
                    device=device,
                    batch_size=1,
                    torch_compile=torch_compile,
                    torch_compile_max_autotune=torch_compile_max_autotune,
                    torch_compile_cache_dir=torch_compile_cache_dir,
                    verbose=master_process,
                )
                val_metrics = run_validation_loss(
                    lm_obj,
                    distributed=True,
                    rank=rank,
                    world_size=world_size,
                    master_process=master_process,
                )
                if master_process:
                    results[ckpt] = {"validation_loss": val_metrics}
                    print("\nResults:")
                    print(f"  validation_loss: {val_metrics['loss']:.4f}")
                    print("-" * 80)
                del lm_obj
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.destroy_process_group()

        if not master_process:
            return results

        print0(master_process, "Running downstream tasks on rank 0 with no active process group.")
        print0(master_process, "-" * 80)
        for ckpt in checkpoints:
            if not os.path.exists(ckpt):
                print0(master_process, f"Skipping missing checkpoint: {ckpt}")
                continue

            print0(master_process, f"\nEvaluating downstream tasks for checkpoint: {ckpt}")
            print0(master_process, "=" * 80)

            lm_obj = OBPMWrapper(
                model_path=ckpt,
                device=device,
                batch_size=1,
                torch_compile=torch_compile,
                torch_compile_max_autotune=torch_compile_max_autotune,
                torch_compile_cache_dir=torch_compile_cache_dir,
                verbose=True,
            )
            eval_output = results.get(ckpt, {})
            task_output = run_downstream_tasks(lm_obj, valid_tasks, device)
            task_output.update(eval_output)
            results[ckpt] = task_output

            print("\nResults:")
            if include_validation and "validation_loss" in eval_output:
                val_metrics = eval_output["validation_loss"]
                print(f"  validation_loss: {val_metrics['loss']:.4f}")
            res_dict = task_output.get("results", {})
            for task_name, metrics in res_dict.items():
                print(f"  Task: {task_name}")
                if "acc_norm,none" in metrics:
                    print(f"    acc_norm: {metrics['acc_norm,none']:.4f}")
                elif "acc_norm" in metrics:
                    print(f"    acc_norm: {metrics['acc_norm']:.4f}")
                if "acc,none" in metrics:
                    print(f"    acc:      {metrics['acc,none']:.4f}")
                elif "acc" in metrics:
                    print(f"    acc:      {metrics['acc']:.4f}")
            skipped_tasks = task_output.get("skipped_tasks", {})
            for task_name, reason in skipped_tasks.items():
                print(f"  Skipped task: {task_name}")
                print(f"    reason: {reason}")

            print("-" * 80)
            del lm_obj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print0(master_process, f"Skipping missing checkpoint: {ckpt}")
            continue

        print0(master_process, f"\nEvaluating Checkpoint: {ckpt}")
        print0(master_process, "=" * 80)

        lm_obj = OBPMWrapper(
            model_path=ckpt,
            device=device,
            batch_size=1,
            torch_compile=torch_compile,
            torch_compile_max_autotune=torch_compile_max_autotune,
            torch_compile_cache_dir=torch_compile_cache_dir,
            verbose=master_process,
        )
        eval_output = {}

        if include_validation:
            val_metrics = run_validation_loss(
                lm_obj,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                master_process=master_process,
            )
            eval_output["validation_loss"] = val_metrics

        if include_tasks and master_process:
            task_output = run_downstream_tasks(lm_obj, valid_tasks, device)
            task_output.update(eval_output)
            eval_output = task_output

        if master_process:
            results[ckpt] = eval_output

        print0(master_process, "\nResults:")
        if include_validation and master_process:
            val_metrics = eval_output["validation_loss"]
            print(f"  validation_loss: {val_metrics['loss']:.4f}")
        if include_tasks and master_process:
            res_dict = eval_output.get("results", {})
            for task_name, metrics in res_dict.items():
                print(f"  Task: {task_name}")
                if "acc_norm,none" in metrics:
                    print(f"    acc_norm: {metrics['acc_norm,none']:.4f}")
                elif "acc_norm" in metrics:
                    print(f"    acc_norm: {metrics['acc_norm']:.4f}")
                if "acc,none" in metrics:
                    print(f"    acc:      {metrics['acc,none']:.4f}")
                elif "acc" in metrics:
                    print(f"    acc:      {metrics['acc']:.4f}")
            skipped_tasks = eval_output.get("skipped_tasks", {})
            for task_name, reason in skipped_tasks.items():
                print(f"  Skipped task: {task_name}")
                print(f"    reason: {reason}")

        print0(master_process, "-" * 80)
        del lm_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_results_text(results: dict) -> str:
    lines = ["OBPM evaluation results", "=" * 80, ""]
    if not results:
        lines.append("No checkpoint results were produced.")
        return "\n".join(lines) + "\n"

    for ckpt, output in results.items():
        lines.append(f"Checkpoint: {ckpt}")
        lines.append("-" * 80)

        val_metrics = output.get("validation_loss")
        if val_metrics:
            lines.append(f"validation_loss: {_format_metric_value(val_metrics.get('loss'))}")
            lines.append(f"validation_tokens: {val_metrics.get('tokens')}")
            lines.append(f"validation_batches: {val_metrics.get('batches')}")

        task_results = output.get("results", {})
        for task_name, metrics in task_results.items():
            lines.append(f"Task: {task_name}")
            for metric_name, metric_value in sorted(metrics.items()):
                if isinstance(metric_value, (int, float, str, bool)):
                    lines.append(f"  {metric_name}: {_format_metric_value(metric_value)}")

        skipped_tasks = output.get("skipped_tasks", {})
        for task_name, reason in skipped_tasks.items():
            lines.append(f"Skipped task: {task_name}")
            lines.append(f"  reason: {reason}")

        lines.append("")

    return "\n".join(lines)


def write_results_file(results: dict, results_file: str):
    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(format_results_text(results))
    print(f"Saved evaluation results to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OLMES evaluations on OBPM checkpoints.")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        help="List of checkpoint paths (.pt files). If omitted, every checkpoint in --ckpt-dir is evaluated.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=DEFAULT_CKPT_DIR,
        help=f"Directory to scan when --ckpts is omitted (default: {DEFAULT_CKPT_DIR})",
    )
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help=f"Text file where evaluation results are saved (default: {DEFAULT_RESULTS_FILE})",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--validation-only",
        action="store_true",
        help="Only compute validation loss; skip downstream task evaluation.",
    )
    mode_group.add_argument(
        "--tasks-only",
        action="store_true",
        help="Only run downstream task evaluation; skip validation loss.",
    )
    parser.add_argument(
        "--torch_compile",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Compile the eval model with torch.compile.",
    )
    parser.add_argument("--no-torch_compile", dest="torch_compile", action="store_false")
    parser.add_argument(
        "--torch_compile_max_autotune",
        "--torch-max-autotune",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Compile the eval model with torch.compile(mode='max-autotune').",
    )
    parser.add_argument(
        "--no-torch_compile_max_autotune",
        "--no-torch-max-autotune",
        dest="torch_compile_max_autotune",
        action="store_false",
    )
    parser.add_argument(
        "--torch_compile_cache_dir",
        type=str,
        default=_DEFAULT_TORCH_COMPILE_CACHE_DIR,
        help="Directory for TorchInductor/Triton compile caches. Use a large persistent path for max-autotune.",
    )
    args = parser.parse_args()
    if args.torch_compile_max_autotune:
        args.torch_compile = True

    distributed, rank, world_size, local_rank = setup_distributed()

    downstream_eval_tasks = [
        "mmlu",
        "arc_challenge",
        "arc_easy",
        "commonsense_qa",
        "hellaswag",
        "openbookqa",
        "piqa",
        "siqa",
        "winogrande",
    ]

    checkpoints = args.ckpts if args.ckpts is not None else discover_checkpoints(args.ckpt_dir)
    if not checkpoints:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(
            f"No checkpoints found in {args.ckpt_dir!r}. "
            f"Expected files matching {os.path.join(args.ckpt_dir, CHECKPOINT_PATTERN)!r}."
        )

    results = evaluate_checkpoints(
        checkpoints,
        downstream_eval_tasks,
        include_validation=not args.tasks_only,
        include_tasks=not args.validation_only,
        torch_compile=args.torch_compile,
        torch_compile_max_autotune=args.torch_compile_max_autotune,
        torch_compile_cache_dir=args.torch_compile_cache_dir,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    if rank == 0:
        write_results_file(results, args.results_file)
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
