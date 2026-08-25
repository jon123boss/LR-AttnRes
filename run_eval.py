# run_eval.py
#   python run_eval.py
#   python run_eval.py --ckpts out/ckpt_step:38146.pt
#   python run_eval.py --validation-only
#   python run_eval.py --tasks-only

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import List, Union

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

from tokenizer_utils import GPT4_TOKENIZER_MODEL as _GPT4_TOKENIZER_MODEL, get_tiktoken_encoding
import torch.distributed as dist

try:
    from lm_eval import simple_evaluate
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval.tasks import TaskManager
except ModuleNotFoundError as exc:
    if exc.name != "lm_eval":
        raise

    simple_evaluate = None
    TaskManager = None
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

from criterion import get_criterion
from dataloader import warmup_boundaries
from utils import (
    capture_attnres_kernel_environment,
    compute_validation_loss,
    get_validation_dataloader,
    load_model_checkpoint,
    unwrap_model,
)

DEFAULT_NUM_FEWSHOT = 5
DEFAULT_CKPT_DIR = "out"
CHECKPOINT_PATTERN = "ckpt_step:*.pt"
DATASET_SCRIPT_ERROR = "Dataset scripts are no longer supported"
EVAL_PROTOCOL_VERSION = "lr-attnres-lm-eval-5shot-v1"
EVAL_SEEDS = {
    "random_seed": 0,
    "numpy_random_seed": 1234,
    "torch_random_seed": 1234,
    "fewshot_random_seed": 1234,
}

DEFAULT_TASKS = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "commonsense_qa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "social_iqa",
    "winogrande",
]

# One declared paper-facing metric per task. Several harness tasks emit both
# raw and length-normalized accuracy; silently choosing between them can reverse
# an individual prediction and materially change a reported result.
PRIMARY_METRICS = {
    "mmlu": "acc",
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "boolq": "acc",
    "commonsense_qa": "acc",
    "hellaswag": "acc_norm",
    "openbookqa": "acc_norm",
    "piqa": "acc_norm",
    "social_iqa": "acc",
    "winogrande": "acc",
}

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
    "SIQA": "social_iqa",
    "siqa": "social_iqa",
    "social_iqa": "social_iqa",
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


def _merge_eval_outputs(combined_output: dict, task_output: dict):
    for key, value in task_output.items():
        if isinstance(value, dict):
            combined_output.setdefault(key, {}).update(value)
        else:
            combined_output[key] = value


def _preflight_tasks(valid_tasks: List[str]):
    if TaskManager is None:
        raise RuntimeError("lm_eval TaskManager is unavailable.")
    task_manager = TaskManager()
    missing = sorted(set(valid_tasks) - set(task_manager.all_tasks))
    if missing:
        raise ValueError(
            "Unknown lm-eval task name(s): " + ", ".join(missing)
        )
    return task_manager


def _metric_from_container(metrics: dict, metric_name: str):
    for key in (f"{metric_name},none", metric_name):
        if key in metrics:
            return key, metrics[key]
    return None, None


def _collect_primary_metrics(combined_output: dict, valid_tasks: List[str]):
    results = combined_output.get("results", {})
    groups = combined_output.get("groups", {})
    primary = {}
    for task in valid_tasks:
        metric_name = PRIMARY_METRICS.get(task)
        if metric_name is None:
            continue
        metrics = results.get(task) or groups.get(task)
        if not isinstance(metrics, dict):
            raise RuntimeError(
                f"Task {task!r} completed without an aggregate result needed for "
                f"its declared primary metric {metric_name!r}."
            )
        output_key, value = _metric_from_container(metrics, metric_name)
        if output_key is None:
            raise RuntimeError(
                f"Task {task!r} did not return declared primary metric {metric_name!r}."
            )
        primary[task] = {
            "metric": metric_name,
            "output_key": output_key,
            "value": value,
        }
    return primary


def _effective_sample_counts(n_samples):
    counts = []
    if isinstance(n_samples, dict):
        if "effective" in n_samples:
            counts.append(n_samples["effective"])
        else:
            for value in n_samples.values():
                counts.extend(_effective_sample_counts(value))
    return counts


def run_downstream_tasks(
    lm_obj: "OBPMWrapper",
    valid_tasks: List[str],
    device: str,
    allow_skipped: bool = False,
    limit=None,
):
    if simple_evaluate is None:
        raise RuntimeError(
            "run_eval.py requires lm_eval. Install it with `pip install lm_eval` "
            "before running downstream evaluations."
        ) from _LM_EVAL_IMPORT_ERROR

    task_manager = _preflight_tasks(valid_tasks)
    combined_output = {"results": {}}
    skipped_tasks = {}
    completed_tasks = []

    for task in valid_tasks:
        print(f"Running task: {task}")
        try:
            task_output = simple_evaluate(
                model=lm_obj,
                tasks=[task],
                num_fewshot=DEFAULT_NUM_FEWSHOT,
                batch_size=1,
                device=device,
                log_samples=False,
                limit=limit,
                task_manager=task_manager,
                **EVAL_SEEDS,
            )
        except RuntimeError as exc:
            if DATASET_SCRIPT_ERROR not in str(exc) or not allow_skipped:
                raise

            reason = (
                "dataset script is incompatible with the installed datasets package; "
                "install datasets<4 or omit this task"
            )
            print(f"Skipping task {task}: {reason}.")
            skipped_tasks[task] = reason
            continue

        if not isinstance(task_output, dict):
            raise RuntimeError(f"Task {task!r} returned no structured lm-eval output.")
        if not task_output.get("results") and not task_output.get("groups"):
            raise RuntimeError(f"Task {task!r} returned no result metrics.")
        sample_counts = _effective_sample_counts(task_output.get("n-samples", {}))
        if sample_counts and not any(float(count) > 0 for count in sample_counts):
            raise RuntimeError(f"Task {task!r} evaluated zero effective samples.")

        _merge_eval_outputs(combined_output, task_output)
        completed_tasks.append(task)

    if skipped_tasks:
        combined_output["skipped_tasks"] = skipped_tasks
    combined_output["protocol"] = {
        "version": EVAL_PROTOCOL_VERSION,
        "requested_tasks": list(valid_tasks),
        "completed_tasks": completed_tasks,
        "num_fewshot_override": DEFAULT_NUM_FEWSHOT,
        "batch_size": 1,
        "limit": limit,
        "allow_skipped": allow_skipped,
        "seeds": dict(EVAL_SEEDS),
    }
    combined_output["primary_metrics"] = _collect_primary_metrics(
        combined_output,
        completed_tasks,
    )

    return combined_output


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

        checkpoint, self.model, config = load_model_checkpoint(
            model_path,
            self._device,
            verbose=self.verbose,
            load_training_state=False,
        )
        self.checkpoint_config = checkpoint.get("config", {})
        self.checkpoint_step = checkpoint.get("step")
        self.checkpoint_tokens_processed = checkpoint.get("tokens_processed")

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

    def _encode_pair(self, context: str, continuation: str):
        # Match lm-eval's causal-tokenizer contract. A trailing prompt space can
        # merge with the first answer token, so it belongs to the continuation.
        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]

        if context:
            full_ids = self.tokenizer.encode(context + continuation)
            context_ids = self.tokenizer.encode(context)
            continuation_ids = full_ids[len(context_ids):]
        else:
            context_ids = [self.eot_token_id]
            continuation_ids = self.tokenizer.encode(continuation)

        return context_ids, continuation_ids

    def _prepare_loglikelihood_tokens(self, context_ids, continuation_ids):
        if not continuation_ids:
            raise ValueError("Continuation encoded to zero tokens; refusing to report a false 0.0 score.")
        if len(continuation_ids) > self.max_length:
            raise ValueError(
                f"Continuation has {len(continuation_ids)} tokens, exceeding the "
                f"model context limit of {self.max_length}; refusing to score only a suffix."
            )

        # Keep one conditioning token plus at most max_length target tokens.
        target_ids = continuation_ids
        combined = (context_ids + continuation_ids)[-(self.max_length + 1):]
        input_ids = combined[:-1]
        if not input_ids or len(target_ids) > len(input_ids):
            raise RuntimeError("Invalid causal scoring window.")
        return input_ids, target_ids

    def loglikelihood(self, requests):
        res = []

        for instance in tqdm(requests, desc="Evaluating (loglikelihood)", leave=False):
            context, continuation = instance.args
            context_ids, continuation_ids = self._encode_pair(context, continuation)
            input_ids, target_ids = self._prepare_loglikelihood_tokens(
                context_ids,
                continuation_ids,
            )
            continuation_length = len(target_ids)
            x = torch.tensor([input_ids], dtype=torch.long, device=self._device)

            with torch.inference_mode():
                logits = self.model(x)
                # Training validation computes CE from float32 logits too.
                log_probs = torch.log_softmax(logits.float(), dim=-1)

            target = torch.tensor(target_ids, dtype=torch.long, device=self._device)
            token_log_probs = log_probs[0, -continuation_length:, :]

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
            prefix = [self.eot_token_id]
            for target_id in ids:
                # Match training's next-token shift: the target itself is never
                # included in the model input, and all max_length positions are
                # available as conditioning context.
                window = prefix[-self.max_length :]
                x = torch.tensor([window], dtype=torch.long, device=self._device)

                with torch.inference_mode():
                    logits = self.model(x)
                    log_probs = torch.log_softmax(logits.float(), dim=-1)

                lp = log_probs[0, -1, target_id]
                total += float(lp.item())
                prefix.append(target_id)

            out.append(total)

        return out

    def generate_until(self, requests):
        res = []
        for instance in tqdm(requests, desc="Generating", leave=False):
            context, gen_kwargs = instance.args

            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            elif until is None:
                until = []
            else:
                until = list(until)
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            do_sample = bool(gen_kwargs.get("do_sample", False))
            temperature = float(gen_kwargs.get("temperature", 1.0 if do_sample else 0.0))
            if not do_sample:
                temperature = 0.0
            top_k = gen_kwargs.get("top_k")
            if top_k is not None:
                top_k = int(top_k)

            tokens = self.tokenizer.encode(context)
            if len(tokens) == 0:
                tokens = [self.eot_token_id]

            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length :]

            x = torch.tensor([tokens], dtype=torch.long, device=self._device)

            with torch.inference_mode():
                out_idx = unwrap_model(self.model).generate(
                    x,
                    max_new_tokens=max_gen_toks,
                    temperature=temperature,
                    top_k=top_k,
                )

            out = out_idx[0].tolist()
            new_tokens = out[len(x[0]) :]
            if self.eot_token_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(self.eot_token_id)]
            text = self.tokenizer.decode(new_tokens)

            stop_positions = [text.find(term) for term in until if term and term in text]
            if stop_positions:
                text = text[: min(stop_positions)]

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


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package_name: str):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_commit():
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return process.stdout.strip() or None


def _git_dirty():
    try:
        process = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(process.stdout.strip())


def _evaluation_source_sha256():
    digest = hashlib.sha256()
    source_root = os.path.dirname(os.path.abspath(__file__))
    source_files = (
        "attnres_ops.py",
        "criterion.py",
        "dataloader.py",
        "model.py",
        "run_eval.py",
        "tokenizer_utils.py",
        "utils.py",
    )
    for relative_path in source_files:
        path = os.path.join(source_root, relative_path)
        digest.update(relative_path.encode("utf-8") + b"\0")
        with open(path, "rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _attach_provenance(
    output: dict,
    checkpoint_path: str,
    lm_obj: OBPMWrapper,
    valid_tasks: List[str],
    include_validation: bool,
    include_tasks: bool,
    limit,
):
    output.setdefault("protocol", {})
    output["protocol"].update(
        {
            "version": EVAL_PROTOCOL_VERSION,
            "requested_tasks": list(valid_tasks),
            "include_validation": include_validation,
            "include_tasks": include_tasks,
            "num_fewshot_override": DEFAULT_NUM_FEWSHOT if include_tasks else None,
            "limit": limit if include_tasks else None,
            "seeds": dict(EVAL_SEEDS),
        }
    )
    output["provenance"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_step": lm_obj.checkpoint_step,
        "checkpoint_tokens_processed": lm_obj.checkpoint_tokens_processed,
        "tokenizer": lm_obj.tokenizer_name,
        "tokenizer_model": lm_obj.tokenizer_model,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "evaluation_source_sha256": _evaluation_source_sha256(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "lm_eval": _package_version("lm_eval"),
            "datasets": _package_version("datasets"),
            "tiktoken": _package_version("tiktoken"),
        },
        "device": str(lm_obj._device),
        "cuda_available": torch.cuda.is_available(),
        "attnres_kernel_environment": capture_attnres_kernel_environment(),
    }


def _print_metric_sections(output: dict):
    for section_label, section_key in (("Task", "results"), ("Group", "groups")):
        for name, metrics in output.get(section_key, {}).items():
            print(f"  {section_label}: {name}")
            if "acc_norm,none" in metrics:
                print(f"    acc_norm: {metrics['acc_norm,none']:.4f}")
            elif "acc_norm" in metrics:
                print(f"    acc_norm: {metrics['acc_norm']:.4f}")
            if "acc,none" in metrics:
                print(f"    acc:      {metrics['acc,none']:.4f}")
            elif "acc" in metrics:
                print(f"    acc:      {metrics['acc']:.4f}")
    for task_name, metric in output.get("primary_metrics", {}).items():
        print(
            f"  Primary: {task_name} {metric['metric']}="
            f"{_format_metric_value(metric['value'])}"
        )
    for task_name, reason in output.get("skipped_tasks", {}).items():
        print(f"  Skipped task: {task_name}")
        print(f"    reason: {reason}")


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
    allow_skipped: bool = False,
    limit=None,
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
    missing_checkpoints = [path for path in checkpoints if not os.path.isfile(path)]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Missing checkpoint(s): " + ", ".join(missing_checkpoints)
        )
    if include_tasks:
        # Fail before loading a large checkpoint or running earlier tasks.
        _preflight_tasks(valid_tasks)

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
            task_output = run_downstream_tasks(
                lm_obj,
                valid_tasks,
                device,
                allow_skipped=allow_skipped,
                limit=limit,
            )
            task_output.update(eval_output)
            _attach_provenance(
                task_output,
                ckpt,
                lm_obj,
                valid_tasks,
                include_validation,
                include_tasks,
                limit,
            )
            results[ckpt] = task_output

            print("\nResults:")
            if include_validation and "validation_loss" in eval_output:
                val_metrics = eval_output["validation_loss"]
                print(f"  validation_loss: {val_metrics['loss']:.4f}")
            _print_metric_sections(task_output)

            print("-" * 80)
            del lm_obj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    for ckpt in checkpoints:
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
            task_output = run_downstream_tasks(
                lm_obj,
                valid_tasks,
                device,
                allow_skipped=allow_skipped,
                limit=limit,
            )
            task_output.update(eval_output)
            eval_output = task_output

        if master_process:
            _attach_provenance(
                eval_output,
                ckpt,
                lm_obj,
                valid_tasks,
                include_validation,
                include_tasks,
                limit,
            )
            results[ckpt] = eval_output

        print0(master_process, "\nResults:")
        if include_validation and master_process:
            val_metrics = eval_output["validation_loss"]
            print(f"  validation_loss: {val_metrics['loss']:.4f}")
        if include_tasks and master_process:
            _print_metric_sections(eval_output)

        print0(master_process, "-" * 80)
        del lm_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _default_results_file():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = uuid.uuid4().hex[:8]
    return os.path.join(DEFAULT_CKPT_DIR, f"eval_results_{timestamp}_{run_id}.txt")


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

        protocol = output.get("protocol", {})
        if protocol:
            lines.append(f"protocol_version: {protocol.get('version')}")
            lines.append(f"requested_tasks: {protocol.get('requested_tasks')}")
            lines.append(f"completed_tasks: {protocol.get('completed_tasks')}")
            lines.append(f"num_fewshot_override: {protocol.get('num_fewshot_override')}")
            lines.append(f"limit: {protocol.get('limit')}")
            lines.append(f"seeds: {protocol.get('seeds')}")

        provenance = output.get("provenance", {})
        if provenance:
            for key in (
                "completed_at_utc",
                "checkpoint_sha256",
                "checkpoint_step",
                "checkpoint_tokens_processed",
                "tokenizer",
                "tokenizer_model",
                "git_commit",
                "git_dirty",
                "evaluation_source_sha256",
                "python",
                "platform",
                "device",
            ):
                lines.append(f"{key}: {provenance.get(key)}")
            lines.append(f"packages: {provenance.get('packages')}")

        for task_name, metric in output.get("primary_metrics", {}).items():
            lines.append(f"Primary metric: {task_name}")
            lines.append(f"  metric: {metric.get('metric')}")
            lines.append(f"  output_key: {metric.get('output_key')}")
            lines.append(f"  value: {_format_metric_value(metric.get('value'))}")

        for section_label, section_key in (("Task", "results"), ("Group", "groups")):
            for name, metrics in output.get(section_key, {}).items():
                lines.append(f"{section_label}: {name}")
                for metric_name, metric_value in sorted(metrics.items()):
                    if isinstance(metric_value, (int, float, str, bool)):
                        lines.append(f"  {metric_name}: {_format_metric_value(metric_value)}")

        skipped_tasks = output.get("skipped_tasks", {})
        for task_name, reason in skipped_tasks.items():
            lines.append(f"Skipped task: {task_name}")
            lines.append(f"  reason: {reason}")

        lines.append("")

    return "\n".join(lines)


def _json_default(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return repr(value)


def _atomic_write_text(path: str, contents: str):
    target_path = os.path.abspath(path)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target_path)}.",
        suffix=".tmp",
        dir=target_dir,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(contents)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, target_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def write_results_file(results: dict, results_file: str):
    results_json_file = os.path.splitext(results_file)[0] + ".json"
    if os.path.abspath(results_json_file) == os.path.abspath(results_file):
        results_json_file = results_file + ".structured.json"
    _atomic_write_text(results_file, format_results_text(results))
    structured_output = {
        "schema_version": 1,
        "complete": results_suite_complete(results),
        "checkpoint_results": results,
    }
    _atomic_write_text(
        results_json_file,
        json.dumps(structured_output, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    print(f"Saved evaluation results to: {results_file}")
    print(f"Saved structured evaluation results to: {results_json_file}")
    return results_json_file


def results_suite_complete(results: dict) -> bool:
    if not results:
        return False
    for output in results.values():
        if output.get("skipped_tasks"):
            return False
        protocol = output.get("protocol", {})
        if protocol.get("limit") is not None:
            return False
        requested = protocol.get("requested_tasks")
        completed = protocol.get("completed_tasks")
        if requested is not None and completed is not None and requested != completed:
            return False
    return True


def results_run_status(results: dict) -> str:
    if any(
        output.get("protocol", {}).get("limit") is not None
        for output in results.values()
    ):
        return "limited"
    return "complete" if results_suite_complete(results) else "partial"


def write_run_status(results_file: str, status: str, **details):
    status_file = results_file + ".status.json"
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    _atomic_write_text(
        status_file,
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    return status_file


def _parse_limit(value: str):
    try:
        parsed = int(value)
    except ValueError:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("limit must be a positive integer or fraction") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run validation loss and lm-eval downstream tasks on OBPM checkpoints."
    )
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
        default=None,
        help="Text result path. Default: a unique UTC-timestamped file under out/.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="lm-eval tasks to run (aliases such as SIQA and ARC-E are accepted).",
    )
    parser.add_argument(
        "--limit",
        type=_parse_limit,
        default=None,
        help="Evaluate only N samples, or a fractional subset below 1. Intended for smoke tests.",
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Explicitly permit incompatible dataset-script tasks to be skipped.",
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
    args.results_file = args.results_file or _default_results_file()
    downstream_eval_tasks = args.tasks

    checkpoints = args.ckpts if args.ckpts is not None else discover_checkpoints(args.ckpt_dir)
    if not checkpoints:
        if rank == 0:
            write_run_status(
                args.results_file,
                "failed",
                error_type="NoCheckpointsFound",
                error=(
                    f"No checkpoints found in {args.ckpt_dir!r}; expected "
                    f"{os.path.join(args.ckpt_dir, CHECKPOINT_PATTERN)!r}."
                ),
            )
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(
            f"No checkpoints found in {args.ckpt_dir!r}. "
            f"Expected files matching {os.path.join(args.ckpt_dir, CHECKPOINT_PATTERN)!r}."
        )

    if rank == 0:
        write_run_status(
            args.results_file,
            "running",
            checkpoints=[os.path.abspath(path) for path in checkpoints],
            requested_tasks=downstream_eval_tasks,
        )
    try:
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
            allow_skipped=args.allow_skipped,
            limit=args.limit,
        )
        if rank == 0:
            results_json_file = write_results_file(results, args.results_file)
            write_run_status(
                args.results_file,
                results_run_status(results),
                text_results_file=os.path.abspath(args.results_file),
                results_json_file=os.path.abspath(results_json_file),
            )
    except BaseException as exc:
        if rank == 0:
            write_run_status(
                args.results_file,
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        raise
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
