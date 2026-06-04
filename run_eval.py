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

import torch
from tqdm import tqdm

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import tiktoken
from tokenizer_utils import GPT4_TOKENIZER_MODEL as _GPT4_TOKENIZER_MODEL, get_tiktoken_encoding

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
from utils import compute_validation_loss, get_validation_dataloader

OLMES_DEFAULT_SHOTS = 5
DEFAULT_CKPT_DIR = "out"
CHECKPOINT_PATTERN = "ckpt_step:*.pt"
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
    ):
        super().__init__()
        self._device = torch.device(device)

        self.batch_size_per_gpu = _parse_batch_size(batch_size)
        self.max_batch_size = max_batch_size

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
        self.model.eval()

        if self._device.type == "cuda" and hasattr(self.model, "to_mixed_precision"):
            self.model.to_mixed_precision(dtype=torch.bfloat16)

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
                out_idx = self.model.generate(x, max_new_tokens=max_gen_toks, temperature=0.0)

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


def run_validation_loss(lm_obj: OBPMWrapper):
    config = getattr(lm_obj, "checkpoint_config", None)
    if not config:
        raise RuntimeError(
            "Checkpoint does not contain a training config, so run_eval.py cannot "
            "build the validation dataloader for validation loss."
        )

    eval_config = dict(config)
    eval_config["pin_memory"] = bool(lm_obj._device.type == "cuda" and eval_config.get("pin_memory", False))

    val_loader = get_validation_dataloader(eval_config)
    if eval_config.get("use_doc_masking", False):
        print("Warming up validation document boundary cache...")
        warmup_boundaries(val_loader.dataset)
        print("Validation boundary warmup complete.")

    criterion = get_criterion(eval_config)
    val_metrics = compute_validation_loss(
        lm_obj.model,
        criterion,
        val_loader,
        lm_obj._device,
        lm_obj.vocab_size,
        use_doc_masking=eval_config.get("use_doc_masking", False),
    )
    print(
        f"Validation loss: {val_metrics['loss']:.4f} "
        f"({val_metrics['tokens']:,} tokens across {val_metrics['batches']:,} batches)"
    )
    print("-" * 80)
    return val_metrics


def evaluate_checkpoints(
    checkpoints: List[str],
    tasks_list: List[str],
    include_validation: bool = True,
    include_tasks: bool = True,
):
    if not include_validation and not include_tasks:
        raise ValueError("At least one evaluation mode must be enabled.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    valid_tasks = [TASK_MAPPING.get(t, t) for t in tasks_list] if include_tasks else []

    if include_validation and include_tasks:
        print("Evaluation mode: validation loss + downstream tasks")
    elif include_validation:
        print("Evaluation mode: validation loss only")
    else:
        print("Evaluation mode: downstream tasks only")

    if include_tasks:
        print(f"Tasks to evaluate: {valid_tasks}")
    print("-" * 80)

    results = {}

    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print(f"Skipping missing checkpoint: {ckpt}")
            continue

        print(f"\nEvaluating Checkpoint: {ckpt}")
        print("=" * 80)

        lm_obj = OBPMWrapper(model_path=ckpt, device=device, batch_size=1)
        eval_output = {}

        if include_validation:
            val_metrics = run_validation_loss(lm_obj)
            eval_output["validation_loss"] = val_metrics

        if include_tasks:
            task_output = run_downstream_tasks(lm_obj, valid_tasks, device)
            task_output.update(eval_output)
            eval_output = task_output

        results[ckpt] = eval_output

        print("\nResults:")
        if include_validation:
            val_metrics = eval_output["validation_loss"]
            print(f"  validation_loss: {val_metrics['loss']:.4f}")
        if include_tasks:
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

        print("-" * 80)

    return results


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
    args = parser.parse_args()

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
        raise SystemExit(
            f"No checkpoints found in {args.ckpt_dir!r}. "
            f"Expected files matching {os.path.join(args.ckpt_dir, CHECKPOINT_PATTERN)!r}."
        )

    evaluate_checkpoints(
        checkpoints,
        downstream_eval_tasks,
        include_validation=not args.tasks_only,
        include_tasks=not args.validation_only,
    )
