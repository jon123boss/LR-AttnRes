# run_eval.py
#   python run_eval.py --ckpts out/ckpt_step:38146.pt

import argparse
import os
from typing import List, Union

import torch
from tqdm import tqdm

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import tiktoken

from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from model import OBPM, ModelConfig

OLMES_DEFAULT_SHOTS = 5

TASK_MAPPING = {
    "arc_challenge": "arc_challenge",
    "arc_easy": "arc_easy",
    "boolq": "boolq",
    "commonsense_qa": "commonsense_qa",
    "hellaswag": "hellaswag",
    "openbookqa": "openbookqa",
    "piqa": "piqa",
    "winogrande": "winogrande",
    "mmlu": "mmlu",
}


def _parse_batch_size(batch_size: Union[int, str, None]) -> int:
    if isinstance(batch_size, int):
        return max(1, batch_size)
    if isinstance(batch_size, str) and batch_size.isdigit():
        return max(1, int(batch_size))
    return 1


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
        checkpoint = torch.load(model_path, map_location=self._device)

        model_args = checkpoint.get("model_args", {})
        if isinstance(model_args, dict):
            config = ModelConfig(**model_args)
        else:
            config = model_args

        self.model = OBPM(config)


        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self._device)
        self.model.eval()

        if hasattr(self.model, "to_mixed_precision"):
            self.model.to_mixed_precision(dtype=torch.bfloat16)

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token_id = self.tokenizer.eot_token

        self.vocab_size = int(config.vocab_size)
        self.max_length = int(config.block_size)

    @property
    def device(self):
        return str(self._device)

    def _truncate_left(self, ids: List[int], split: int) -> tuple[List[int], int]:
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

            with torch.no_grad():
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

            if len(ids) < 2:
                out.append(0.0)
                continue

            total = 0.0
            for t in range(1, len(ids)):
                start = max(0, (t + 1) - self.max_length)
                window = ids[start : t + 1]
                x = torch.tensor([window], dtype=torch.long, device=self._device)

                with torch.no_grad():
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

            with torch.no_grad():
                try:
                    out_idx = self.model.generate(x, max_new_tokens=max_gen_toks, temperature=0.0)
                except TypeError:
                    out_idx = self.model.generate(x, max_new_tokens=max_gen_toks)

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


def evaluate_checkpoints(checkpoints: List[str], tasks_list: List[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    valid_tasks = [TASK_MAPPING.get(t, t) for t in tasks_list]

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

        eval_output = simple_evaluate(
            model=lm_obj,
            tasks=valid_tasks,
            num_fewshot=OLMES_DEFAULT_SHOTS,
            batch_size=1,
            device=device,
        )

        results[ckpt] = eval_output

        print("\nResults:")
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

        print("-" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OLMES evaluations on OBPM checkpoints.")
    parser.add_argument("--ckpts", nargs="+", required=True, help="List of checkpoint paths (.pt files)")
    args = parser.parse_args()

    downstream_eval_tasks = [
        "arc_challenge",
        "arc_easy",
        "hellaswag",
        "openbookqa",
        "piqa",
        "winogrande",
    ]

    evaluate_checkpoints(args.ckpts, downstream_eval_tasks)

