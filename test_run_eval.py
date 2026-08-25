import os
import json
import random
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import torch
import numpy as np

import run_eval
import utils
from criterion import get_criterion
from dataloader import (
    DataLoaderConfig,
    ResumableDistributedSampler,
    create_dataloaders,
    create_training_evaluation_dataloader,
    create_validation_dataloader,
)
from model import ModelConfig, OBPM
from tokenizer_utils import get_tiktoken_encoding


class RunEvalScoringTests(unittest.TestCase):
    def make_wrapper_without_checkpoint(self, block_size=8):
        wrapper = object.__new__(run_eval.OBPMWrapper)
        wrapper._device = torch.device("cpu")
        wrapper.max_length = block_size
        wrapper.eot_token_id = 0
        return wrapper

    def test_trailing_prompt_space_is_scored_as_part_of_continuation(self):
        wrapper = self.make_wrapper_without_checkpoint()
        wrapper.tokenizer = get_tiktoken_encoding("gpt-4")
        wrapper.eot_token_id = wrapper.tokenizer.eot_token

        context_ids, continuation_ids = wrapper._encode_pair("The answer is ", "Paris")

        self.assertTrue(context_ids)
        self.assertTrue(continuation_ids)
        self.assertEqual(
            wrapper.tokenizer.decode(context_ids + continuation_ids),
            "The answer is Paris",
        )

    def test_artificial_multiple_choice_selects_the_known_nonfirst_answer(self):
        wrapper = self.make_wrapper_without_checkpoint(block_size=32)
        wrapper.tokenizer = get_tiktoken_encoding("gpt-4")
        wrapper.eot_token_id = wrapper.tokenizer.eot_token

        _, paris_ids = wrapper._encode_pair("The answer is ", "Paris")
        _, london_ids = wrapper._encode_pair("The answer is ", "London")
        self.assertEqual(len(paris_ids), 1)
        self.assertEqual(len(london_ids), 1)

        paris_id = paris_ids[0]
        london_id = london_ids[0]

        class KnownChoiceModel(torch.nn.Module):
            def forward(_self, input_ids):
                logits = torch.full(
                    (1, input_ids.size(1), wrapper.tokenizer.n_vocab),
                    -8.0,
                )
                logits[:, :, paris_id] = 1.0
                logits[:, :, london_id] = 5.0
                return logits

        wrapper.model = KnownChoiceModel()
        requests = [
            SimpleNamespace(args=("The answer is ", "Paris")),
            SimpleNamespace(args=("The answer is ", "London")),
        ]

        scores = [score for score, _ in wrapper.loglikelihood(requests)]

        # The old boundary bug returned [0.0, 0.0], so argmax selected Paris
        # merely because it was first. The known model prefers London.
        self.assertEqual(int(np.argmax(scores)), 1)
        self.assertGreater(scores[1], scores[0])

    def test_scoring_window_matches_shifted_training_inputs_and_targets(self):
        wrapper = self.make_wrapper_without_checkpoint(block_size=4)
        input_ids, target_ids = wrapper._prepare_loglikelihood_tokens(
            context_ids=[10, 11, 12],
            continuation_ids=[20, 21],
        )

        self.assertEqual(input_ids, [10, 11, 12, 20])
        self.assertEqual(target_ids, [20, 21])
        self.assertEqual(input_ids[-len(target_ids):], [12, 20])

    def test_long_continuation_is_rejected_instead_of_partially_scored(self):
        wrapper = self.make_wrapper_without_checkpoint(block_size=4)
        with self.assertRaisesRegex(ValueError, "refusing to score only a suffix"):
            wrapper._prepare_loglikelihood_tokens(
                context_ids=[10, 11],
                continuation_ids=[20, 21, 22, 23, 24, 25],
            )

    def test_loglikelihood_matches_manual_shifted_float32_scoring(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_input = None
                self.last_logits = None

            def forward(self, input_ids):
                self.last_input = input_ids.detach().clone()
                positions = torch.arange(input_ids.size(1), dtype=torch.float32).view(1, -1, 1)
                tokens = torch.arange(8, dtype=torch.float32).view(1, 1, -1)
                self.last_logits = (positions * 0.25 + tokens * 0.5).to(torch.bfloat16)
                return self.last_logits

        wrapper = self.make_wrapper_without_checkpoint(block_size=4)
        wrapper.model = RecordingModel()
        wrapper._encode_pair = lambda _context, _continuation: ([1, 2], [3, 4])
        request = SimpleNamespace(args=("context", " continuation"))

        (score, is_greedy), = wrapper.loglikelihood([request])

        self.assertEqual(wrapper.model.last_input.tolist(), [[1, 2, 3]])
        expected_log_probs = torch.log_softmax(wrapper.model.last_logits.float(), dim=-1)
        expected = expected_log_probs[0, -2:, :].gather(
            1,
            torch.tensor([[3], [4]]),
        ).sum().item()
        self.assertAlmostEqual(score, expected, places=7)
        self.assertFalse(is_greedy)

    def test_rolling_loglikelihood_never_includes_the_target_in_input(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, input_ids):
                self.inputs.append(input_ids.detach().clone().tolist()[0])
                return torch.zeros(
                    (1, input_ids.size(1), 8),
                    dtype=torch.bfloat16,
                )

        class FixedTokenizer:
            @staticmethod
            def encode(_text):
                return [2, 3, 4]

        wrapper = self.make_wrapper_without_checkpoint(block_size=2)
        wrapper.model = RecordingModel()
        wrapper.tokenizer = FixedTokenizer()
        request = SimpleNamespace(args=("text",))

        (score,) = wrapper.loglikelihood_rolling([request])

        self.assertEqual(wrapper.model.inputs, [[0], [0, 2], [2, 3]])
        self.assertAlmostEqual(score, -3.0 * np.log(8.0), places=6)

    def test_generate_until_honors_string_earliest_stop_eos_and_sampling_args(self):
        wrapper = self.make_wrapper_without_checkpoint(block_size=16)
        wrapper.eot_token_id = 99

        class FixedTokenizer:
            @staticmethod
            def encode(_text):
                return [1, 2]

            @staticmethod
            def decode(tokens):
                if tokens == [7, 8]:
                    return "START HELLO STOP trailing END"
                if tokens == [7]:
                    return "before-eos"
                raise AssertionError(tokens)

        class RecordingGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
                self.outputs = [[7, 8], [7, 99, 8]]

            def generate(self, x, max_new_tokens, temperature, top_k):
                self.calls.append((temperature, top_k))
                suffix = torch.tensor([self.outputs.pop(0)], dtype=torch.long)
                return torch.cat((x, suffix), dim=1)

        wrapper.tokenizer = FixedTokenizer()
        wrapper.model = RecordingGenerator()
        first, second = wrapper.generate_until(
            [
                SimpleNamespace(
                    args=(
                        "prompt",
                        {
                            "until": ["END", "STOP"],
                            "do_sample": True,
                            "temperature": 0.8,
                            "top_k": 17,
                        },
                    )
                ),
                SimpleNamespace(args=("prompt", {"until": "STOP"})),
            ]
        )

        self.assertEqual(first, "START HELLO ")
        self.assertEqual(second, "before-eos")
        self.assertEqual(wrapper.model.calls, [(0.8, 17), (0.0, None)])


class EvaluationProtocolTests(unittest.TestCase):
    def test_siqa_alias_resolves_to_registered_social_iqa_name(self):
        self.assertEqual(run_eval.TASK_MAPPING["siqa"], "social_iqa")
        self.assertIn("social_iqa", run_eval.DEFAULT_TASKS)
        self.assertNotIn("siqa", run_eval.DEFAULT_TASKS)

    def test_task_output_declares_primary_metric_and_protocol(self):
        fake_manager = SimpleNamespace(all_tasks=["piqa"])
        fake_output = {
            "results": {
                "piqa": {
                    "acc,none": 0.0,
                    "acc_norm,none": 1.0,
                }
            },
            "n-samples": {"piqa": {"original": 1838, "effective": 1}},
            "n-shot": {"piqa": 5},
            "versions": {"piqa": 1.0},
        }
        with mock.patch.object(run_eval, "TaskManager", return_value=fake_manager), mock.patch.object(
            run_eval,
            "simple_evaluate",
            return_value=fake_output,
        ) as evaluate:
            output = run_eval.run_downstream_tasks(object(), ["piqa"], "cpu", limit=1)

        self.assertEqual(output["primary_metrics"]["piqa"]["metric"], "acc_norm")
        self.assertEqual(output["primary_metrics"]["piqa"]["value"], 1.0)
        self.assertEqual(output["protocol"]["completed_tasks"], ["piqa"])
        self.assertEqual(output["protocol"]["limit"], 1)
        self.assertEqual(evaluate.call_args.kwargs["fewshot_random_seed"], 1234)

    def test_mmlu_group_aggregate_is_preserved_and_reported(self):
        fake_manager = SimpleNamespace(all_tasks=["mmlu"])
        fake_output = {
            "results": {"mmlu_abstract_algebra": {"acc,none": 0.4}},
            "groups": {"mmlu": {"acc,none": 0.5}},
            "n-samples": {"mmlu_abstract_algebra": {"effective": 5}},
        }
        with mock.patch.object(run_eval, "TaskManager", return_value=fake_manager), mock.patch.object(
            run_eval,
            "simple_evaluate",
            return_value=fake_output,
        ):
            output = run_eval.run_downstream_tasks(object(), ["mmlu"], "cpu")

        report = run_eval.format_results_text({"checkpoint.pt": output})
        self.assertEqual(output["primary_metrics"]["mmlu"]["value"], 0.5)
        self.assertIn("Group: mmlu", report)
        self.assertIn("Primary metric: mmlu", report)

    def test_unknown_task_and_missing_checkpoint_fail_before_evaluation(self):
        fake_manager = SimpleNamespace(all_tasks=["piqa"])
        with mock.patch.object(run_eval, "TaskManager", return_value=fake_manager):
            with self.assertRaisesRegex(ValueError, "unknown_task"):
                run_eval._preflight_tasks(["unknown_task"])

        with self.assertRaises(FileNotFoundError):
            run_eval.evaluate_checkpoints(
                ["definitely-missing-checkpoint.pt"],
                [],
                include_validation=True,
                include_tasks=False,
            )

    def test_atomic_text_json_and_status_outputs_mark_completion(self):
        results = {
            "checkpoint.pt": {
                "results": {"piqa": {"acc_norm,none": 1.0}},
                "primary_metrics": {
                    "piqa": {"metric": "acc_norm", "output_key": "acc_norm,none", "value": 1.0}
                },
                "protocol": {
                    "requested_tasks": ["piqa"],
                    "completed_tasks": ["piqa"],
                },
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = os.path.join(temp_dir, "results.txt")
            json_path = run_eval.write_results_file(results, text_path)
            status_path = run_eval.write_run_status(text_path, "complete")
            with open(json_path, encoding="utf-8") as result_file:
                structured = json.load(result_file)
            with open(status_path, encoding="utf-8") as status_file:
                status = json.load(status_file)
            leftovers = [name for name in os.listdir(temp_dir) if name.endswith(".tmp")]

        self.assertTrue(structured["complete"])
        self.assertEqual(status["status"], "complete")
        self.assertEqual(leftovers, [])

    def test_limited_run_is_never_marked_complete(self):
        results = {
            "checkpoint.pt": {
                "protocol": {
                    "requested_tasks": ["piqa"],
                    "completed_tasks": ["piqa"],
                    "limit": 1,
                }
            }
        }
        self.assertFalse(run_eval.results_suite_complete(results))
        self.assertEqual(run_eval.results_run_status(results), "limited")

    def test_json_named_text_path_cannot_overwrite_structured_output(self):
        results = {"checkpoint.pt": {"protocol": {}}}
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = os.path.join(temp_dir, "results.json")
            structured_path = run_eval.write_results_file(results, text_path)
            with open(text_path, encoding="utf-8") as text_file:
                text_report = text_file.read()
            with open(structured_path, encoding="utf-8") as structured_file:
                structured = json.load(structured_file)

        self.assertNotEqual(text_path, structured_path)
        self.assertTrue(text_report.startswith("OBPM evaluation results"))
        self.assertIn("checkpoint_results", structured)

    def test_provenance_source_hash_covers_dirty_evaluator_code(self):
        source_hash = run_eval._evaluation_source_sha256()
        self.assertRegex(source_hash, r"^[0-9a-f]{64}$")
        self.assertIn(run_eval._git_dirty(), (True, False, None))

    def test_default_result_paths_are_unique_and_git_lookup_is_cwd_independent(self):
        self.assertNotEqual(run_eval._default_results_file(), run_eval._default_results_file())
        expected_commit = run_eval._git_commit()
        previous_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            actual_commit = run_eval._git_commit()
        finally:
            os.chdir(previous_cwd)
        self.assertEqual(actual_commit, expected_commit)


class SharedLoadingTests(unittest.TestCase):
    def assert_nested_exact(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(torch.equal(actual, expected))
            return
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_exact(actual[key], expected[key])
            return
        if isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_nested_exact(actual_item, expected_item)
            return
        self.assertEqual(actual, expected)

    def test_compiled_checkpoint_loads_through_shared_training_eval_path(self):
        config = ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_hidden_dim=32,
            vocab_size=32,
            block_size=8,
            flash_attention=False,
        )
        torch.manual_seed(1234)
        source_model = OBPM(config).eval()
        sample = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.inference_mode():
            expected_logits = source_model(sample)
        compiled_state = {
            f"_orig_mod.{key}": value.clone()
            for key, value in source_model.state_dict().items()
        }
        payload = {
            "step": 7,
            "tokens_processed": 123456,
            "train_batches_consumed": 56,
            "model_args": asdict(config),
            "model": compiled_state,
            "config": {"tokenizer_model": "gpt-4"},
            "muon_optimizer": {"state": {0: {"momentum_buffer": torch.randn(3, 3)}}},
            "adamw_optimizer": {"state": {1: {"step": torch.tensor(7.0)}}},
            "muon_scheduler": {"last_epoch": 7},
            "adamw_scheduler": {"last_epoch": 7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "ckpt_step:7.pt")
            torch.save(payload, checkpoint_path)
            checkpoint, loaded_model, loaded_config = utils.load_model_checkpoint(
                checkpoint_path,
                torch.device("cpu"),
                verbose=False,
            )

        self.assertEqual(checkpoint["step"], 7)
        self.assertEqual(checkpoint["tokens_processed"], 123456)
        self.assertEqual(checkpoint["train_batches_consumed"], 56)
        self.assert_nested_exact(checkpoint["model"], compiled_state)
        self.assert_nested_exact(checkpoint["muon_optimizer"], payload["muon_optimizer"])
        self.assert_nested_exact(checkpoint["adamw_optimizer"], payload["adamw_optimizer"])
        self.assert_nested_exact(checkpoint["muon_scheduler"], payload["muon_scheduler"])
        self.assert_nested_exact(checkpoint["adamw_scheduler"], payload["adamw_scheduler"])
        self.assertEqual(loaded_config, config)
        for key, expected in source_model.state_dict().items():
            actual = loaded_model.state_dict()[key]
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected), key)

        loaded_model.eval()
        with torch.inference_mode():
            actual_logits = loaded_model(sample)
        self.assertTrue(torch.equal(actual_logits, expected_logits))

    def test_baseline_cached_generation_uses_prefill_next_token_logits(self):
        config = ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_hidden_dim=32,
            vocab_size=32,
            block_size=8,
            flash_attention=False,
            use_attnres=False,
        )
        torch.manual_seed(5)
        model = OBPM(config).eval()
        prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            expected = model(prompt)[:, -1, :].argmax(dim=-1)
            generated = model.generate(prompt, max_new_tokens=1, temperature=0.0)

        self.assertTrue(torch.equal(generated[:, -1], expected))

    def test_bfloat16_training_precision_round_trip_is_bit_exact(self):
        config = ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_hidden_dim=32,
            vocab_size=32,
            block_size=8,
            flash_attention=False,
            use_attnres=True,
            attnres_type="block",
            attnres_num_blocks=1,
            attnres_block_alpha_learned=True,
            attnres_block_beta_learned=True,
        )
        source_model = OBPM(config).to_mixed_precision(dtype=torch.bfloat16)
        # Simulate fp32 optimizer updates that are not exactly representable in
        # bf16. A lossy post-load mixed-precision conversion changes these.
        with torch.no_grad():
            source_model.transformer.attnres_block_alphas.fill_(0.123456791)
            source_model.transformer.attnres_block_betas.fill_(0.987654328)
        payload = {
            "step": 3,
            "model_args": asdict(config),
            "model": {key: value.clone() for key, value in source_model.state_dict().items()},
            "config": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "ckpt_step:3.pt")
            torch.save(payload, checkpoint_path)
            checkpoint, loaded_model, _ = utils.load_model_checkpoint(
                checkpoint_path,
                torch.device("cpu"),
                verbose=False,
            )

        # This is the same post-load conversion train.py and CUDA eval perform.
        loaded_model.to_mixed_precision(dtype=torch.bfloat16)
        for key, expected in source_model.state_dict().items():
            saved = checkpoint["model"][key]
            actual = loaded_model.state_dict()[key]
            self.assertEqual(saved.dtype, expected.dtype)
            self.assertTrue(torch.equal(saved, expected), f"saved: {key}")
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected), f"loaded: {key}")

    def test_eval_load_discards_optimizer_state_without_changing_model(self):
        config = ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_hidden_dim=32,
            vocab_size=32,
            block_size=8,
            flash_attention=False,
        )
        source_model = OBPM(config)
        payload = {
            "step": 9,
            "model_args": asdict(config),
            "model": source_model.state_dict(),
            "config": {"tokenizer_model": "gpt-4"},
            "muon_optimizer": {"large_unused_state": torch.randn(128, 128)},
            "adamw_optimizer": {"large_unused_state": torch.randn(128, 128)},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            torch.save(payload, checkpoint_path)
            checkpoint, loaded_model, _ = utils.load_model_checkpoint(
                checkpoint_path,
                torch.device("cpu"),
                verbose=False,
                load_training_state=False,
            )

        self.assertNotIn("muon_optimizer", checkpoint)
        self.assertNotIn("adamw_optimizer", checkpoint)
        for name, expected in source_model.state_dict().items():
            self.assertTrue(torch.equal(loaded_model.state_dict()[name], expected), name)

    def test_training_and_validation_build_identical_loader_configs(self):
        config = {
            "dataset_dir": "dataset",
            "batch_size": 3,
            "block_size": 16,
            "grad_accum_steps": 2,
            "use_doc_masking": True,
            "doc_separator_token": 99,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "data_dtype": "uint32",
            "rank": 1,
            "world_size": 2,
            "prefetch_factor": 7,
        }

        with mock.patch.object(utils, "create_dataloaders", side_effect=lambda cfg: (cfg, cfg)):
            _, training_val_config = utils.get_dataloader(config)
        with mock.patch.object(utils, "create_validation_dataloader", side_effect=lambda cfg: cfg):
            standalone_val_config = utils.get_validation_dataloader(config)

        self.assertEqual(training_val_config, standalone_val_config)
        self.assertEqual(training_val_config.prefetch_factor, 7)

    def test_train_diagnostic_loader_cannot_move_training_sampler_cursor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (np.arange(257, dtype=np.uint32) % 32).tofile(
                os.path.join(temp_dir, "finewebedu_train_000.bin")
            )
            (np.arange(257, 322, dtype=np.uint32) % 32).tofile(
                os.path.join(temp_dir, "finewebedu_val_000.bin")
            )
            config = DataLoaderConfig(
                data_dir=temp_dir,
                batch_size=2,
                block_size=4,
                grad_accum_steps=1,
                use_doc_masking=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                seed=42,
            )
            train_loader, _ = create_dataloaders(config)
            diagnostic_loader = create_training_evaluation_dataloader(
                config,
                train_loader.dataset,
            )
            self.assertIsNot(train_loader.sampler, diagnostic_loader.sampler)

            train_loader.sampler.set_epoch(3)
            live_iterator = iter(train_loader)
            first_live = next(live_iterator)

            diagnostic_loader.sampler.set_epoch(99)
            next(iter(diagnostic_loader))
            second_live = next(live_iterator)

            reference_sampler = ResumableDistributedSampler(
                train_loader.dataset,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=42,
                drop_last=True,
            )
            reference_sampler.set_epoch(3)
            reference_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=2,
                sampler=reference_sampler,
                drop_last=True,
                collate_fn=train_loader.collate_fn,
            )
            reference_iterator = iter(reference_loader)
            first_reference = next(reference_iterator)
            second_reference = next(reference_iterator)

        self.assertTrue(torch.equal(first_live[0], first_reference[0]))
        self.assertTrue(torch.equal(second_live[0], second_reference[0]))

    def test_training_and_standalone_validation_load_identical_real_batches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_tokens = np.arange(97, dtype=np.uint32)
            validation_tokens = np.arange(1000, 1065, dtype=np.uint32)
            train_tokens.tofile(os.path.join(temp_dir, "finewebedu_train_000.bin"))
            validation_tokens.tofile(os.path.join(temp_dir, "finewebedu_val_000.bin"))
            config = DataLoaderConfig(
                data_dir=temp_dir,
                batch_size=3,
                block_size=4,
                grad_accum_steps=2,
                use_doc_masking=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                seed=42,
            )

            _, training_val_loader = create_dataloaders(config)
            standalone_val_loader = create_validation_dataloader(config)
            training_batches = list(training_val_loader)
            standalone_batches = list(standalone_val_loader)

        self.assertEqual(len(training_batches), len(standalone_batches))
        for training_batch, standalone_batch in zip(training_batches, standalone_batches):
            self.assertTrue(torch.equal(training_batch[0], standalone_batch[0]))
            self.assertTrue(torch.equal(training_batch[1], standalone_batch[1]))

    def test_real_obpm_training_and_standalone_validation_metrics_are_identical(self):
        model_config = ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_hidden_dim=32,
            vocab_size=32,
            block_size=4,
            flash_attention=False,
        )
        torch.manual_seed(31415)
        source_model = OBPM(model_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            (np.arange(65, dtype=np.uint32) % 32).tofile(
                os.path.join(temp_dir, "finewebedu_train_000.bin")
            )
            (np.arange(96, 161, dtype=np.uint32) % 32).tofile(
                os.path.join(temp_dir, "finewebedu_val_000.bin")
            )
            checkpoint_config = {
                "dataset_dir": temp_dir,
                "batch_size": 3,
                "block_size": 4,
                "grad_accum_steps": 2,
                "use_doc_masking": False,
                "doc_separator_token": 31,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "data_dtype": "uint32",
                "rank": 0,
                "world_size": 1,
                "seed": 42,
                "ignore_index": -100,
                "reduction": "mean",
                "z_loss": False,
                "z_loss_weight": 0.0,
                "ce_inplace_backward": False,
                "lm_head_chunk_size": 0,
                "flash_attention": False,
                "tokenizer_model": "gpt-4",
            }
            checkpoint_path = os.path.join(temp_dir, "ckpt_step:1.pt")
            utils.atomic_torch_save(
                {
                    "step": 1,
                    "model_args": asdict(model_config),
                    "model": source_model.state_dict(),
                    "config": checkpoint_config,
                },
                checkpoint_path,
            )

            _, training_model, _ = utils.load_model_checkpoint(
                checkpoint_path,
                torch.device("cpu"),
                verbose=False,
            )
            _, training_val_loader = utils.get_dataloader(checkpoint_config)
            self.assertTrue(training_model.training)
            training_metrics = utils.compute_validation_loss(
                training_model,
                get_criterion(checkpoint_config),
                training_val_loader,
                torch.device("cpu"),
                model_config.vocab_size,
                use_doc_masking=False,
            )
            self.assertTrue(training_model.training)

            standalone_wrapper = run_eval.OBPMWrapper(
                checkpoint_path,
                device="cpu",
                verbose=False,
            )
            self.assertFalse(standalone_wrapper.model.training)
            standalone_metrics = run_eval.run_validation_loss(standalone_wrapper)
            self.assertFalse(standalone_wrapper.model.training)

        self.assertEqual(standalone_metrics, training_metrics)

    def test_lr_attnres_forward_is_identical_in_train_and_validation_modes(self):
        config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=16,
            mlp_hidden_dim=32,
            vocab_size=32,
            block_size=8,
            flash_attention=False,
            use_attnres=True,
            use_fused_attnres=True,
            attnres_type="block",
            attnres_num_blocks=2,
            use_lrid=True,
            lrid_rank=4,
        )
        torch.manual_seed(2718)
        model = OBPM(config)
        sample = torch.randint(0, config.vocab_size, (2, config.block_size))

        model.train()
        with torch.no_grad():
            training_logits = model(sample)
        model.eval()
        with torch.no_grad():
            validation_logits = model(sample)

        self.assertTrue(torch.equal(training_logits, validation_logits))

    def test_grad_enabled_attnres_training_forward_matches_validation_forward(self):
        config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=16,
            mlp_hidden_dim=32,
            vocab_size=32,
            block_size=8,
            flash_attention=False,
            use_attnres=True,
            use_fused_attnres=True,
            attnres_type="block",
            attnres_num_blocks=2,
            use_lrid=True,
            lrid_rank=4,
        )
        torch.manual_seed(1729)
        model = OBPM(config)
        sample = torch.randint(0, config.vocab_size, (2, config.block_size))

        model.train()
        training_logits = model(sample)
        training_logits.float().square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(sample)

        torch.testing.assert_close(training_logits.detach(), validation_logits, rtol=0, atol=0)

    def test_rng_state_round_trip_replays_the_next_random_values(self):
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        state = utils.capture_rng_state()

        expected = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )
        random.random()
        np.random.random(10)
        torch.rand(10)

        self.assertTrue(utils.restore_rng_state(state))
        actual = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )

        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_atomic_checkpoint_save_round_trip_is_exact(self):
        payload = {
            "step": 11,
            "tensor": torch.randn(5, 7, dtype=torch.bfloat16),
            "nested": {"values": [1, 2, 3], "flag": True},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "ckpt_step:11.pt")
            utils.atomic_torch_save(payload, checkpoint_path)
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            leftovers = [name for name in os.listdir(temp_dir) if name.endswith(".tmp")]

        self.assert_nested_exact(loaded, payload)
        self.assertEqual(leftovers, [])

    def test_resumable_sampler_restarts_at_the_exact_next_index(self):
        dataset = list(range(31))
        sampler = ResumableDistributedSampler(
            dataset,
            num_replicas=2,
            rank=1,
            shuffle=True,
            seed=42,
            drop_last=True,
        )
        sampler.set_epoch(5)
        uninterrupted_indices = list(sampler)

        resumed_sampler = ResumableDistributedSampler(
            dataset,
            num_replicas=2,
            rank=1,
            shuffle=True,
            seed=42,
            drop_last=True,
        )
        resumed_sampler.set_epoch(5)
        resumed_sampler.set_start_index(8)

        self.assertEqual(list(resumed_sampler), uninterrupted_indices[8:])

    def test_exact_resume_rejects_changed_data_topology(self):
        checkpoint = {
            "train_batches_consumed": 2,
            "config": {
                "dataset_dir": "dataset",
                "batch_size": 3,
                "block_size": 8,
                "grad_accum_steps": 2,
                "world_size": 1,
                "seed": 42,
                "use_doc_masking": False,
                "data_dtype": "uint32",
                "doc_separator_token": 31,
            },
        }
        changed_config = dict(checkpoint["config"], batch_size=4)

        with self.assertRaisesRegex(ValueError, "batch_size"):
            utils.validate_exact_resume_data_config(checkpoint, changed_config)

        utils.validate_exact_resume_data_config(checkpoint, checkpoint["config"])

    def test_exact_resume_rejects_changed_objective_schedule_and_shards(self):
        saved_config = {
            "dataset_dir": "dataset",
            "batch_size": 3,
            "block_size": 8,
            "grad_accum_steps": 2,
            "world_size": 1,
            "seed": 42,
            "use_doc_masking": False,
            "data_dtype": "uint32",
            "doc_separator_token": 31,
            "z_loss_weight": 1e-5,
            "grad_clip": 1.0,
            "max_steps": 1000,
            "flash_attention": False,
            "torch_compile": False,
        }
        manifest = [{"path": "/data/train.bin", "size": 100, "mtime_ns": 5, "inode": 10}]
        checkpoint = {
            "train_batches_consumed": 2,
            "config": saved_config,
            "train_data_manifest": manifest,
        }
        for changed in (
            dict(saved_config, z_loss_weight=1e-2),
            dict(saved_config, grad_clip=0.0),
            dict(saved_config, max_steps=2000),
            dict(saved_config, flash_attention=True),
            dict(saved_config, torch_compile=True),
        ):
            with self.assertRaises(ValueError):
                utils.validate_exact_resume_data_config(
                    checkpoint,
                    changed,
                    current_data_manifest=manifest,
                )

        changed_manifest = [dict(manifest[0], mtime_ns=6)]
        with self.assertRaisesRegex(ValueError, "ordered training shards"):
            utils.validate_exact_resume_data_config(
                checkpoint,
                saved_config,
                current_data_manifest=changed_manifest,
            )

    def test_rank_specific_rng_state_selection_is_lossless(self):
        states = [{"rank": 0}, {"rank": 1}]
        checkpoint = {"rng_state": states[0], "rng_states_by_rank": states}
        self.assertIs(utils.checkpoint_rng_state_for_rank(checkpoint, 1), states[1])
        with self.assertRaisesRegex(ValueError, "rank 2"):
            utils.checkpoint_rng_state_for_rank(checkpoint, 2)
        self.assertEqual(
            utils.checkpoint_rng_state_for_rank({"rng_state": states[0]}, 1),
            states[0],
        )

    def test_exact_resume_rejects_changed_kernel_runtime(self):
        runtime = {
            "torch_version": torch.__version__,
            "device_type": "cpu",
            "attnres_kernel_environment": {"ATTNRES_TRAIN_KERNEL": "auto"},
        }
        checkpoint = {"training_runtime": runtime}
        utils.validate_training_runtime(checkpoint, runtime)
        changed = dict(
            runtime,
            attnres_kernel_environment={"ATTNRES_TRAIN_KERNEL": "torch"},
        )
        with self.assertRaisesRegex(ValueError, "kernel environment"):
            utils.validate_training_runtime(checkpoint, changed)

    def test_interrupted_training_resume_matches_uninterrupted_next_step(self):
        torch.manual_seed(2026)
        features = torch.randn(12, 4)
        targets = torch.randn(12, 2)

        def make_training_state():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.Dropout(p=0.25),
                torch.nn.Linear(8, 2),
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            return model, optimizer, scheduler

        def train_step(model, optimizer, scheduler, indices):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features[indices])
            loss = torch.nn.functional.mse_loss(prediction, targets[indices])
            loss.backward()
            optimizer.step()
            scheduler.step()

        sampler = ResumableDistributedSampler(
            list(range(len(features))),
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=42,
            drop_last=True,
        )
        sampler.set_epoch(0)
        epoch_indices = list(sampler)
        first_batch = epoch_indices[:3]
        second_batch = epoch_indices[3:6]

        uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler = make_training_state()
        train_step(
            uninterrupted_model,
            uninterrupted_optimizer,
            uninterrupted_scheduler,
            first_batch,
        )
        checkpoint_payload = {
            "model": uninterrupted_model.state_dict(),
            "optimizer": uninterrupted_optimizer.state_dict(),
            "scheduler": uninterrupted_scheduler.state_dict(),
            "rng_state": utils.capture_rng_state(),
            "train_batches_consumed": 1,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "resume.pt")
            utils.atomic_torch_save(checkpoint_payload, checkpoint_path)
            train_step(
                uninterrupted_model,
                uninterrupted_optimizer,
                uninterrupted_scheduler,
                second_batch,
            )
            expected_model_state = {
                key: value.clone() for key, value in uninterrupted_model.state_dict().items()
            }
            expected_optimizer_state = uninterrupted_optimizer.state_dict()
            expected_scheduler_state = uninterrupted_scheduler.state_dict()
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        resumed_model, resumed_optimizer, resumed_scheduler = make_training_state()
        resumed_model.load_state_dict(loaded["model"], strict=True)
        resumed_optimizer.load_state_dict(loaded["optimizer"])
        resumed_scheduler.load_state_dict(loaded["scheduler"])
        utils.restore_rng_state(loaded["rng_state"])

        resumed_sampler = ResumableDistributedSampler(
            list(range(len(features))),
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=42,
            drop_last=True,
        )
        resumed_sampler.set_epoch(0)
        resumed_sampler.set_start_index(loaded["train_batches_consumed"] * 3)
        resumed_second_batch = list(resumed_sampler)[:3]
        self.assertEqual(resumed_second_batch, second_batch)

        train_step(
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            resumed_second_batch,
        )
        self.assert_nested_exact(resumed_model.state_dict(), expected_model_state)
        self.assert_nested_exact(resumed_optimizer.state_dict(), expected_optimizer_state)
        self.assert_nested_exact(resumed_scheduler.state_dict(), expected_scheduler_state)


if __name__ == "__main__":
    unittest.main()
