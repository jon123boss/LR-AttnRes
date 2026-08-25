import os
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
from dataloader import (
    DataLoaderConfig,
    ResumableDistributedSampler,
    create_dataloaders,
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

    def test_scoring_window_matches_shifted_training_inputs_and_targets(self):
        wrapper = self.make_wrapper_without_checkpoint(block_size=4)
        input_ids, target_ids = wrapper._prepare_loglikelihood_tokens(
            context_ids=[10, 11, 12],
            continuation_ids=[20, 21],
        )

        self.assertEqual(input_ids, [10, 11, 12, 20])
        self.assertEqual(target_ids, [20, 21])
        self.assertEqual(input_ids[-len(target_ids):], [12, 20])

    def test_long_continuation_uses_full_model_context_without_target_leakage(self):
        wrapper = self.make_wrapper_without_checkpoint(block_size=4)
        input_ids, target_ids = wrapper._prepare_loglikelihood_tokens(
            context_ids=[10, 11],
            continuation_ids=[20, 21, 22, 23, 24, 25],
        )

        self.assertEqual(input_ids, [21, 22, 23, 24])
        self.assertEqual(target_ids, [22, 23, 24, 25])
        self.assertEqual(len(input_ids), wrapper.max_length)

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

    def test_bfloat16_training_precision_round_trip_is_bit_exact(self):
        config = ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_hidden_dim=32,
            vocab_size=32,
            block_size=8,
            flash_attention=False,
        )
        source_model = OBPM(config).to_mixed_precision(dtype=torch.bfloat16)
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
        }

        with mock.patch.object(utils, "create_dataloaders", side_effect=lambda cfg: (cfg, cfg)):
            _, training_val_config = utils.get_dataloader(config)
        with mock.patch.object(utils, "create_validation_dataloader", side_effect=lambda cfg: cfg):
            standalone_val_config = utils.get_validation_dataloader(config)

        self.assertEqual(training_val_config, standalone_val_config)

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
