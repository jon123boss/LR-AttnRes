"""CPU regression tests for historical checkpoint reconstruction."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import torch
from torch.nn import functional as F

from checkpoint_config import model_config_from_checkpoint
import load_jonnester_model_to_out as importer
from model import ModelConfig, OBPM
import utils


PRIOR = "attnres_block_count_prior"


class CheckpointMigrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(291)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "ckpt_step:7.pt")
        self.tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def make_config(self, **overrides):
        args = dict(
            n_layer=3, n_head=2, n_embd=8, mlp_hidden_dim=16,
            vocab_size=19, block_size=4, flash_attention=False,
            norm_pos="before", use_attnres=True, attnres_type="block",
            attnres_num_blocks=2, attnres_block_average=False,
            attnres_block_count_prior=False, attn_res_query_init="normal",
            lrid_rank=4, lrid_use_logit_scale=False,
        )
        args.update(overrides)
        return ModelConfig(**args)

    def save_legacy(self, config, object_config=False, training_config=None):
        model = OBPM(config)
        if object_config:
            saved_args = ModelConfig(**asdict(config))
            del vars(saved_args)[PRIOR]
            # This is the pickle compatibility trap: ordinary attribute lookup
            # sees today's class default even though it was never serialized.
            self.assertTrue(getattr(saved_args, PRIOR))
        else:
            saved_args = asdict(config)
            saved_args.pop(PRIOR)
        torch.save({
            "step": 7, "model_args": saved_args, "model": model.state_dict(),
            "config": training_config or {},
        }, self.path)
        return model

    def assert_forward_and_gradients_equal(self, reference, restored):
        reference.train()
        restored.train()
        expected = reference(self.tokens)
        actual = restored(self.tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        targets = self.tokens.roll(-1, dims=1).reshape(-1)
        for logits, model in ((expected, reference), (actual, restored)):
            F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets).backward()
        for (name, param), (restored_name, other) in zip(
            reference.named_parameters(), restored.named_parameters()
        ):
            self.assertEqual(name, restored_name)
            self.assertEqual(param.grad is None, other.grad is None, name)
            if param.grad is not None:
                torch.testing.assert_close(other.grad, param.grad, rtol=0, atol=0, msg=name)

    def test_legacy_dict_and_pickled_object_restore_forward_and_gradients(self):
        for object_config in (False, True):
            for use_lrid in (False, True):
                with self.subTest(object_config=object_config, use_lrid=use_lrid):
                    reference = self.save_legacy(
                        self.make_config(use_lrid=use_lrid, lrid_key_from_output_tail=use_lrid),
                        object_config=object_config,
                    )
                    with self.assertWarnsRegex(UserWarning, "legacy checkpoint default"):
                        _, restored, config = utils.load_model_checkpoint(
                            self.path, "cpu", verbose=False, load_training_state=False
                        )
                    self.assertFalse(config.attnres_block_count_prior)
                    self.assert_forward_and_gradients_equal(reference, restored)
                    # Ensure this fixture would detect the original bug.
                    restored.config.attnres_block_count_prior = True
                    with torch.no_grad():
                        difference = (reference(self.tokens) - restored(self.tokens)).abs().max()
                    self.assertGreater(difference.item(), 1e-4)

    def test_resume_uses_migrated_checkpoint_not_current_cli_defaults(self):
        reference = self.save_legacy(self.make_config())
        with self.assertWarns(UserWarning):
            step, _, restored, config = utils.get_model({
                "init_from": "resume", "out_dir": self.temp_dir.name,
                "ckpt_file_name": "", "master_process": False, PRIOR: True,
            }, "cpu")
        self.assertEqual(step, 7)
        self.assertFalse(config.attnres_block_count_prior)
        self.assert_forward_and_gradients_equal(reference, restored)

    def test_explicit_saved_model_flag_wins_and_is_not_mutated(self):
        for value in (False, True):
            for object_config in (False, True):
                with self.subTest(value=value, object_config=object_config):
                    config = self.make_config(**{PRIOR: value})
                    saved_args = config if object_config else asdict(config)
                    before = dict(vars(saved_args)) if object_config else dict(saved_args)
                    with warnings.catch_warnings(record=True) as caught:
                        resolved = model_config_from_checkpoint(saved_args, {PRIOR: not value})
                    self.assertEqual(resolved.attnres_block_count_prior, value)
                    self.assertEqual(asdict(resolved), asdict(config))
                    self.assertEqual(dict(vars(saved_args)) if object_config else saved_args, before)
                    self.assertFalse(caught)

    def test_explicit_saved_training_flag_restores_model_behavior(self):
        for value in (False, True):
            with self.subTest(value=value):
                reference = self.save_legacy(
                    self.make_config(**{PRIOR: value}), training_config={PRIOR: value}
                )
                with self.assertWarnsRegex(UserWarning, "saved training config"):
                    _, restored, config = utils.load_model_checkpoint(self.path, "cpu", verbose=False)
                self.assertEqual(config.attnres_block_count_prior, value)
                self.assert_forward_and_gradients_equal(reference, restored)

    def test_missing_training_config_and_migration_do_not_mutate_model_args(self):
        for training_config in (None, {}, "unavailable"):
            saved_args = asdict(self.make_config())
            saved_args.pop(PRIOR)
            with self.assertWarns(UserWarning):
                config = model_config_from_checkpoint(saved_args, training_config)
            self.assertFalse(config.attnres_block_count_prior)
            self.assertNotIn(PRIOR, saved_args)

    def test_explicit_block_beta_and_other_variant_settings_are_preserved(self):
        config = self.make_config(
            attnres_block_average=True, attnres_block_beta=0.5,
            attnres_block_alpha=0.25,
        )
        saved_args = asdict(config)
        saved_args.pop(PRIOR)
        with self.assertWarns(UserWarning):
            restored = model_config_from_checkpoint(saved_args)
        self.assertEqual(asdict(restored), asdict(config))

    def test_new_model_default_stays_enabled(self):
        self.assertTrue(ModelConfig().attnres_block_count_prior)

    def test_legacy_full_model_outputs_are_unchanged(self):
        reference = self.save_legacy(self.make_config(attnres_type="full"))
        _, restored, _ = utils.load_model_checkpoint(self.path, "cpu", verbose=False)
        self.assert_forward_and_gradients_equal(reference, restored)

    def test_unknown_model_fields_remain_errors(self):
        args = asdict(self.make_config())
        args["unknown_architecture_option"] = True
        with self.assertRaisesRegex(TypeError, "unknown_architecture_option"):
            model_config_from_checkpoint(args)

    def test_invalid_model_args_remain_errors(self):
        for args in (None, [], ModelConfig):
            with self.subTest(args=args), self.assertRaises(TypeError):
                model_config_from_checkpoint(args)

    def test_hf_resave_persists_resolved_setting_and_unchanged_weights(self):
        for value in (False, True):
            with self.subTest(value=value):
                saved_training = {PRIOR: True} if value else {}
                reference = self.save_legacy(
                    self.make_config(**{PRIOR: value}), training_config=saved_training
                )
                args = SimpleNamespace(
                    repo_id="unused", filename=None, revision=None, cache_dir=None,
                    token=None, local_files_only=True, output_name="resaved.pt",
                    out_dir=os.path.join(self.temp_dir.name, "imported"), copy_only=False,
                )
                with mock.patch.object(importer, "resolve_checkpoint_path", return_value=(self.path, "old.pt")):
                    with self.assertWarns(UserWarning), redirect_stdout(io.StringIO()):
                        output_path = importer.load_and_save_checkpoint(args)
                checkpoint, restored, config = utils.load_model_checkpoint(str(output_path), "cpu", verbose=False)
                self.assertEqual(checkpoint["model_args"][PRIOR], value)
                self.assertEqual(config.attnres_block_count_prior, value)
                self.assert_forward_and_gradients_equal(reference, restored)

    def test_analysis_loaders_use_historical_settings(self):
        import analyze_depthwise_routing as routing
        import output_magni_analysis as magnitude

        reference = self.save_legacy(self.make_config())
        for loader, args in (
            (routing.load_model_from_checkpoint, ("unused", self.path, torch.device("cpu"), torch.float32)),
            (magnitude.load_model_from_checkpoint, (self.path, torch.device("cpu"), torch.float32)),
        ):
            with self.subTest(loader=loader.__module__):
                with self.assertWarns(UserWarning), redirect_stdout(io.StringIO()):
                    loaded = loader(*args)
                with torch.no_grad():
                    torch.testing.assert_close(loaded.model(self.tokens), reference(self.tokens), rtol=0, atol=0)
                self.assertFalse(loaded.model.config.attnres_block_count_prior)

    def test_evaluation_provenance_hash_includes_checkpoint_migration(self):
        import run_eval

        original_hash = run_eval._evaluation_source_sha256()
        original_open = open

        def changed_migration(path, *args, **kwargs):
            if os.path.basename(path) == "checkpoint_config.py":
                return io.BytesIO(b"different checkpoint migration semantics")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(run_eval, "open", side_effect=changed_migration, create=True):
            self.assertNotEqual(run_eval._evaluation_source_sha256(), original_hash)


if __name__ == "__main__":
    unittest.main()
