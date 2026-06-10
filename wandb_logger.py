import os
import time
import pathlib
from typing import Dict, Any, Optional

class WandbLogger:
    def __init__(self, enabled=False, project="", run_name="", config=None, out_dir="out", num_params=None):
        self.enabled = bool(enabled)
        self.project = project or "obpm"
        self.run_name = run_name or ("OBPM-" + str(int(time.time())))
        self.config = config or {}
        self.out_dir = out_dir

        self.run = None
        self.active = False
        self.wandb = None

        if not self.enabled:
            return

        try:
            import wandb
            self.wandb = wandb
        except Exception:
            raise Exception("Wandb unavaliable, not installed or error during import")

        run_id_file = os.path.join(self.out_dir, "wandb_run_id.txt")
        run_id = None
        resume_wandb = self.config.get("init_from") == "resume"
        if resume_wandb and os.path.exists(run_id_file):
            try:
                run_id = open(run_id_file).read().strip()
            except Exception:
                run_id = None

        self.run = self.wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.config,
            id=run_id,
            resume="allow",
            save_code=True,
            reinit="finish_previous",
        )
        if not resume_wandb or run_id is None:
            try:
                pathlib.Path(run_id_file).write_text(self.run.id)
            except Exception:
                pass

        self.wandb.define_metric("tokens_processed")
        self.wandb.define_metric("train/*", step_metric="tokens_processed")
        self.wandb.define_metric("val/*",   step_metric="tokens_processed")
        self.wandb.define_metric("lr",      step_metric="tokens_processed")
        self.wandb.define_metric("grad_norm", step_metric="tokens_processed")
        self.wandb.define_metric("ms_per_step", step_metric="tokens_processed")
        self.wandb.define_metric("tokens_per_s", step_metric="tokens_processed")
        self.wandb.define_metric("gpu/*", step_metric="tokens_processed")
        self.wandb.define_metric("lambdas/*", step_metric="tokens_processed")
        self.wandb.define_metric("attnres_block_scales/*", step_metric="tokens_processed")
        self.wandb.define_metric("model/num_params", step_metric="tokens_processed")

        self.active = True
        if num_params is not None:
            num_params = int(num_params)
            self.run.config.update({"num_params": num_params}, allow_val_change=resume_wandb)
            self.run.summary["num_params"] = num_params
            self.run.log({
                "tokens_processed": 0,
                "model/num_params": num_params,
            })

    def log_train(
        self,
        step,
        iter_loss,
        grad_norm,
        lr,
        ms_per_step,
        tokens_per_s,
        tokens_processed,
        peak_gpu_memory_gb=None,
        peak_gpu_memory_reserved_gb=None,
    ):
        if not self.active:
            return
        gnorm = grad_norm.item() if hasattr(grad_norm, "item") else (float(grad_norm) if grad_norm is not None else 0.0)
        log_dict = {
            "tokens_processed": int(tokens_processed),
            "train/step_loss": float(iter_loss),
            "grad_norm": gnorm,
            "lr": float(lr),
            "ms_per_step": float(ms_per_step),
            "tokens_per_s": float(tokens_per_s),
        }
        if peak_gpu_memory_gb is not None:
            log_dict["gpu/peak_memory_gb"] = float(peak_gpu_memory_gb)
            log_dict["gpu/peak_allocated_gb"] = float(peak_gpu_memory_gb)
        if peak_gpu_memory_reserved_gb is not None:
            log_dict["gpu/peak_reserved_gb"] = float(peak_gpu_memory_reserved_gb)
        self.run.log(log_dict)

    def log_eval(self, step, train_loss, val_loss, lr, tokens_processed):
        if not self.active:
            return
        log_dict = {
            "tokens_processed": int(tokens_processed),
            "train/loss": float(train_loss),
            "val/loss": float(val_loss),
            "lr": float(lr),
        }
        self.run.log(log_dict)

    def log_validation(self, val_loss, tokens_processed, lr=None):
        if not self.active:
            return
        log_dict = {
            "tokens_processed": int(tokens_processed),
            "val/loss": float(val_loss),
        }
        if lr is not None:
            log_dict["lr"] = float(lr)
        self.run.log(log_dict)

    def log_lambda_ratios(self, step, lambda_dict, tokens_processed):
        if not self.active:
            return
        
        log_dict = {
            "tokens_processed": int(tokens_processed)
        }
        for key, value in lambda_dict.items():
            log_dict[f"lambdas/{key}"] = float(value)
        
        self.run.log(log_dict)

    def log_attnres_block_scales(self, scales, block_ends, tokens_processed):
        if not self.active or scales is None:
            return
        if hasattr(scales, "detach"):
            scales = scales.detach().float().cpu().tolist()
        else:
            scales = [float(scale) for scale in scales]
        block_ends = set(int(idx) for idx in (block_ends or ()))
        log_dict = {
            "tokens_processed": int(tokens_processed),
        }
        for source_idx, scale in enumerate(scales, start=1):
            source_kind = "completed" if source_idx in block_ends else "partial"
            log_dict[f"attnres_block_scales/source_{source_idx:03d}_{source_kind}"] = float(scale)
        if scales:
            log_dict["attnres_block_scales/min"] = float(min(scales))
            log_dict["attnres_block_scales/max"] = float(max(scales))
            log_dict["attnres_block_scales/mean"] = float(sum(scales) / len(scales))
        self.run.log(log_dict)

    def log_checkpoint(self, step, ckpt_path, config=None, artifact_name_prefix="obpm-ckpt-step"):
        if not self.active:
            return
        if not os.path.exists(ckpt_path):
            return
        art = self.wandb.Artifact(
            name="%s-%d" % (artifact_name_prefix, step),
            type="model",
            metadata={"step": step, "config": config or self.config},
        )
        art.add_file(ckpt_path)
        self.run.log_artifact(art)

    def finish(self):
        if self.active:
            try:
                self.run.finish()
            finally:
                self.active = False


def get_logger(config, num_params=None):
    logger = WandbLogger(
        enabled=config["wandb_log"],
        project=config["wandb_project"],
        run_name=config["wandb_run_name"],
        config=config,
        out_dir=config["out_dir"],
        num_params=num_params,
    )
    return logger
