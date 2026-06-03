import torch
import torch.nn as nn
from contextlib import nullcontext
import os
import tempfile
from model import OBPM, ModelConfig
from dataloader import DataLoaderConfig, create_dataloaders

def get_config(module_globals=None):
    config_keys = [k for k, v in module_globals.items()  if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
    config = {k: module_globals[k] for k in config_keys} 
    return config

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def get_model(config, device):
    start_step = 0
    checkpoint = None
    if config["init_from"] == 'resume':
        ckpt_path = os.path.join(config["out_dir"], config["ckpt_file_name"])
        if not os.path.exists(ckpt_path):
            import glob, re
            step_ckpts = glob.glob(os.path.join(config["out_dir"], 'ckpt_step:*.pt'))
            def extract_step_number(path):
                match = re.search(r'ckpt_step:(\d+)\.pt', path)
                return int(match.group(1)) if match else 0
            step_ckpts.sort(key=extract_step_number)
            ckpt_path = step_ckpts[-1] 
        print(f"Resuming from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        ckpt_model_args = checkpoint["model_args"]
        model_config = ModelConfig(**ckpt_model_args)
        model = OBPM(model_config)
        model_state_dict = checkpoint['model']
        prefix = '_orig_mod.'
        if any(k.startswith(prefix) for k in model_state_dict.keys()):
            print(f"Detected compiled model checkpoint. Removing '{prefix}' prefix from state dict keys.")
            new_state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in model_state_dict.items()}
            model.load_state_dict(new_state_dict, strict=True)
        else:
            model.load_state_dict(model_state_dict, strict=True)
        start_step = checkpoint["step"]
    elif config["init_from"] == 'scratch':
        print("Initializing new model from scratch")
        model_config = ModelConfig(
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_embd=config["n_embd"],
            vocab_size=config["vocab_size"],
            block_size=config["block_size"],
            mlp_hidden_dim=config["mlp_hidden_dim"],
            mlp_ratio=config["mlp_ratio"],
            weight_tying=config["weight_tying"],
            rope_theta=config["rope_theta"],
            rmsnorm_eps=config["rmsnorm_eps"],
            rmsnorm_use_weight=config["rmsnorm_use_weight"],
            rmsnorm_use_bias=config["rmsnorm_use_bias"],
            norm_pos=config["norm_pos"],
            qk_norm=config["qk_norm"],
            clip_qkv=config["clip_qkv"],
            flash_attention=config["flash_attention"],
            init_std=config["init_std"],
            init_cutoff_factor=config["init_cutoff_factor"],
            use_attnres = config["use_attnres"],
            attnres_type = config["attnres_type"],
            attnres_num_blocks = config["attnres_num_blocks"],
            attnres_key_norm = config["attnres_key_norm"],
            attn_res_query_norm = config["attn_res_query_norm"],
            attn_res_query_init = config["attn_res_query_init"],
            use_lrid = config["use_lrid"],
            lrid_rank = config["lrid_rank"],
            lrid_num_heads = config.get("lrid_num_heads", 1),
            lrid_input_dependent_query = config.get("lrid_input_dependent_query", False),
            lrid_use_logit_scale = config["lrid_use_logit_scale"],
            lrid_logit_scale = config["lrid_logit_scale"],
            )
        model = OBPM(model_config)
    else:
        raise Exception("Init_from has to be either 'scratch' or 'resume'")
        
    model.to(device)
    
    return start_step, checkpoint, model, model_config


def get_dataloader(config):
    dataloader_config = DataLoaderConfig(
        data_dir=config["dataset_dir"],
        batch_size=config["batch_size"],
        block_size=config["block_size"],
        grad_accum_steps=config["grad_accum_steps"],
        use_doc_masking=config["use_doc_masking"],
        doc_separator_token=config["doc_separator_token"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"],
    )
    return create_dataloaders(dataloader_config)
