# optimizer.py
import torch
from dataclasses import dataclass
from typing import List
from muon.muon import Muon


def unwrap_model(model):
    while True:
        if hasattr(model, "module"):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


@dataclass
class OptimizerConfig:
    muon_lr: float = 0.03
    adamw_lr: float = 0.008
    muon_weight_decay: float = 0.0
    adamw_weight_decay: float = 0.0
    cautious: bool = True
    beta1: float = 0.9
    beta2: float = 0.95
    muon_momentum: float = 0.95
    verbose: bool = True


def configure_optimizers(model, config: OptimizerConfig):
    muon_params = []
    adamw_params = []
    core_model = unwrap_model(model)
    
    if hasattr(core_model, "transformer"):
        for name, p in core_model.transformer.named_parameters():
            if name.startswith("wte."):
                continue
            if name.startswith("lrid_queries."):
                adamw_params.append(p)
                continue
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)

    if hasattr(core_model.transformer, "wte"):
        for p in core_model.transformer.wte.parameters():
            adamw_params.append(p)
    
    if hasattr(core_model, "lm_head") and core_model.lm_head is not None:
        for p in core_model.lm_head.parameters():
            adamw_params.append(p)
    
    optimizers = []

    if muon_params:
        muon = Muon(
            muon_params,
            lr=config.muon_lr,
            weight_decay=config.muon_weight_decay,
            momentum=config.muon_momentum,
            cautious=config.cautious,
        )
        optimizers.append(muon)
    
    use_cuda = torch.cuda.is_available()
    
    if adamw_params:
        adamw = torch.optim.AdamW(
            adamw_params,
            lr=config.adamw_lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.adamw_weight_decay,
            fused=use_cuda,
            capturable=use_cuda,
        )
        optimizers.append(adamw)
    
    if config.verbose:
        print(f"Muon optimizer: {len(muon_params)} parameters")
        print(f"AdamW optimizer: {len(adamw_params)} parameters")
    
    return optimizers


def get_optimizers(config, model):
    optimizer_config = OptimizerConfig(
        muon_lr=config["muon_lr"],
        adamw_lr=config["adamw_lr"],
        muon_weight_decay=config["muon_weight_decay"],
        adamw_weight_decay=config["adamw_weight_decay"],
        cautious=config["cautious"],
        beta1=config["beta1"],
        beta2=config["beta2"],
        muon_momentum=config["muon_momentum"],
        verbose=bool(config.get("master_process", True)),
    )

    optimizers = configure_optimizers(model, optimizer_config)
    
    return optimizers
