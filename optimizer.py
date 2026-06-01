# optimizer.py
import torch
from dataclasses import dataclass
from typing import List
from muon.muon import Muon


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


def configure_optimizers(model, config: OptimizerConfig):
    muon_params = []
    adamw_params = []
    
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        for layer_idx, block in enumerate(model.transformer.layers):
            for name, p in block.named_parameters():
                if p.ndim >= 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

    if hasattr(model.transformer, "wte"):
        for p in model.transformer.wte.parameters():
            adamw_params.append(p)
    
    if hasattr(model, "lm_head") and model.lm_head is not None:
        for p in model.lm_head.parameters():
            adamw_params.append(p)
    
    if hasattr(model.transformer, "final_norm"):
        for p in model.transformer.final_norm.parameters():
            if p.ndim < 2: 
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
    )

    optimizers = configure_optimizers(model, optimizer_config)
    
    return optimizers
