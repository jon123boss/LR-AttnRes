# Residual baseline sources

`residual_gated.py` adapts the Gated Residuals equations and normalization from
Transformers' Qwen4Exp implementation, Copyright 2026 The Qwen Team and
The HuggingFace Inc. team, under Apache-2.0. The adaptation exposes a residual
read/write interface and adds optional FLA grouped RMSNorm.

Source: https://github.com/huggingface/transformers/blob/v5.16.0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py

`residual_ihc.py` adapts Hy4's HyperConnection and HyperHead, Copyright 2026
Tencent HunYuan Team and The HuggingFace Inc. team, under Apache-2.0. The
adaptation exposes the same residual interface and uses the existing backbone.

Source: https://github.com/huggingface/transformers/blob/0f53828e83f2212060233d2ea552cc76617b8712/src/transformers/models/hy_v4/modeling_hy_v4.py

The Apache license is included in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

`residual_mhc.py` independently expresses mHC's equations and DeepSeek-V4's
reference operator conventions. Its optional Sinkhorn path imports the MIT
licensed TileKernels package; no TileKernels code is vendored.

Sources:
- https://arxiv.org/abs/2512.24880
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py
- https://github.com/deepseek-ai/TileKernels/tree/79461d72d6f97f91b89403705a841c9d4c1eddd6
