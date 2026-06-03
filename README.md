## Download Repository

```bash
git clone https://github.com/jon123boss/LRID
cd LRID
```

## Prerequisites

Install required dependencies via pip. Liger Kernel is required by the model and
training loss path.

```bash
pip install flash-attn --no-build-isolation
pip install tiktoken
pip install huggingface-hub
pip install lm_eval
pip install hf_transfer
pip install liger-kernel
pip install wandb  # Optional, for experiment tracking
```

## Data Preparation

Download and preprocess the GPT-2 tokenized FinewebEDU10B dataset:

```bash
python prepdata.py
```

## Training Kernels

The model uses Liger kernels directly for RMSNorm, RoPE, SwiGLU, CrossEntropy,
and fused linear CrossEntropy.
