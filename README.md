## Download Repository

```bash
git clone https://github.com/jon123boss/LRID
cd LRID
```

## Prerequisites

Install required dependencies via pip:

```bash
pip install flash-attn --no-build-isolation
pip install tiktoken
pip install huggingface-hub
pip install lm_eval
pip install hf_transfer
pip install wandb  # Optional, for experiment tracking
```

## Data Preparation

Download and preprocess the GPT-2 tokenized FinewebEDU10B dataset:

```bash
python prepdata.py
```
