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

## LRID Attention Residuals

LRID can be enabled as a block Attention Residuals variant:

```bash
python train.py --use_lrid --attnres_type block --lrid_rank 64
```

`--use_lrid` automatically enables `use_attnres`. The default `--lrid_init zero_query`
keeps step-0 depth weights uniform while leaving key projections trainable; `zero_both`
is available for exact spec experiments but is not recommended because it blocks
query/key gradient flow at initialization.
