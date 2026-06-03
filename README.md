## Download Repository

```bash
git clone https://github.com/jon123boss/LR-AttnRes
cd LR-AttnRes
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

## LR AttnRes

LR AttnRes can be enabled as a block Attention Residuals variant:

```bash
python train.py --use_lrid --attnres_type block --lrid_rank 64
```

`--use_lrid` automatically enables `use_attnres`. LR AttnRes uses the same learned,
input-independent depth queries as normal Attention Residuals, but routes over
low-rank input-dependent source keys. Logit scaling defaults to `1 / sqrt(lrid_rank)`;
disable it with `--no-lrid_logit_scale` or set it explicitly with `--lrid_logit_scale`.

See [LR_ATTNRES.md](LR_ATTNRES.md) for the full design note, parameter cost,
stability rationale, and experiment matrix.
