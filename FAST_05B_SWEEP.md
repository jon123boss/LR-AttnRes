# Fast-AttnRes 0.5B block sweep

This workspace runs the sliced low-rank **Block AttnRes** matrix requested for
`n in {4, 8, 16}` and `r in {16, 32, 64, 128, 256, 512, 768, 1024}`.

## Frozen training recipe

- 24 layers, width 1024, 16 heads, FFN width 2816, untied vocabulary head
- GPT-4/cl100k tokenizer, vocabulary 100277, document-boundary masking
- first 131 pinned Ultra-FineWeb training shards and the full validation shard
- 10B training tokens, seed 42, global batch 262144, sequence length 2048
- Muon/AdamW parameters and schedule from public `train.py`
- output-tail low-rank keys, static learned depth queries, one routing head
- raw block sums: no block averaging and no count prior
- neutral LR routing scale (`1.0`)
- `fast-attnres==2.0.1`, selected explicitly with `--attnres_backend fast`
- `torch.compile(fullgraph=True, dynamic=False)` with CUDA graphs disabled

Fast-AttnRes selection is fail closed. Startup must report every multi-source
routed read as a Fast-AttnRes read; a missing package, unsupported semantic
option, or legacy fallback stops the run.

## Scheduling and recovery

The owner reported ranks 1024 and 64 complete for all three block counts. The
public model inventory also confirms n=4/r=32, n=8/r=32, n=16/r=32, and
n=8/r=128 checkpoints; these are recorded and skipped. A separate worker is
covering lower ranks, so this machine runs 768, 512, 256, and the remaining
128 cells first, in that order. Within each rank it runs n=16, n=8, then n=4.
The lower-rank queue is added only after re-auditing W&B to avoid duplicate
training.

`scripts/run_fast_05b_sweep.py` resumes the newest checkpoint automatically,
keeps the newest checkpoint to bound disk usage, evaluates the entire
99,999,744-token validation shard, and records state atomically under
`/root/sweep-runs/sweep_state.json`. Create `/root/sweep-runs/STOP` to stop at
the next job boundary.

The publisher also rewrites `/root/sweep-runs/results.csv` and `results.md`
as a 24-cell ledger containing status, final full-shard validation loss, W&B
URL, and public Hugging Face URL.

W&B metrics are logged online. After full-shard validation,
`scripts/sync_publish_fast_05b.py` records the final loss in the W&B summary
and publishes each final checkpoint, launch manifest, and evaluation report
to a public Hugging Face model repository.
