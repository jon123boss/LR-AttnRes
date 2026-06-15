# Depthwise Routing Findings

These findings are generated from the validation batches used by `analyze_depthwise_routing.py`.

## Run Summary

- Models analyzed: 20
- Models skipped: 0
- Batches per routed model: [123]
- Tokens per routed model: [1007616]

## Query Sparsity / Utilization

- Standard AttnRes: mean participation fraction 0.3626, Hoyer sparsity 0.1994, dims above 1% of per-site max 0.9758.
- Projected LR-AttnRes: mean participation fraction 0.7971, Hoyer sparsity 0.0622, dims above 1% of per-site max 0.9976.
- Sliced LR-AttnRes: mean participation fraction 0.6420, Hoyer sparsity 0.1124, dims above 1% of per-site max 0.9917.

## Depthwise Attention Pattern

- Standard AttnRes: normalized entropy 0.6705, effective sources 5.9910, top-1 mass 0.5129, generalized JS 0.0430.
- Projected LR-AttnRes: normalized entropy 0.8600, effective sources 12.6667, top-1 mass 0.3186, generalized JS 0.0144.
- Sliced LR-AttnRes: normalized entropy 0.8680, effective sources 14.9294, top-1 mass 0.2887, generalized JS 0.0370.

## Projected vs Sliced Quirks

- Projected LR-AttnRes: pairwise key/value similarity corr 0.7201, key/value norm corr 0.1705, tail energy fraction 0.0353.
- Sliced LR-AttnRes: pairwise key/value similarity corr 0.9254, key/value norm corr 0.9518, tail energy fraction 0.1575.
- Standard AttnRes: pairwise key/value similarity corr 1.0000, key/value norm corr 1.0000, tail energy fraction 1.0000.

## Per-Model Summary

| Model | Group | Val. loss | Query participation | Entropy | JS | Top-1 | Pair sim corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Baseline Transformer | 3.0009 | n/a | n/a | n/a | n/a | n/a |
| Full | Standard AttnRes | 2.9752 | 0.3733 | 0.8036 | 0.0613 | 0.2627 | 1.0000 |
| n=4 | Standard AttnRes | 2.9797 | 0.3538 | 0.5395 | 0.0291 | 0.7408 | 1.0000 |
| n=8 | Standard AttnRes | 2.9778 | 0.3584 | 0.6246 | 0.0413 | 0.5979 | 1.0000 |
| n=16 | Standard AttnRes | 2.9673 | 0.3650 | 0.7142 | 0.0401 | 0.4500 | 1.0000 |
| Full r=16 | Sliced LR-AttnRes | 2.9762 | 0.8536 | 0.9651 | 0.0159 | 0.1337 | 0.8209 |
| Full r=32 | Sliced LR-AttnRes | 2.9682 | 0.8220 | 0.9326 | 0.0302 | 0.1643 | 0.8588 |
| Full r=64 | Sliced LR-AttnRes | 2.9638 | 0.6943 | 0.8971 | 0.0448 | 0.1927 | 0.8889 |
| Full r=128 | Sliced LR-AttnRes | 2.9630 | 0.5778 | 0.8757 | 0.0560 | 0.2086 | 0.9330 |
| Full r=256 | Sliced LR-AttnRes | 2.9626 | 0.4955 | 0.8598 | 0.0607 | 0.2177 | 0.9712 |
| Full r=512 | Sliced LR-AttnRes | 2.9617 | 0.4163 | 0.8392 | 0.0630 | 0.2349 | 0.9951 |
| n=4 r=64 | Sliced LR-AttnRes | 2.9533 | 0.6128 | 0.7822 | 0.0200 | 0.6069 | 0.9559 |
| n=8 r=64 | Sliced LR-AttnRes | 2.9480 | 0.6471 | 0.8105 | 0.0204 | 0.4838 | 0.9535 |
| n=16 r=64 | Sliced LR-AttnRes | 2.9494 | 0.6585 | 0.8500 | 0.0218 | 0.3557 | 0.9511 |
| Full r=16 | Projected LR-AttnRes | 2.9606 | 0.8304 | 0.9676 | 0.0098 | 0.1280 | 0.6751 |
| Full r=32 | Projected LR-AttnRes | 2.9538 | 0.8368 | 0.9317 | 0.0189 | 0.1606 | 0.6686 |
| Full r=64 | Projected LR-AttnRes | 2.9501 | 0.7017 | 0.8890 | 0.0318 | 0.1932 | 0.7794 |
| n=4 r=32 | Projected LR-AttnRes | 2.9488 | 0.8001 | 0.6880 | 0.0087 | 0.6309 | 0.7910 |
| n=8 r=32 | Projected LR-AttnRes | 2.9477 | 0.8129 | 0.7942 | 0.0084 | 0.4853 | 0.7367 |
| n=16 r=32 | Projected LR-AttnRes | 2.9543 | 0.8009 | 0.8892 | 0.0088 | 0.3139 | 0.6697 |
