# dataloader.py
import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader


@dataclass
class DataLoaderConfig:
    data_dir: str = "finewebedu10B"
    batch_size: int = 4
    block_size: int = 1024
    grad_accum_steps: int = 1
    use_doc_masking: bool = True
    doc_separator_token: Optional[int] = 50256
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True 
    prefetch_factor: int = 2
    dtype: np.dtype = np.uint16


class DocumentPackingDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        block_size: int,
        use_doc_masking: bool,
        doc_separator_token: Optional[int],
        dtype=np.uint16,
    ):
        super().__init__()

        self.split = split
        self.block_size = block_size
        self.use_doc_masking = use_doc_masking
        self.doc_separator_token = doc_separator_token
        self.dtype = dtype

        pattern = os.path.join(
            data_dir,
            f"finewebedu_{split}_*.bin" if split in ("train", "val") else None
        )
        if pattern is None:
            raise ValueError(f"Unknown split: {split!r}")

        shard_paths = sorted(glob.glob(pattern))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shards found for split={split!r} in {data_dir!r} (pattern: {pattern})"
            )

        self.shards = []
        self.shard_sizes = []
        self.shard_seq_counts = []
        
        self._shard_paths = []
        self._boundary_cache = {}

        for p in shard_paths:
            mm = np.memmap(p, mode="r", dtype=dtype)
            n_tokens = mm.shape[0]
            n_seq = max(0, (n_tokens - 1) // block_size)
            
            if n_seq == 0:
                continue

            self.shards.append(mm)
            self.shard_sizes.append(n_tokens)
            self.shard_seq_counts.append(n_seq)
            self._shard_paths.append(p)

        if not self.shards:
            raise RuntimeError(
                f"All shards for split={split!r} were too small for block_size={block_size}"
            )

        self.shard_seq_offsets = np.cumsum([0] + self.shard_seq_counts).astype(np.int64)
        self._num_sequences = int(self.shard_seq_offsets[-1])
        self.total_tokens = sum(self.shard_sizes)

        self._shard_seq_offsets_searchable = self.shard_seq_offsets[1:]

        print(
            f"Dataset split: {split} | "
            f"shards: {len(self.shards)} | "
            f"total_tokens: {self.total_tokens:,} | "
            f"sequences: {self._num_sequences:,} | "
            f"doc_masking: {use_doc_masking}"
        )

    def _get_boundaries(self, shard_idx: int) -> Optional[np.ndarray]:
        if not self.use_doc_masking or self.doc_separator_token is None:
            return None

        if shard_idx in self._boundary_cache:
            return self._boundary_cache[shard_idx]

        tokens = self.shards[shard_idx]
        boundaries = self._find_doc_boundaries_fast(tokens, self.doc_separator_token)
        self._boundary_cache[shard_idx] = boundaries
        return boundaries

    def _find_doc_boundaries_fast(
        self, tokens: np.memmap, separator_token: int
    ) -> np.ndarray:
        chunk_size = 50_000_000
        n_tokens = len(tokens)
        
        estimated_separators = n_tokens // 500
        boundaries = np.empty(estimated_separators + 2, dtype=np.int64)
        boundaries[0] = 0
        write_idx = 1

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            chunk = tokens[start:end]
            
            sep_positions = np.flatnonzero(chunk == separator_token)
            
            if len(sep_positions) > 0:
                needed = write_idx + len(sep_positions)
                if needed >= len(boundaries):
                    boundaries = np.resize(boundaries, max(needed * 2, len(boundaries) * 2))
                
                boundaries[write_idx:write_idx + len(sep_positions)] = sep_positions + start
                write_idx += len(sep_positions)

        if write_idx >= len(boundaries):
            boundaries = np.resize(boundaries, write_idx + 1)
        boundaries[write_idx] = n_tokens
        
        return boundaries[:write_idx + 1]

    def __len__(self) -> int:
        return self._num_sequences

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard_idx = int(np.searchsorted(self._shard_seq_offsets_searchable, idx, side="right"))
        local_idx = idx - int(self.shard_seq_offsets[shard_idx])
        return shard_idx, local_idx

    def _get_doc_info_fast(
        self, shard_idx: int, start: int, end: int
    ) -> Tuple[torch.Tensor, int]:
        if not self.use_doc_masking:
            return torch.tensor([0, self.block_size], dtype=torch.int32), self.block_size

        boundaries = self._get_boundaries(shard_idx)
        if boundaries is None:
            return torch.tensor([0, self.block_size], dtype=torch.int32), self.block_size

        left_idx = np.searchsorted(boundaries, start, side="left")
        right_idx = np.searchsorted(boundaries, end, side="right")
        
        relevant = boundaries[left_idx:right_idx]
        
        if len(relevant) == 0:
            cu_doc_len = torch.tensor([0, self.block_size], dtype=torch.int32)
            return cu_doc_len, self.block_size

        relative = relevant - start
        
        doc_positions = np.empty(len(relative) + 2, dtype=np.int32)
        doc_positions[0] = 0
        
        write_pos = 1
        for boundary in relative:
            if 0 < boundary < self.block_size:
                doc_positions[write_pos] = boundary
                write_pos += 1
        
        if doc_positions[write_pos - 1] != self.block_size:
            doc_positions[write_pos] = self.block_size
            write_pos += 1

        cu_doc_len = torch.from_numpy(doc_positions[:write_pos].copy())
        
        doc_lengths = cu_doc_len[1:] - cu_doc_len[:-1]
        max_doc_len = int(doc_lengths.max().item())

        return cu_doc_len, max_doc_len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._num_sequences:
            raise IndexError(idx)

        shard_idx, local_seq_idx = self._locate(idx)
        tokens = self.shards[shard_idx]

        start = local_seq_idx * self.block_size
        end = start + self.block_size + 1
        
        seq = np.asarray(tokens[start:end], dtype=np.int64)
        
        x = torch.from_numpy(seq[:-1].copy())
        y = torch.from_numpy(seq[1:].copy())

        cu_doc_len, max_doc_len = self._get_doc_info_fast(shard_idx, start, end - 1)

        return x, y, cu_doc_len, max_doc_len


def collate_with_doc_masking(batch):
    batch_size = len(batch)
    seq_len = batch[0][0].size(0)
    
    x_batch = torch.empty(batch_size, seq_len, dtype=torch.int64)
    y_batch = torch.empty(batch_size, seq_len, dtype=torch.int64)
    
    total_cu_len = sum(len(item[2]) for item in batch) - (batch_size - 1)
    cu_doc_len_batch = torch.empty(total_cu_len, dtype=torch.int32)
    
    max_doc_len_batch = 0
    offset = 0
    cu_write_idx = 0
    
    for i, (x, y, cu_doc_len, max_doc_len) in enumerate(batch):
        x_batch[i] = x
        y_batch[i] = y
        
        if max_doc_len > max_doc_len_batch:
            max_doc_len_batch = max_doc_len
        
        if i == 0:
            adjusted = cu_doc_len + offset
            n = len(adjusted)
            cu_doc_len_batch[cu_write_idx:cu_write_idx + n] = adjusted
            cu_write_idx += n
        else:
            adjusted = cu_doc_len[1:] + offset
            n = len(adjusted)
            cu_doc_len_batch[cu_write_idx:cu_write_idx + n] = adjusted
            cu_write_idx += n
        
        offset += seq_len

    return x_batch, y_batch, cu_doc_len_batch[:cu_write_idx], max_doc_len_batch


def collate_simple(batch):
    xs, ys, _, _ = zip(*batch)
    return torch.stack(xs), torch.stack(ys), None, None


def create_dataloaders(config: DataLoaderConfig):
    train_dataset = DocumentPackingDataset(
        data_dir=config.data_dir,
        split="train",
        block_size=config.block_size,
        use_doc_masking=config.use_doc_masking,
        doc_separator_token=config.doc_separator_token,
        dtype=config.dtype,
    )

    val_dataset = DocumentPackingDataset(
        data_dir=config.data_dir,
        split="val",
        block_size=config.block_size,
        use_doc_masking=config.use_doc_masking,
        doc_separator_token=config.doc_separator_token,
        dtype=config.dtype,
    )

    if config.use_doc_masking:
        collate_fn = collate_with_doc_masking
    else:
        collate_fn = collate_simple

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(config.persistent_workers and config.num_workers > 0),
        collate_fn=collate_fn,
    )

    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = TorchDataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = TorchDataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    iters_per_epoch = len(train_dataset) // (config.batch_size * config.grad_accum_steps)

    print(
        f"Dataloader: 1 epoch ≈ {iters_per_epoch} iterations | "
        f"Train sequences={len(train_dataset):,} | "
        f"Batch_size={config.batch_size} | "
        f"Grad_accum_steps={config.grad_accum_steps} | "
        f"Workers={config.num_workers} | "
        f"Prefetch={config.prefetch_factor if config.num_workers > 0 else 'N/A'}"
    )

    return train_loader, val_loader


def warmup_boundaries(dataset: DocumentPackingDataset, num_shards: Optional[int] = None):
    from concurrent.futures import ThreadPoolExecutor
    
    if not dataset.use_doc_masking:
        return
    
    n = num_shards or len(dataset.shards)
    n = min(n, len(dataset.shards))
    
    def compute_boundary(shard_idx):
        dataset._get_boundaries(shard_idx)
        return shard_idx
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(compute_boundary, range(n)))
    
    print(f"Warmed up boundaries for {n} shards")
