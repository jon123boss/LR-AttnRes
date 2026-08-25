"""Two-rank regression for validation shards with unequal batch counts.

Run with:
    python tests/ddp_validation_uneven.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from criterion import get_criterion
from utils import compute_validation_loss

if sys.platform == "darwin":
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")


class BufferedLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.projection = torch.nn.Linear(4, 8)
        # DDP synchronizes registered buffers before wrapped forwards. Unequal
        # forward counts therefore reproduced the old collective mismatch.
        self.register_buffer("forward_marker", torch.ones(()))

    def forward(self, tokens, cu_doc_len=None, max_doc_len=None):
        return self.projection(self.embedding(tokens)) + self.forward_marker * 0


def worker(rank, world_size, rendezvous_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(123)
    model = DDP(BufferedLanguageModel())
    criterion = get_criterion(
        {
            "ignore_index": -100,
            "reduction": "mean",
            "z_loss": False,
            "z_loss_weight": 0.0,
            "ce_inplace_backward": False,
            "lm_head_chunk_size": 0,
            "flash_attention": False,
        }
    )
    batch = (
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[2, 3, 4]], dtype=torch.long),
    )
    local_loader = [batch, batch] if rank == 0 else [batch]
    metrics = compute_validation_loss(
        model,
        criterion,
        local_loader,
        torch.device("cpu"),
        vocab_size=8,
        use_doc_masking=False,
        distributed=True,
    )
    assert metrics["tokens"] == 9, metrics
    assert metrics["batches"] == 3, metrics
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, metrics)
    assert gathered[0] == gathered[1], gathered
    if rank == 0:
        print("uneven DDP validation passed", metrics)
    dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous_path = os.path.join(temp_dir, "ddp-rendezvous")
        mp.spawn(worker, args=(2, rendezvous_path), nprocs=2, join=True)


if __name__ == "__main__":
    main()
