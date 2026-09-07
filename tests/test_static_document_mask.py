import pytest
import torch

from dataloader import collate_with_doc_masking


def _sample(tokens, boundaries):
    x = torch.tensor(tokens, dtype=torch.int64)
    y = torch.roll(x, shifts=-1)
    cu = torch.tensor(boundaries, dtype=torch.int32)
    maximum = max(b - a for a, b in zip(boundaries, boundaries[1:]))
    return x, y, cu, maximum


def test_static_document_mask_pads_with_zero_length_sequences():
    batch = [
        _sample([1, 2, 3, 4], [0, 2, 4]),
        _sample([5, 6, 7, 8], [0, 1, 4]),
    ]

    x, y, cu, maximum = collate_with_doc_masking(
        batch,
        cu_seqlens_size=9,
        static_max_seqlen=4,
    )

    assert x.shape == y.shape == (2, 4)
    assert cu.tolist() == [0, 2, 4, 5, 8, 8, 8, 8, 8]
    assert maximum == 4


def test_static_document_mask_fails_before_truncating_boundaries():
    batch = [
        _sample([1, 2, 3, 4], [0, 2, 4]),
        _sample([5, 6, 7, 8], [0, 1, 4]),
    ]

    with pytest.raises(ValueError, match="capacity is too small"):
        collate_with_doc_masking(
            batch,
            cu_seqlens_size=4,
            static_max_seqlen=4,
        )
