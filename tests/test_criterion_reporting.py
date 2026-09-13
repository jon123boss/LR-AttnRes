"""Regression tests for the upstream FlashAttention loss-return contract."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
import criterion


def upstream_contract(logits, labels, **kwargs):
    """Faithful public API: first output includes Z; second is logging-only."""
    ce=F.cross_entropy(logits,labels,ignore_index=kwargs['ignore_index'],reduction='none')
    z=kwargs['lse_square_scale']*logits.logsumexp(-1).square()
    z=z.masked_fill(labels==kwargs['ignore_index'],0.)
    return ce+z,z.detach()


class CriterionReportingTests(unittest.TestCase):
    def check_backend(self, backend, device):
        torch.manual_seed(718)
        logits=torch.randn(13,257,device=device,requires_grad=True)
        labels=torch.arange(13,device=device); labels[2]=-100
        for enabled in (False,True):
            for reduction in ('mean','sum','none'):
                for sum_api in (False,True):
                    cfg=criterion.CriterionConfig(z_loss=enabled,z_loss_weight=0.03,
                                                  reduction=reduction,inplace_backward=False)
                    obj=criterion.CrossEntropyLoss(cfg,flash_attention=False)
                    obj.flash_attention=True; obj._flash_ce=backend
                    expected=F.cross_entropy(logits,labels,reduction='none',ignore_index=-100)
                    if enabled:
                        z=0.03*logits.logsumexp(-1).square()
                        expected=expected+z.masked_fill(labels==-100,0.)
                    if sum_api or reduction=='sum': expected=expected.sum()
                    elif reduction=='mean': expected=expected.sum()/(labels!=-100).sum()
                    actual=obj.sum_loss(logits,labels) if sum_api else obj(logits,labels)
                    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
                    grad_expected=torch.autograd.grad(expected.sum(),logits,retain_graph=True)[0]
                    grad_actual=torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0]
                    torch.testing.assert_close(grad_actual,grad_expected,atol=3e-6,rtol=3e-5)

    def test_public_loss_return_contract(self):
        self.check_backend(upstream_contract,'cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_actual_flashattention_values_and_gradients(self):
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
        except ImportError:
            path=os.environ.get('AUDIT_UPSTREAM_FLASH_CE')
            if not path: self.skipTest('FlashAttention is unavailable')
            spec=importlib.util.spec_from_file_location('upstream_flash_ce',path)
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            cross_entropy_loss=module.cross_entropy_loss
        self.check_backend(cross_entropy_loss,'cuda')

if __name__=='__main__': unittest.main()
