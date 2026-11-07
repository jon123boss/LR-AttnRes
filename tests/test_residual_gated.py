import builtins
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from residual_gated import GatedResidual, GatedResidualHead


def reference_read(module, state):
    # Compute each branch's norm separately, then evaluate the paper's read.
    normalized = torch.stack(
        [
            (branch.float() / (branch.float().square().mean(-1, keepdim=True) + module.eps).sqrt())
            * (1 + module.norm_weight.reshape(module.num_streams, module.dim)[index].float())
            for index, branch in enumerate(state.unbind(-2))
        ],
        dim=-2,
    ).to(state.dtype)
    features = normalized.flatten(-2)
    scores = F.silu(features @ module.mix_down.weight.T / module.num_streams)
    gates = (scores @ module.mix_up.weight.T).sigmoid().reshape_as(state)
    mixed = sum(gates[..., index, :] * normalized[..., index, :] for index in range(module.num_streams))
    weights = 2 * (features @ module.inject.weight.T / module.num_streams).sigmoid()
    return mixed / module.num_streams, weights


def test_read_and_write_match_paper_including_gradients():
    torch.manual_seed(17)
    module = GatedResidual(5, num_streams=3, rank=2)
    with torch.no_grad():
        module.norm_weight.normal_(std=0.3)
    state = torch.randn(2, 4, 3, 5, requires_grad=True)
    output = torch.randn(2, 4, 5, requires_grad=True)
    original = state.detach().clone()
    mixed, weights = module.read(state)
    expected_mixed, expected_weights = reference_read(module, state)
    actual = module.write(state, output, weights)
    expected = torch.stack(
        [state[..., index, :] + expected_weights[..., index, None] * output for index in range(3)],
        dim=-2,
    )
    torch.testing.assert_close(mixed, expected_mixed)
    torch.testing.assert_close(actual, expected)
    variables = (state, output, *module.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum() + mixed.square().sum(), variables, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum() + expected_mixed.square().sum(), variables)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    torch.testing.assert_close(state, original)


def test_zero_gates_give_half_normalized_read_and_unit_write():
    module = GatedResidual(4, num_streams=2, rank=2)
    for parameter in module.parameters():
        torch.nn.init.zeros_(parameter)
    state = torch.tensor([[[[1., 2., 3., 4.], [2., 0., -2., 0.]]]])
    mixed, weights = module.read(state)
    normalized = F.rms_norm(state, (4,), eps=module.eps)
    torch.testing.assert_close(mixed, normalized.mean(-2) / 2)
    torch.testing.assert_close(weights, torch.ones_like(weights))
    output = torch.randn(1, 1, 4)
    torch.testing.assert_close(module.write(state, output, weights), state + output.unsqueeze(-2))


def test_head_is_read_only_and_keeps_gate_amplitude():
    torch.manual_seed(7)
    module = GatedResidual(8)
    head = GatedResidualHead(8)
    head.load_state_dict({name: value for name, value in module.state_dict().items() if not name.startswith("inject.")})
    state = torch.randn(2, 3, 4, 8)
    torch.testing.assert_close(head(state), module.read(state)[0])
    assert not any(name.startswith("inject.") for name, _ in head.named_parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dtype_backward_and_token_locality(dtype):
    module = GatedResidual(8).to(dtype=dtype)
    state = torch.randn(2, 3, 4, 8, dtype=dtype, requires_grad=True)
    mixed, weights = module.read(state)
    actual = module.write(state, mixed, weights)
    assert actual.dtype == mixed.dtype == weights.dtype == dtype
    torch.testing.assert_close(module.read(state[:, :1])[0], mixed[:, :1])
    actual.float().square().mean().backward()
    assert torch.isfinite(state.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_rejects_incorrect_stream_layout():
    with pytest.raises(ValueError, match="Expected trailing state shape"):
        GatedResidual(8).read(torch.randn(2, 3, 8, 4))


def test_fla_backend_reports_missing_dependency(monkeypatch):
    original_import = builtins.__import__

    def import_without_fla(name, *args, **kwargs):
        if name == "fla.modules.norm.layernorm":
            raise ModuleNotFoundError("fla")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fla)
    with pytest.raises(ImportError, match="requires flash-linear-attention"):
        GatedResidual(8, norm_backend="fla")


def test_fla_backend_rejects_cpu_without_calling_kernel(monkeypatch):
    original_import = builtins.__import__

    def unexpected_kernel(*args, **kwargs):
        pytest.fail("CUDA kernel called for a CPU tensor")

    def import_fla_stub(name, *args, **kwargs):
        if name == "fla.modules.norm.layernorm":
            return SimpleNamespace(group_norm=unexpected_kernel)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_fla_stub)
    module = GatedResidual(8, norm_backend="fla")
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        module.read(torch.randn(2, 3, 4, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA norm requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("module_class", [GatedResidual, GatedResidualHead])
def test_fla_cuda_matches_torch_outputs_and_all_gradients(dtype, module_class):
    pytest.importorskip("fla.modules.norm.layernorm")
    torch.manual_seed(29)
    reference = module_class(64, rank=8).cuda().to(dtype=dtype)
    candidate = module_class(64, rank=8, norm_backend="fla").cuda().to(dtype=dtype)
    reference.norm_weight = torch.nn.Parameter(reference.norm_weight.float())
    candidate.norm_weight = torch.nn.Parameter(candidate.norm_weight.float())
    with torch.no_grad():
        reference.norm_weight.normal_(std=0.1)
    candidate.load_state_dict(reference.state_dict())
    state = torch.randn(2, 5, 4, 64, device="cuda", dtype=dtype, requires_grad=True)
    ref_state = state.detach().clone().requires_grad_()
    tolerance = dict(atol=2e-3, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(candidate._normalize(state), reference._normalize(ref_state), **tolerance)
    if module_class is GatedResidual:
        mixed, weights = candidate.read(state)
        ref_mixed, ref_weights = reference.read(ref_state)
        actual = candidate.write(state, mixed, weights)
        expected = reference.write(ref_state, ref_mixed, ref_weights)
        torch.testing.assert_close(mixed, ref_mixed, **tolerance)
        torch.testing.assert_close(weights, ref_weights, **tolerance)
    else:
        actual, expected = candidate(state), reference(ref_state)
    torch.testing.assert_close(actual, expected, **tolerance)
    probe = torch.randn_like(actual)
    actual.mul(probe).float().sum().backward()
    expected.mul(probe).float().sum().backward()
    torch.testing.assert_close(state.grad, ref_state.grad, **tolerance)
    for actual_parameter, expected_parameter in zip(candidate.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, **tolerance)
