import copy
import math

import pytest
import torch

from residual_ihc import IHCConnection, IHCHead


def reference_gates(state, module):
    flat = state.float().reshape(*state.shape[:-2], -1)
    numerator = torch.einsum("...j,ij->...i", flat, module.fn.float())
    return numerator / (flat.square().sum(-1, keepdim=True) / flat.shape[-1] + module.norm_eps).sqrt()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_connection_matches_equations_and_gradients(dtype):
    torch.manual_seed(19)
    connection = IHCConnection(7, num_streams=3)
    reference = copy.deepcopy(connection)
    state = torch.randn(2, 5, 3, 7, dtype=dtype, requires_grad=True)
    branch = torch.randn(2, 5, 7, dtype=dtype, requires_grad=True)
    ref_state = state.detach().clone().requires_grad_()
    ref_branch = branch.detach().clone().requires_grad_()
    branch_input, post = connection.read(state)
    result = connection.write(state, branch, post)

    mixes = reference_gates(ref_state, reference)
    read = torch.sigmoid(mixes[..., :3] * reference.scale[0] + reference.base[:3]) + 1e-6
    write = 2 * torch.sigmoid(mixes[..., 3:] * reference.scale[1] + reference.base[3:]) + 1e-6
    ref_input = torch.stack([read[..., i, None] * ref_state[..., i, :].float() for i in range(3)]).sum(0).to(dtype)
    ref_branch_float = ref_branch.float()
    ref_result = torch.stack([
        ref_state[..., i, :].float() + write[..., i, None] * ref_branch_float for i in range(3)
    ], dim=-2).to(dtype)
    torch.testing.assert_close(branch_input, ref_input)
    torch.testing.assert_close(result, ref_result)
    torch.testing.assert_close(post, write)
    assert post.dtype == torch.float32

    input_probe, output_probe = torch.randn_like(branch_input), torch.randn_like(result)
    (branch_input.mul(input_probe).sum() + result.mul(output_probe).sum()).backward()
    (ref_input.mul(input_probe).sum() + ref_result.mul(output_probe).sum()).backward()
    torch.testing.assert_close(state.grad, ref_state.grad)
    torch.testing.assert_close(branch.grad, ref_branch.grad)
    for actual, expected in zip(connection.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-5)


def test_initial_gates_preserve_identity_bypass():
    connection = IHCConnection(5)
    torch.testing.assert_close(connection.scale, torch.full((2,), .01))
    torch.testing.assert_close(connection.base[:4], torch.full((4,), -math.log(3)))
    torch.testing.assert_close(connection.base[4:], torch.zeros(4))
    assert all(p.dtype == torch.float32 for p in connection.parameters())
    with torch.no_grad():
        connection.fn.zero_()
    state = torch.randn(2, 3, 4, 5)
    branch_input, post = connection.read(state)
    torch.testing.assert_close(branch_input, state.sum(-2) * (.25 + 1e-6))
    torch.testing.assert_close(post, torch.full((2, 3, 4), 1 + 1e-6))
    torch.testing.assert_close(connection.write(state, torch.zeros_like(branch_input), post), state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_learned_head_matches_equations_and_gradients(dtype):
    torch.manual_seed(23)
    head = IHCHead(5)
    reference = copy.deepcopy(head)
    state = torch.randn(2, 3, 4, 5, dtype=dtype, requires_grad=True)
    ref_state = state.detach().clone().requires_grad_()
    actual = head(state)
    read = torch.sigmoid(reference_gates(ref_state, reference) * reference.scale + reference.base) + 1e-6
    expected = torch.stack([
        read[..., i, None] * ref_state[..., i, :].float() for i in range(4)
    ]).sum(0).to(dtype)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    actual.mul(probe).sum().backward()
    expected.mul(probe).sum().backward()
    torch.testing.assert_close(state.grad, ref_state.grad)
    for actual_parameter, expected_parameter in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=3e-6, rtol=3e-5)


def test_fp32_controller_inside_autocast_and_zero_input():
    connection, head = IHCConnection(5), IHCHead(5)
    state = torch.zeros(2, 3, 4, 5, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        branch_input, post = connection.read(state)
        result = head(connection.write(state, branch_input, post))
    assert branch_input.dtype == result.dtype == state.dtype
    assert post.dtype == torch.float32
    result.float().sum().backward()
    assert torch.isfinite(state.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for module in (connection, head) for parameter in module.parameters())
