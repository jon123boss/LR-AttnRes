import pytest
import torch

from residual_mhc import MHCConnection, MHCHead


def reference_read(module, state):
    """Scalar-indexed stream equations, independent of production matmul layout."""
    n = module.num_streams
    flat = state.float().flatten(-2)
    mixes = torch.stack([(flat * row).sum(-1) for row in module.fn], -1)
    mixes = mixes / torch.sqrt(flat.square().mean(-1, keepdim=True) + module.norm_eps)
    pre = (mixes[..., :n] * module.scale[0] + module.base[:n]).sigmoid() + module.gate_eps
    post = 2 * (mixes[..., n:2*n] * module.scale[1] + module.base[n:2*n]).sigmoid()
    logits = (mixes[..., 2*n:] * module.scale[2] + module.base[2*n:]).reshape(*state.shape[:-2], n, n)
    positive = (logits - logits.amax(-1, keepdim=True)).exp()
    weights = positive / positive.sum(-1, keepdim=True) + module.gate_eps
    for iteration in range(module.sinkhorn_iters):
        if iteration:
            weights = weights / (weights.sum(-1, keepdim=True) + module.gate_eps)
        weights = weights / (weights.sum(-2, keepdim=True) + module.gate_eps)
    collapsed = sum(pre[..., i, None] * state[..., i, :].float() for i in range(n))
    return collapsed.to(state.dtype), (post, weights)


def reference_write(state, branch, context):
    post, weights = context
    return torch.stack([
        sum(weights[..., i, j, None] * state[..., i, :].float()
            for i in range(state.shape[-2])) + post[..., j, None] * branch.float()
        for j in range(state.shape[-2])
    ], -2).to(state.dtype)


def test_read_write_and_gradients_match_independent_equations():
    torch.manual_seed(41)
    module = MHCConnection(5, num_streams=3, sinkhorn_iters=7, scale_init=0.7, init_std=0.3)
    state = torch.randn(2, 4, 3, 5, requires_grad=True)
    branch_weight = torch.randn(5, 5, requires_grad=True)
    collapsed, context = module.read(state)
    expected_input, expected_context = reference_read(module, state)
    torch.testing.assert_close(collapsed, expected_input)
    for actual, expected in zip(context, expected_context):
        torch.testing.assert_close(actual, expected)
    actual = module.write(state, collapsed @ branch_weight, context)
    expected = reference_write(state, expected_input @ branch_weight, expected_context)
    torch.testing.assert_close(actual, expected)
    inputs = (state, branch_weight, *module.parameters())
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_merge_routes_input_axis_to_output_axis():
    module = MHCConnection(1, num_streams=3)
    state = torch.tensor([[[[1.], [10.], [100.]]]])
    # A nonsymmetric cyclic permutation detects an accidentally omitted transpose.
    comb = torch.tensor([[[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]]])
    result = module.write(state, torch.zeros(1, 1, 1), (torch.zeros(1, 1, 3), comb))
    torch.testing.assert_close(result.flatten(), torch.tensor([100., 1., 10.]))


def test_sinkhorn_is_positive_and_approximately_doubly_stochastic():
    torch.manual_seed(12)
    module = MHCConnection(5, num_streams=4, scale_init=1., init_std=0.3)
    _, (_, comb) = module.read(torch.randn(2, 3, 4, 5))
    assert (comb > 0).all()
    torch.testing.assert_close(comb.sum(-1), torch.ones(2, 3, 4), atol=3e-6, rtol=0)
    torch.testing.assert_close(comb.sum(-2), torch.ones(2, 3, 4), atol=2e-6, rtol=0)


def test_finite_sinkhorn_does_not_claim_exact_row_sums():
    module = MHCConnection(1, num_streams=3, sinkhorn_iters=1, init_std=0.)
    with torch.no_grad():
        module.base[6:].copy_(torch.tensor([[4., 0., -1.], [0., 0., 1.], [0., 0., 0.]]).flatten())
    _, (_, comb) = module.read(torch.zeros(1, 1, 3, 1))
    assert (comb.sum(-1) - 1).abs().max() > 0.1
    torch.testing.assert_close(comb.sum(-2), torch.ones(1, 1, 3), atol=2e-6, rtol=0)


def test_zero_input_initial_gates_and_head_are_not_hardcoded_stream_average():
    module = MHCConnection(3)
    assert torch.equal(module.scale.detach(), torch.full((3,), .01))
    assert torch.count_nonzero(module.base) == 0
    collapsed, (post, _) = module.read(torch.zeros(2, 1, 4, 3))
    assert torch.count_nonzero(collapsed) == 0
    torch.testing.assert_close(post, torch.ones_like(post))
    head = MHCHead(3, init_std=0.)
    state = torch.arange(24.).reshape(2, 1, 4, 3)
    torch.testing.assert_close(head(state), (.5 + head.gate_eps) * state.sum(-2))
    with torch.no_grad():
        head.base.copy_(torch.tensor([-2., -1., 1., 2.]))
    expected = ((head.base.sigmoid() + head.gate_eps)[:, None] * state).sum(-2)
    torch.testing.assert_close(head(state), expected)


def test_bfloat16_autocast_keeps_controller_and_gradient_precision():
    torch.manual_seed(3)
    module, head = MHCConnection(5), MHCHead(5)
    state = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        branch, context = module.read(state)
        merged = module.write(state, branch.sin(), context)
        output = head(merged)
    assert branch.dtype == merged.dtype == output.dtype == torch.bfloat16
    assert all(part.dtype == torch.float32 for part in context)
    output.float().square().sum().backward()
    for parameter in (*module.parameters(), *head.parameters()):
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.isfinite(state.grad).all()


@pytest.mark.parametrize("kwargs", [{"sinkhorn_iters": 0}, {"num_streams": 0},
                                    {"norm_eps": 0}, {"gate_eps": float("nan")}])
def test_invalid_controls_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        MHCConnection(5, **kwargs)


def test_explicit_tile_backend_rejects_cpu_without_silent_fallback():
    module = MHCConnection(5, sinkhorn_backend="tile")
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        module.read(torch.randn(1, 2, 4, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM90/SM100 CUDA")
def test_tile_sinkhorn_matches_reference_forward_and_backward():
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("requires SM90/SM100")
    pytest.importorskip("tile_kernels.modeling.mhc.ops.sinkhorn")
    torch.manual_seed(25)
    reference = MHCConnection(16, scale_init=0.7, init_std=0.1).cuda()
    fused = MHCConnection(16, scale_init=0.7, init_std=0.1, sinkhorn_backend="tile").cuda()
    fused.load_state_dict(reference.state_dict())
    state = torch.randn(2, 17, 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other_state = state.detach().clone().requires_grad_()
    expected_input, expected_context = reference.read(state)
    actual_input, actual_context = fused.read(other_state)
    torch.testing.assert_close(actual_input, expected_input, atol=0, rtol=0)
    for actual, expected in zip(actual_context, expected_context):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    # Include direct cotangents on comb, so its backward is tested independently
    # of the large direct-residual gradient in the final merge.
    cotangent = torch.randn_like(expected_context[1]).transpose(-1, -2)
    expected_grad = torch.autograd.grad(expected_context[1], (state, *reference.parameters()), grad_outputs=cotangent)
    actual_grad = torch.autograd.grad(actual_context[1], (other_state, *fused.parameters()), grad_outputs=cotangent)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-3)
