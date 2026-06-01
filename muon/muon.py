# Taken from https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
# Computed for num_iters=5, safety_factor=1e-2, cushion=0.02
# -----------------------------------------------------------------------------

polar_express_coeffs = [
(8.156554524902461, -22.48329292557795, 15.878769915207462),
(4.042929935166739, -2.808917465908714, 0.5000178451051316),
(3.8916678022926607, -2.772484153217685, 0.5060648178503393),
(3.285753657755655, -2.3681294933425376, 0.46449024233003106),
(2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

_TRITON_AVAILABLE = False
try:
    import triton # type: ignore
    import triton.language as tl # type: ignore
    _TRITON_AVAILABLE = True
except ImportError:
    print("Not using polar express with triton")
    pass


def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


if _TRITON_AVAILABLE:
    @triton.jit
    def _pid_to_block(
        pid,
        M,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
        batch_idx = pid // (num_pid_m * num_pid_n)
        pid = pid % (num_pid_m * num_pid_n)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
        m_idx = pid_m * BLOCK_SIZE_M
        n_idx = pid_n * BLOCK_SIZE_N
        return batch_idx, m_idx, n_idx

    @triton.autotune(
        configs=_get_autotune_configs(),
        key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
    )
    @triton.jit
    def XXT_kernel(
        A_ptr, C_ptr,
        M, K,
        a_stride_b, a_stride_r, a_stride_c,
        c_stride_b, c_stride_r, c_stride_c,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        LOWER_UPPER: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_idx, m_idx, n_idx = _pid_to_block(
            pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
        )

        skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
        skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
        if skip_block_below_diag or skip_block_above_diag:
            return

        A_ptr += batch_idx * a_stride_b
        C_ptr += batch_idx * c_stride_b

        offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
        at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, at, accumulator)
            a_ptrs += BLOCK_SIZE_K * a_stride_c
            at_ptrs += BLOCK_SIZE_K * a_stride_c

        out_dtype = C_ptr.dtype.element_ty
        output = accumulator.to(out_dtype)

        offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        tl.store(c_ptrs, output, mask=c_mask)

        c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
        c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(c_ptrs_t, output.T, mask=c_mask_t)

    @triton.autotune(
        configs=_get_autotune_configs(),
        key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
    )
    @triton.jit
    def ba_plus_cAA_kernel(
        A_ptr, C_ptr,
        M,
        a_stride_b, a_stride_r, a_stride_c,
        c_stride_b, c_stride_r, c_stride_c,
        alpha, beta,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        LOWER_UPPER: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_idx, m_idx, n_idx = _pid_to_block(
            pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
        )

        skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
        skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
        if skip_block_below_diag or skip_block_above_diag:
            return

        A_ptr += batch_idx * a_stride_b
        C_ptr += batch_idx * c_stride_b

        offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
        at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
            at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, at, accumulator)
            a_ptrs += BLOCK_SIZE_K * a_stride_c
            at_ptrs += BLOCK_SIZE_K * a_stride_c

        offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
        a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
        a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
        a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

        accumulator *= alpha
        accumulator += a_add * beta

        out_dtype = C_ptr.dtype.element_ty
        output = accumulator.to(out_dtype)

        offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        tl.store(c_ptrs, output, mask=c_mask)

        c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
        c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(c_ptrs_t, output.T, mask=c_mask_t)

    def _XXT(A: torch.Tensor, out: torch.Tensor):
        """Launch Triton kernel to compute C = A @ A.T"""
        assert A.ndim == 2 or A.ndim == 3
        M, K = A.shape[-2:]
        assert out.size(-2) == M and out.size(-1) == M

        batch_size = A.size(0) if A.ndim == 3 else 1
        input_batch_stride = A.stride(0) if A.ndim == 3 else 0
        output_batch_stride = out.stride(0) if out.ndim == 3 else 0

        grid = lambda meta: (
            batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
        )
        XXT_kernel[grid](
            A_ptr=A,
            C_ptr=out,
            M=M,
            K=K,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2),
            a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2),
            c_stride_c=out.stride(-1),
        )
        return out

    def _ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
        """Launch Triton kernel to compute C = alpha * A @ A.T + beta * A"""
        assert A.ndim == 2 or A.ndim == 3
        M, K = A.shape[-2:]
        assert M == K
        assert out.size(-2) == M and out.size(-1) == M

        batch_size = A.size(0) if A.ndim == 3 else 1
        input_batch_stride = A.stride(0) if A.ndim == 3 else 0
        output_batch_stride = out.stride(0) if out.ndim == 3 else 0

        grid = lambda meta: (
            batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
        )
        ba_plus_cAA_kernel[grid](
            A_ptr=A,
            C_ptr=out,
            M=M,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2),
            a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2),
            c_stride_c=out.stride(-1),
            alpha=alpha,
            beta=beta,
        )
        return out

@torch.compile(dynamic=False, fullgraph=True)
def polar_express_triton(G: torch.Tensor) -> torch.Tensor:
    """
    Polar Express Sign Method using Triton kernels.
    Reference: https://arxiv.org/pdf/2505.16932
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Use polar_express_pytorch instead.")

    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True


    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        _XXT(X, out=A)                          # A = X @ X.mT
        _ba_plus_cAA(A, alpha=c, beta=b, out=B) # B = b * A + c * A @ A
        aX_plus_BX(X, B, X, beta=a, out=C)      # C = a * X + B @ X
        X, C = C, X                             # Swap references

    if transposed:
        X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def polar_express_pytorch(G: torch.Tensor) -> torch.Tensor:
    """
    Polar Express Sign Method using pure PyTorch (no Triton).
    Reference: https://arxiv.org/pdf/2505.16932
    """
    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        X_f32 = X.float()
        A = X_f32 @ X_f32.mT              # A = X @ X.T
        B = b * A + c * (A @ A)           # B = b * A + c * A @ A
        X = (a * X_f32 + B @ X_f32).to(torch.bfloat16)  # X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


def polar_express(G: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    if use_triton and _TRITON_AVAILABLE:
        return polar_express_triton(G)
    else:
        return polar_express_pytorch(G)


# -----------------------------------------------------------------------------
# Muon optimizer with Polar Express and second-moment adaptation
# -----------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        beta2: float | None = 0.95,
        nesterov: bool = True,
        use_triton: bool = True,
        cautious: bool = True,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=beta2,
            nesterov=nesterov,
            use_triton=use_triton,
            cautious=cautious,
        )
        super().__init__(params, defaults)

    def reset(self):
        """Reset all momentum buffers to zero."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "momentum_buffer" in state:
                    state["momentum_buffer"].zero_()
                if "variance_buffer" in state:
                    state["variance_buffer"].zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            nesterov = group["nesterov"]
            use_triton = group["use_triton"]
            cautious = group["cautious"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    if beta2 is not None:
                        if p.size(-2) >= p.size(-1):
                            state["variance_buffer"] = torch.zeros(
                                *p.shape[:-1], 1, device=p.device, dtype=p.dtype
                            )
                        else:
                            state["variance_buffer"] = torch.zeros(
                                *p.shape[:-2], 1, p.shape[-1], device=p.device, dtype=p.dtype
                            )

                momentum_buffer = state["momentum_buffer"]

                orig_shape = grad.shape
                if grad.ndim == 4:
                    grad = grad.view(grad.size(0), -1)
                    momentum_buffer_view = momentum_buffer.view(grad.shape)
                else:
                    momentum_buffer_view = momentum_buffer

                momentum_buffer_view.lerp_(grad, 1 - momentum)
                if nesterov:
                    update = grad.lerp(momentum_buffer_view, momentum)
                else:
                    update = momentum_buffer_view.clone()

                update = polar_express(update, use_triton=use_triton)
                
                if beta2 is not None:
                    variance_buffer = state["variance_buffer"]
                    
                    v_norm = update.norm(dim=(-2, -1), keepdim=True)
                    
                    if p.size(-2) >= p.size(-1):
                        v_mean = update.square().mean(dim=-1, keepdim=True)
                    else:
                        v_mean = update.square().mean(dim=-2, keepdim=True)
                    
                    variance_buffer.lerp_(v_mean.to(dtype=p.dtype), 1 - beta2)
                    
                    step_size = variance_buffer.clamp(min=1e-10).rsqrt()
                    update = update * step_size
                    
                    v_norm_new = update.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
                    update = update * (v_norm / v_norm_new)

                update = update * max(1, p.size(-2) / p.size(-1)) ** 0.5

                update = update.view(orig_shape)

                if weight_decay > 0:
                    if cautious:
                        mask = (update * p >= 0).to(dtype=p.dtype)
                        p.addcmul_(mask, p, value=-lr * weight_decay)
                    else:
                        p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

        return loss