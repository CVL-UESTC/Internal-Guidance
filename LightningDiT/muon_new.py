import math
import torch
from typing import Iterable, Optional
import torch.distributed as dist

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)

    # working view (2D)
    X = G.bfloat16()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    if X.numel() == 0:
        return torch.zeros_like(G)

    # normalize spectral norm (use a stable norm)
    try:
        norm = X.norm(dim=(-2, -1), keepdim=True)
    except Exception:
        norm = torch.tensor(0.0, device=X.device, dtype=X.dtype)
    X = X / (norm + 1e-7)

    # try fast iteration
    try:
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        result = X
    except Exception:
        # fallback: operate in float/double and/or SVD
        Xf = G.float()
        if Xf.size(-2) != Xf.size(-1):
            # non-square -> nearest orthogonal via SVD (U @ Vt)
            U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
            result = (U @ Vh).to(G.dtype)
        else:
            # square: safer NS in double precision with scaling
            A = Xf.double()
            normA = torch.norm(A)
            if normA > 0:
                A = A / normA
            Y = A.clone()
            Z = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
            for _ in range(max(steps, 6)):
                Y = 0.5 * Y @ (3.0 * Z - Y @ Y)
                Z = 0.5 * (3.0 * Z - Y @ Y) @ Z
            result = (Y * (normA if normA > 0 else 1.0)).to(G.dtype)

    if transposed:
        result = result.mT
    return result.to(G.dtype)


def adjust_lr_for_muon(lr: float, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    return lr * adjusted_ratio


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta=0.95, ns_steps=5, nesterov=True):
    """
    与原作者一致的 muon_update，但使用上面 robust zeropower。
    grad, momentum 均为 2D（或可 reshape 为 2D）
    """
    # momentum 被原地更新以保持 state
    momentum.lerp_(grad, 1 - beta)      # buf = beta * buf + (1-beta) * grad
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    # handle conv filters shape collapsing if needed outside
    if update.ndim == 4:
        update = update.view(len(update), -1)
    u = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # scale by row/col ratio like原作者
    if update.ndim >= 2:
        shape_scale = max(1, grad.size(-2) / grad.size(-1)) ** 0.5
        u = u * shape_scale
    return u


def adam_update(g, exp_avg, exp_avg_sq, step, betas, eps):
    """
    标准 Adam 风格两矩估计 + 偏差校正，返回未乘 lr 的更新项（即 m_hat / (sqrt(v_hat)+eps)）。
    """
    beta1, beta2 = betas
    # update moving averages (这里使用 lerp 与之前风格一致)
    exp_avg.lerp_(g, 1 - beta1)
    exp_avg_sq.lerp_(g.square(), 1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    # 计算偏差校正后的 m_hat, v_hat
    m_hat = exp_avg / (bias_correction1 + 1e-16)
    v_hat = exp_avg_sq / (bias_correction2 + 1e-16)

    update = m_hat / (v_hat.sqrt() + eps)
    return update


# -------------------------
# MuonWithAuxAdam（Distributed variant） - 修改版
# -------------------------
class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant（修改版）:
    - 对 use_muon=True: 使用 Muon 路径（动量->Newton-Schulz），并按 param shape 调整 lr，应用 AdamW 风格 weight decay
    - 对 use_muon=False: 使用 AdamW 风格两矩估计（偏差校正，RMS 缩放）
    保持 param_groups 格式与原始实现兼容。
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "ns_steps"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # Muon 路径：遍历该组所有参数
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # 强制同步（分布式时会需要）
                    state = self.state[p]
                    if len(state) == 0:
                        # momentum buffer 尺寸应与 collapsed grad 相同 -> 我们保留与原实现一致的形状
                        state["momentum_buffer"] = torch.zeros_like(p)

                    g = p.grad
                    if g.ndim > 2:
                        g_flat = g.view(g.size(0), -1)
                    else:
                        g_flat = g

                    # ensure momentum buffer has same shape as g_flat
                    if state["momentum_buffer"].shape != g_flat.shape:
                        state["momentum_buffer"] = torch.zeros_like(g_flat)

                    # compute muon-style update (uses robust zeropower)
                    u = muon_update(g_flat, state["momentum_buffer"], beta=group["momentum"],
                                   ns_steps=group.get("ns_steps", 5), nesterov=True)

                    # adjust lr by param shape
                    adjusted_lr = adjust_lr_for_muon(group["lr"], p.shape)

                    # weight decay (AdamW 风格：在更新前进行乘性衰减)
                    if group["weight_decay"] and group["weight_decay"] != 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])

                    # apply update (reshape back)
                    p.data.add_(u.reshape(p.shape), alpha=-adjusted_lr)
            else:
                # AdamW-style backup for non-muon params
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    g = p.grad
                    update = adam_update(g, state["exp_avg"], state["exp_avg_sq"], state["step"],
                                         group["betas"], group["eps"])

                    # AdamW-style weight decay (multiplicative)
                    if group["weight_decay"] and group["weight_decay"] != 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])

                    # apply update (标准 AdamW: p <- p - lr * update)
                    p.data.add_(update, alpha=-group["lr"])
        return loss


# -------------------------
# SingleDeviceMuonWithAuxAdam（Non-distributed variant） - 修改版
# -------------------------
class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant 的对应修改（行为与上面的 Distributed 版一致，
    但保留单设备遍历逻辑，接口兼容原始实现）。
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "ns_steps"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    g = p.grad
                    if g.ndim > 2:
                        g_flat = g.view(g.size(0), -1)
                    else:
                        g_flat = g

                    if state["momentum_buffer"].shape != g_flat.shape:
                        state["momentum_buffer"] = torch.zeros_like(g_flat)

                    u = muon_update(g_flat, state["momentum_buffer"], beta=group["momentum"],
                                   ns_steps=group.get("ns_steps", 5), nesterov=True)

                    adjusted_lr = adjust_lr_for_muon(group["lr"], p.shape)
                    if group["weight_decay"] and group["weight_decay"] != 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    p.data.add_(u.reshape(p.shape), alpha=-adjusted_lr)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    g = p.grad
                    update = adam_update(g, state["exp_avg"], state["exp_avg_sq"], state["step"],
                                         group["betas"], group["eps"])

                    if group["weight_decay"] and group["weight_decay"] != 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    p.data.add_(update, alpha=-group["lr"])
        return loss
