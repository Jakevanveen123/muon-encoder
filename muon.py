import math
import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def ns_polynomial(x, coeffs=NS_COEFFS, steps=NS_STEPS):
    a, b, c = coeffs
    for _ in range(steps):
        x = a * x + b * x**3 + c * x**5
    return x


def ns_threshold(target=0.1):
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ns_polynomial(mid) < target else (lo, mid)
    return hi


NS_THRESHOLD = ns_threshold()


def newton_schulz(G, dtype=torch.bfloat16, coeffs=NS_COEFFS, steps=NS_STEPS):
    a, b, c = coeffs
    X = G.to(dtype)
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def spectral_quantiles(G):
    s = torch.linalg.svdvals(G.float() / G.norm())
    r = s.numel()
    out = {f"sv_q{q}": s[math.ceil(q * r) - 1].item() for q in QUANTILES}
    out["frac_below_threshold"] = (s < NS_THRESHOLD).float().mean().item()
    return out


class Muon(torch.optim.Optimizer):
    def __init__(
        self, params, lr=0.02, momentum=0.95, weight_decay=0.0, nesterov=True, ns_dtype=torch.bfloat16
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)
        self.ns_dtype = ns_dtype
        self.spectra = {}

    @torch.no_grad()
    def step(self, track_spectra=False):
        self.spectra = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                g = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                if track_spectra:
                    self.spectra[p] = spectral_quantiles(g)
                update = newton_schulz(g, self.ns_dtype) * max(1, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
