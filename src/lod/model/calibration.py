"""Metrics and post-hoc temperature scaling.

A decision model is only useful if its probabilities mean something, so every
number here is computed over each question's own option set.
"""

from __future__ import annotations

import math

import torch

EPS = 1e-12


def _masked_log_probs(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """log softmax over the valid slots only; 0 (not -inf) elsewhere."""
    masked = logits.float().masked_fill(~valid, float("-inf"))
    logp = torch.log_softmax(masked, dim=-1)
    return torch.where(valid, logp, torch.zeros((), dtype=logp.dtype, device=logp.device))


def probabilities(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = logits.float().masked_fill(~valid, float("-inf"))
    p = torch.softmax(masked, dim=-1)
    return torch.where(valid, p, torch.zeros((), dtype=p.dtype, device=p.device))


def metrics(
    logits: torch.Tensor,  # [N, K]
    valid: torch.Tensor,  # [N, K] bool
    target: torch.Tensor,  # [N, K] probabilities
    n_bins: int = 15,
) -> dict[str, float]:
    """acc / nll / brier / ece over N questions.

    `nll` is the log score against the target distribution (for a hard target
    that is -log p of the correct option); `ece` uses `n_bins` equal-width bins
    of the top probability.
    """
    n = int(logits.shape[0])
    if n == 0:
        return {"n": 0, "acc": float("nan"), "nll": float("nan"),
                "brier": float("nan"), "ece": float("nan")}

    logp = _masked_log_probs(logits, valid)
    probs = probabilities(logits, valid)
    label = target.argmax(dim=-1)
    pred = probs.argmax(dim=-1)
    conf = probs.max(dim=-1).values

    correct = (pred == label).float()
    acc = correct.mean().item()
    nll = (-(target * logp).sum(dim=-1)).mean().item()
    brier = (((probs - target) ** 2) * valid).sum(dim=-1).mean().item()

    ece = 0.0
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = int(in_bin.sum())
        if m:
            ece += (m / n) * abs(correct[in_bin].mean().item() - conf[in_bin].mean().item())

    return {"n": n, "acc": acc, "nll": nll, "brier": brier, "ece": ece}


def reliability_bins(
    conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15
) -> list[dict[str, float]]:
    """Per-bin counts, mean confidence and accuracy, for reliability diagrams."""
    out = []
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = int(in_bin.sum())
        out.append({
            "lo": lo, "hi": hi, "n": m,
            "conf": conf[in_bin].mean().item() if m else float("nan"),
            "acc": correct[in_bin].float().mean().item() if m else float("nan"),
        })
    return out


def fit_temperature(
    logits: torch.Tensor,  # [N, K]
    valid: torch.Tensor,
    target: torch.Tensor,
    max_iter: int = 100,
) -> float:
    """Scalar T minimising the NLL of `logits / T`, optimised over log T."""
    logits = logits.float().detach()
    # invalid slots hold -inf; dividing those by T would send infinite gradients
    # into the optimiser, so scale the finite slots and re-mask afterwards
    finite = torch.where(valid, logits, torch.zeros((), dtype=logits.dtype))
    neg_inf = torch.full_like(logits, float("-inf"))
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        scaled = torch.where(valid, finite / log_t.exp(), neg_inf)
        logp = torch.log_softmax(scaled, dim=-1)
        logp = torch.where(valid, logp, torch.zeros((), dtype=logp.dtype))
        loss = (-(target * logp).sum(dim=-1)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp().item())
    return t if math.isfinite(t) and t > 0 else 1.0


def temperature_per_question(valid: torch.Tensor, t: float | torch.Tensor,
                             logn: float | torch.Tensor = 0.0) -> torch.Tensor:
    """[N, 1] divisor T(N) = T * (N/2)^logn, N = each question's option count.

    The same map `OptionScoringModel.temperature_for` applies at serve time."""
    n = valid.sum(-1, keepdim=True).clamp_min(1).float()
    return t * torch.exp(logn * (torch.log(n) - math.log(2.0)))


def fit_temperature_logn(
    logits: torch.Tensor,  # [N, K]
    valid: torch.Tensor,
    target: torch.Tensor,
    max_iter: int = 200,
) -> tuple[float, float]:
    """(T, slope) minimising the NLL of `logits / (T * (N/2)^slope)`.

    log N as an input to the calibration map. A fitted slope of ~0 says one
    scalar was enough; a clearly non-zero one says questions with many options want a
    different correction from binary ones, which one scalar averages away.
    """
    logits = logits.float().detach()
    finite = torch.where(valid, logits, torch.zeros((), dtype=logits.dtype))
    neg_inf = torch.full_like(logits, float("-inf"))
    logn_q = torch.log(valid.sum(-1, keepdim=True).clamp_min(1).float()) - math.log(2.0)
    params = torch.zeros(2, requires_grad=True)     # log T, slope
    opt = torch.optim.LBFGS([params], lr=0.1, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        div = torch.exp(params[0] + params[1] * logn_q)
        scaled = torch.where(valid, finite / div, neg_inf)
        logp = torch.log_softmax(scaled, dim=-1)
        logp = torch.where(valid, logp, torch.zeros((), dtype=logp.dtype))
        loss = (-(target * logp).sum(dim=-1)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    t, slope = float(params[0].detach().exp()), float(params[1].detach())
    if not (math.isfinite(t) and t > 0 and math.isfinite(slope)):
        return 1.0, 0.0
    return t, slope


# ---- K-head ensemble: the diagnostics that say whether it bought anything ----


def aurc(logits: torch.Tensor, valid: torch.Tensor, target: torch.Tensor) -> float:
    """Area under the risk-coverage curve, max-prob as the confidence score.

    Questions are ranked most-confident first; the risk at coverage i is the error
    rate over the first i. Lower is better. `lod.evaluation.dumps.aurc` is the same
    definition over an eval dump, so an eval report and a gate report cannot disagree.
    """
    n = int(logits.shape[0])
    if n == 0:
        return float("nan")
    probs = probabilities(logits, valid)
    correct = (probs.argmax(dim=-1) == target.argmax(dim=-1)).float()
    order = torch.argsort(probs.max(dim=-1).values, descending=True)
    wrong = 1.0 - correct[order]
    i = torch.arange(1, n + 1, dtype=torch.float32)
    return float((wrong.cumsum(0) / i).mean())


def p_wrong_at_confidence(
    logits: torch.Tensor, valid: torch.Tensor, target: torch.Tensor, floor: float = 0.9
) -> dict[str, float]:
    """P(wrong | top probability >= floor) -- the confidently-wrong rate.

    The shipped model's cited failure mode. `n` is the size of the conditioning set,
    which matters: on a small eval the rate is a ratio of small integers.
    """
    probs = probabilities(logits, valid)
    conf = probs.max(dim=-1).values
    sel = conf >= floor
    m = int(sel.sum())
    if m == 0:
        return {"floor": floor, "n": 0, "share": 0.0, "p_wrong": float("nan")}
    wrong = (probs.argmax(dim=-1)[sel] != target.argmax(dim=-1)[sel]).float()
    return {"floor": floor, "n": m, "share": m / int(logits.shape[0]),
            "p_wrong": float(wrong.mean())}


def head_probabilities(head_grouped: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[Nq, K, H] per-head logits -> [Nq, K, H] per-head probabilities, 0 on invalid."""
    masked = head_grouped.float().masked_fill(~valid.unsqueeze(-1), float("-inf"))
    p = torch.softmax(masked, dim=1)
    return torch.where(valid.unsqueeze(-1), p, torch.zeros((), dtype=p.dtype))


def head_correlation(head_grouped: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Mean pairwise Pearson correlation between the heads' probability vectors.

    Each head's prediction is flattened over every (question, valid option) cell and
    two heads are correlated over that vector; the report is the mean over the
    H*(H-1)/2 pairs. Flattened rather than per question and then averaged, because a
    two-option question's per-question correlation is +1 or -1 by construction (the
    pair sums to 1) and averaging those measures nothing.

    1.0 means the heads are the same function and the ensemble is one head in a
    trench coat; this is the number that says whether the shards and the per-head
    option orders did their job.
    """
    h = head_grouped.shape[-1]
    if h < 2 or head_grouped.shape[0] == 0:
        return {"heads": h, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    p = head_probabilities(head_grouped, valid)
    flat = p[valid]  # [n_cells, H]
    c = torch.corrcoef(flat.t())
    iu = torch.triu_indices(h, h, offset=1)
    pairs = c[iu[0], iu[1]]
    pairs = pairs[torch.isfinite(pairs)]
    if pairs.numel() == 0:
        return {"heads": h, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"heads": h, "mean": float(pairs.mean()),
            "min": float(pairs.min()), "max": float(pairs.max()),
            "n_pairs": int(pairs.numel())}


def head_disagreement(head_grouped: torch.Tensor, valid: torch.Tensor) -> float:
    """Mean over head pairs of the share of questions where their argmax differs."""
    h = head_grouped.shape[-1]
    if h < 2 or head_grouped.shape[0] == 0:
        return float("nan")
    pick = head_probabilities(head_grouped, valid).argmax(dim=1)  # [Nq, H]
    iu = torch.triu_indices(h, h, offset=1)
    return float((pick[:, iu[0]] != pick[:, iu[1]]).float().mean())


def fit_temperature_ensemble(
    head_grouped: torch.Tensor,  # [Nq, K, H] raw per-head logits
    valid: torch.Tensor,
    target: torch.Tensor,
    max_iter: int = 100,
) -> float:
    """One scalar T minimising the NLL of the ENSEMBLE mean p̄(T), not of any head.

    p̄(T) = mean_h softmax(logits_h / T). The objective is the log score of the thing
    that ships -- fitting a temperature per head and averaging afterwards calibrates
    H distributions none of which is the output. One shared scalar also keeps
    `lod.json.temperature` meaning one scalar applied to all logits at serve time.
    """
    g = head_grouped.float().detach()
    v = valid.unsqueeze(-1)
    # invalid slots hold -inf; dividing those by T sends infinite gradients into LBFGS
    finite = torch.where(v, g, torch.zeros((), dtype=g.dtype))
    neg_inf = torch.full_like(g, float("-inf"))
    n_heads = g.shape[-1]
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        scaled = torch.where(v, finite / log_t.exp(), neg_inf)
        logp = torch.log_softmax(scaled, dim=1)
        mean_logp = torch.logsumexp(logp, dim=-1) - math.log(n_heads)
        mean_logp = torch.where(valid, mean_logp, torch.zeros((), dtype=mean_logp.dtype))
        loss = (-(target * mean_logp).sum(dim=-1)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp().item())
    return t if math.isfinite(t) and t > 0 else 1.0
