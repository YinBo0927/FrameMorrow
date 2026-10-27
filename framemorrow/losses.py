"""Detached future teacher and per-example ranking distillation, Eqs. (5)-(8)."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def teacher_scores(history, future, history_mask=None, future_mask=None, tau_f=0.05):
    if tau_f <= 0 or history.ndim != 3 or future.ndim != 3:
        raise ValueError("positive tau_f and [B,sequence,D] features required")
    if history.shape[0] != future.shape[0] or history.shape[2] != future.shape[2]:
        raise ValueError("teacher feature dimensions differ")
    hm = (torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
          if history_mask is None else history_mask)
    fm = (torch.ones(future.shape[:2], dtype=torch.bool, device=future.device)
          if future_mask is None else future_mask)
    if not fm.any(dim=1).all() or not hm.any(dim=1).all():
        raise ValueError("training examples need eligible history and a continuation")
    h = F.normalize(history.float().masked_fill(~hm[..., None], 0), dim=-1)
    y = F.normalize(future.float().masked_fill(~fm[..., None], 0), dim=-1)
    similarity = (h @ y.transpose(1, 2)).masked_fill(~fm[:, None], -torch.inf)
    scores = tau_f * (torch.logsumexp(similarity / tau_f, dim=-1)
                      - fm.sum(dim=1).float().log()[:, None])
    return scores.masked_fill(~hm, -torch.inf)


def ranking_loss(student, teacher, valid=None, tau_r=0.10, tau_s=1.00,
                 delta=0.05, pair_weight=0.50):
    if min(tau_r, tau_s, delta) <= 0 or pair_weight < 0 or student.shape != teacher.shape:
        raise ValueError("invalid ranking settings or score shapes")
    valid = torch.isfinite(teacher) if valid is None else valid
    if not valid.any(dim=1).all():
        raise ValueError("each training example requires at least one candidate")
    teacher = teacher.detach().float().masked_fill(~valid, -torch.inf)
    student = student.float().masked_fill(~valid, -torch.inf)
    if not torch.isfinite(student[valid]).all() or not torch.isfinite(teacher[valid]).all():
        raise ValueError("valid ranking scores must be finite")
    p = F.softmax(teacher / tau_r, dim=-1)
    log_p = F.log_softmax(student / tau_s, dim=-1).masked_fill(~valid, 0)
    listwise = -(p * log_p).sum(dim=-1).mean()
    # ponytail: all N^2 pairs, N<=128 in training; chunk pairs if that bound grows.
    t = teacher.masked_fill(~valid, 0)
    s = student.masked_fill(~valid, 0)
    pairs = valid[:, :, None] & valid[:, None, :] & ((t[:, :, None] - t[:, None, :]) >= delta)
    terms = F.softplus(-(s[:, :, None] - s[:, None, :]))
    pairwise = (terms.masked_fill(~pairs, 0).sum(dim=(1, 2)) /
                pairs.sum(dim=(1, 2)).clamp_min(1)).mean()
    return {"loss": listwise + pair_weight * pairwise, "listwise": listwise, "pairwise": pairwise}
