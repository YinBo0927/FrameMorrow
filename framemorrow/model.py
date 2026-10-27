"""Causal prospective frame selector: arXiv:2609.38839v1, Eqs. (1)-(4)."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


@dataclass
class SelectorConfig:
    visual_dim: int = 768
    condition_dim: int = 4096
    hidden_dim: int = 512
    layers: int = 4
    heads: int = 8
    ffn_dim: int = 2048
    queries: int = 4
    max_positions: int = 512
    tau_q: float = 0.10
    modality: str = "text"
    condition_normalization: bool = False

    def __post_init__(self):
        if self.modality not in ("text", "action"):
            raise ValueError("modality must be text or action")
        if not isinstance(self.condition_normalization, bool):
            raise ValueError("condition_normalization must be bool")
        if min(self.visual_dim, self.condition_dim, self.hidden_dim, self.layers,
               self.heads, self.ffn_dim, self.queries, self.max_positions) < 1:
            raise ValueError("dimensions and counts must be positive")
        if self.hidden_dim % self.heads or self.tau_q <= 0:
            raise ValueError("hidden_dim must divide into heads; tau_q must be positive")


class FrameMorrow(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SelectorConfig()
        c = self.config
        self.visual_projection = nn.Linear(c.visual_dim, c.hidden_dim)
        self.condition_projection = nn.Linear(c.condition_dim, c.hidden_dim)
        self.temporal_embedding = nn.Embedding(c.max_positions, c.hidden_dim)
        self.type_embedding = nn.Embedding(4, c.hidden_dim)
        self.start_token = nn.Parameter(torch.randn(c.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            c.hidden_dim, c.heads, c.ffn_dim, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, c.layers, norm=nn.LayerNorm(c.hidden_dim), enable_nested_tensor=False,
        )
        # TransformerEncoder clones its input layer; initialize the clones separately.
        for block in self.transformer.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.query_projection = nn.Linear(c.hidden_dim, c.hidden_dim, bias=False)
        self.key_projection = nn.Linear(c.hidden_dim, c.hidden_dim, bias=False)

    def forward(self, history, recent, condition, history_mask=None,
                recent_mask=None, condition_mask=None, return_queries=False):
        """Frozen features [B,N,D], [B,L,D], [B,C,D]; True masks mean valid.

        Future observations deliberately have no argument in this interface.
        Features must be in chronological order; timestamps are checked by data.py.
        """
        inputs = (history, recent, condition)
        dims = (self.config.visual_dim, self.config.visual_dim, self.config.condition_dim)
        if any(x.ndim != 3 or x.shape[0] != history.shape[0] or x.shape[-1] != d
               for x, d in zip(inputs, dims)):
            raise ValueError("feature batch or dimension mismatch")
        masks = []
        for x, mask in zip(inputs, (history_mask, recent_mask, condition_mask)):
            mask = (torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                    if mask is None else mask)
            if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("valid masks must be bool [B,sequence]")
            if not torch.isfinite(x[mask]).all():
                raise ValueError("valid features must be finite")
            masks.append(mask)
        hm, rm, cm = masks
        if not cm.any(dim=1).all():
            raise ValueError("each example requires a known condition")
        h = self.visual_projection(history.masked_fill(~hm[..., None], 0))
        r = self.visual_projection(recent.masked_fill(~rm[..., None], 0))
        condition = condition.masked_fill(~cm[..., None], 0)
        if self.config.condition_normalization:
            condition = F.layer_norm(condition, (self.config.condition_dim,))
        c = self.condition_projection(condition)
        # Pack per example so padding never changes its temporal positions.
        sequences = [torch.cat((h[b, hm[b]] + self.type_embedding.weight[0],
                                r[b, rm[b]] + self.type_embedding.weight[1],
                                c[b, cm[b]] + self.type_embedding.weight[2]))
                     for b in range(history.shape[0])]
        lengths = torch.tensor([len(x) for x in sequences], device=history.device)
        if (lengths + self.config.queries > self.config.max_positions).any():
            raise ValueError("input exceeds learned position capacity; subsample history")
        sequence = pad_sequence(sequences, batch_first=True)
        valid = torch.arange(sequence.shape[1], device=history.device)[None] < lengths[:, None]
        token = self.start_token.expand(history.shape[0], 1, -1)
        queries = []
        # ponytail: M full causal passes (M=4); add a KV cache only if profiling warrants it.
        for _ in range(self.config.queries):
            sequence = torch.cat((sequence, token + self.type_embedding.weight[3]), dim=1)
            valid = torch.cat((valid, torch.ones_like(valid[:, :1])), dim=1)
            positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
            causal = torch.ones(sequence.shape[1], sequence.shape[1],
                                dtype=torch.bool, device=sequence.device).triu(1)
            output = self.transformer(sequence + self.temporal_embedding(positions),
                                      mask=causal, src_key_padding_mask=~valid)
            token = output[:, -1:]
            queries.append(token)
        q = torch.cat(queries, dim=1)
        logits = (self.query_projection(q).float() @
                  self.key_projection(h).float().transpose(1, 2)) / math.sqrt(self.config.hidden_dim)
        scores = self.config.tau_q * (torch.logsumexp(logits / self.config.tau_q, dim=1)
                                      - math.log(self.config.queries))
        scores = scores.masked_fill(~hm, -torch.inf)
        return (scores, q) if return_queries else scores

    @torch.inference_mode()
    def select(self, history, recent, condition, timestamps, k=4, **masks):
        """Return original candidate indices, sorted by time; never duplicate frames."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if timestamps.shape != history.shape[:2]:
            raise ValueError("timestamps must match history [B,N]")
        scores = self(history, recent, condition, **masks)
        result = []
        for b in range(len(scores)):
            eligible = torch.isfinite(scores[b]).nonzero().flatten()
            if not torch.isfinite(timestamps[b, eligible]).all():
                raise ValueError("candidate timestamps must be finite")
            idx = eligible[torch.argsort(scores[b, eligible], descending=True, stable=True)[:k]]
            result.append(idx[torch.argsort(timestamps[b, idx], stable=True)])
        return result, scores

    def checkpoint_config(self):
        return asdict(self.config)
