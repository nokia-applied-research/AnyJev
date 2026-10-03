"""L1 alternative: top-label histogram binning (Zadrozny and Elkan, ICML 2001).

Temperature scaling reshapes every probability with one number. Histogram binning is a lookup
table instead: sort the calibration set by top-1 confidence, cut it into bins of equal size, and
replace any confidence that falls in a bin with the top-1 accuracy measured in that bin. It can
fix a miscalibration of any shape, where one temperature can fix only one, at the cost of needing
more labels per bin.

The argmax never changes (`docs/levels.md`: L1 reshapes confidence, it does not change the
ranking). Binning each option through its own table would break that, since two options in
different bins can swap. So only the top-1 probability is looked up, and the other options are
all rescaled by one common factor, which keeps their order. One case is left: when a bin's
accuracy is far below the confidence, the rescaled runner-up could overtake the top option
([0.50, 0.45, 0.05] with a bin value of 0.30 would become [0.30, 0.63, 0.07]). The new top-1 is
therefore floored just above the rescaled runner-up. Accuracy at this level is then exactly the
accuracy of the probabilities it was given.

Bins with fewer than `min_count` calibration items are merged into a neighbour, so a small or
heavily tied calibration set gives fewer, better-estimated bins; with fewer than `min_count`
items there is one bin, the overall top-1 accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

FLOOR_MARGIN = 1e-9  
EPS = 1e-12             


def _as_rows(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = p[None]
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"probs must be [N, K] or [K] with K >= 2, got shape {np.shape(probs)}")
    return p


@dataclass
class HistogramBinning:
    """Build with `fit` or `from_dict`. There are no defaults: unlike a temperature of 1, no bin
    table leaves probabilities unchanged, so an unfitted calibrator would be wrong, not neutral."""

    edges: np.ndarray     
    values: np.ndarray      
    counts: np.ndarray      

    @classmethod
    def fit(cls, probs: np.ndarray, labels: Sequence[int], n_bins: int = 10,
            min_count: int = 20) -> "HistogramBinning":
        """Equal-mass bins over top-1 confidence, each mapped to its measured top-1 accuracy.
        probs: [N, K]; labels: option indices."""
        if n_bins < 1 or min_count < 1:
            raise ValueError("n_bins and min_count must be at least 1")
        p = _as_rows(probs)
        y = np.asarray(labels, dtype=int)
        if len(y) != len(p) or len(y) == 0:
            raise ValueError(f"need one label per row and at least one row, got {len(p)} rows, {len(y)} labels")
        if y.min() < 0 or y.max() >= p.shape[1]:
            raise ValueError(f"labels must be option indices in [0, {p.shape[1]})")
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == y).astype(np.float64)

        n_groups = max(1, min(n_bins, len(conf) // min_count))
        groups = np.array_split(np.sort(conf), n_groups)
        edges = np.array([(a[-1] + b[0]) / 2 for a, b in zip(groups[:-1], groups[1:])])

        while True:
            idx = np.searchsorted(edges, conf, side="right")
            counts = np.bincount(idx, minlength=len(edges) + 1)
            if len(edges) == 0 or counts.min() >= min_count:
                break
            k = int(np.argmin(counts))
            if k == 0:
                drop = 0
            elif k == len(edges):
                drop = k - 1
            else:
                drop = k - 1 if counts[k - 1] <= counts[k + 1] else k
            edges = np.delete(edges, drop)
        values = np.bincount(idx, weights=correct, minlength=len(counts)) / counts
        return cls(edges=edges, values=values, counts=counts)

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """[N, K] (or [K]) probabilities in, the same shape out; the argmax of every row is kept."""
        shape = np.shape(probs)
        p = _as_rows(probs)
        n, k = p.shape
        rows = np.arange(n)
        top = p.argmax(axis=1)
        c = p[rows, top]
        target = self.values[np.searchsorted(self.edges, c, side="right")]

        others = p.copy()
        others[rows, top] = 0.0
        rest = others.sum(axis=1)                
        runner = others.max(axis=1)
        spread = rest <= 1e-12                   

        with np.errstate(divide="ignore", invalid="ignore"):
            floor = np.where(spread, 1.0 / k, runner / (rest + runner))
        c_new = np.minimum(np.maximum(target, floor + FLOOR_MARGIN), 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(spread, 0.0, (1.0 - c_new) / rest)
        out = others * scale[:, None]
        out[spread] = ((1.0 - c_new[spread]) / (k - 1))[:, None]
        out[rows, top] = c_new
        
        out = np.clip(out, EPS, None)
        out /= out.sum(axis=1, keepdims=True)
        return out.reshape(shape)

    def to_dict(self) -> Dict[str, Any]:
        return {"method": "histogram_binning", "edges": self.edges.tolist(), "values": self.values.tolist(),
                "counts": self.counts.astype(int).tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HistogramBinning":
        if d.get("method") != "histogram_binning":
            raise ValueError(f"not a histogram-binning artifact: method={d.get('method')!r}")
        edges = np.asarray(d["edges"], dtype=np.float64).reshape(-1)
        values = np.asarray(d["values"], dtype=np.float64).reshape(-1)
        if len(values) != len(edges) + 1:
            raise ValueError("a binning artifact needs one more value than edges")
        return cls(edges=edges, values=values, counts=np.asarray(d.get("counts", [0] * len(values)), dtype=int))