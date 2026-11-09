"""GPU line-length histograms and host RQA scalars (plan 01b §8)."""
from __future__ import annotations

import numpy as np

from condpred.geometry import Geometry


def _torch():
    import torch

    return torch


def _run_histogram(padded: "object") -> "object":
    """``padded`` is int8 [B, rows, width], False-padded on both ends of each row.

    Run length = end − start on ``diff``, paired in row-major ``nonzero`` order.
    """
    torch = _torch()
    B = padded.shape[0]
    n_bins = padded.shape[-1] - 1
    diff = padded[:, :, 1:].to(torch.int16) - padded[:, :, :-1].to(torch.int16)
    starts = (diff == 1).nonzero(as_tuple=False)
    ends = (diff == -1).nonzero(as_tuple=False)
    hist = torch.zeros(B, n_bins, dtype=torch.int64, device=padded.device)
    if starts.shape[0] == 0:
        return hist
    if starts.shape[0] != ends.shape[0]:
        raise RuntimeError("line-length starts and ends do not pair")
    lengths = ends[:, 2] - starts[:, 2]
    batch = starts[:, 0]
    hist = torch.bincount(batch * n_bins + lengths, minlength=B * n_bins).reshape(B, n_bins)
    return hist


def line_hist_torch(R_batch, *, device=None) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal and vertical length histograms ``[B, n+1]``, identical to ``line_lengths`` counts.

    Index ``L`` is the number of lines of length ``L``. Upper-triangle diagonals only (``k >= 1``),
    matching ``recplay.rqa.line_lengths``.
    """
    torch = _torch()
    if device is None:
        dev = R_batch.device if hasattr(R_batch, "device") else torch.device("cpu")
    else:
        dev = torch.device(device)
    R = torch.as_tensor(R_batch, dtype=torch.bool, device=dev)
    if R.ndim == 2:
        R = R[None]
    B, n, n2 = R.shape
    if n != n2:
        raise ValueError("recurrence matrices must be square")
    ii = torch.arange(n, device=dev)
    i = ii.view(1, n).expand(n, n)
    k = ii.view(n, 1).expand(n, n)
    j = (i + k).clamp(max=n - 1)
    valid = (i + k) < n
    shear = torch.zeros(B, n, n + 2, dtype=torch.int8, device=dev)
    vals = R[:, i, j] & valid
    shear[:, :, 1 : n + 1] = vals.to(torch.int8)
    shear[:, 0, :] = 0
    diag_hist = _run_histogram(shear)
    # Columns of R are rows of R^T. Full columns, as ``line_lengths`` walks ``R[:, j]``.
    vert_pad = torch.zeros(B, n, n + 2, dtype=torch.int8, device=dev)
    vert_pad[:, :, 1 : n + 1] = R.transpose(1, 2).to(torch.int8)
    vert_hist = _run_histogram(vert_pad)
    # Histogram width is n+1 (lengths 0..n). Diff width is n+1, so bins already match.
    return diag_hist[:, : n + 1].detach().cpu().numpy(), vert_hist[:, : n + 1].detach().cpu().numpy()


def _lengths_from_hist(hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.int64)
    lengths = np.repeat(np.arange(hist.size, dtype=np.int64), hist)
    return lengths[lengths > 0]


def rqa_from_histograms(
    diag_hist: np.ndarray,
    vert_hist: np.ndarray,
    n: int,
    geom: Geometry,
    *,
    rr_from_R: float | None = None,
) -> dict:
    """Host scalars from histograms, same formulas as ``window_rqa._rqa_from_lines``."""
    from condpred.features.window_rqa import _rqa_from_lines

    diag = _lengths_from_hist(diag_hist)
    vert = _lengths_from_hist(vert_hist)
    blank = np.zeros((n, n), dtype=bool)
    return _rqa_from_lines(diag, vert, blank, geom, rr=rr_from_R)
