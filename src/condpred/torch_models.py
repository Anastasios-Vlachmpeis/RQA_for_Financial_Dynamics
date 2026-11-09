"""Torch encoders for E11 (.venv-jepa only; torch imported inside functions)."""
from __future__ import annotations


def ImageCNN(in_channels: int, d: int = 32, depth: int = 2):
    import torch.nn as nn
    from rpjepa.encoders import CNNBlock2D, _coordinate_channels

    class _ImageCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Conv2d(in_channels + 3, d, 1)
            self.blocks = nn.ModuleList(CNNBlock2D(d) for _ in range(depth))

        def forward(self, image):
            import torch

            b, _, _, w = image.shape
            coords = _coordinate_channels(w, image.device, image.dtype).unsqueeze(0).expand(b, -1, -1, -1)
            z = self.input(torch.cat([image, coords], dim=1))
            for block in self.blocks:
                z = block(z)
            return z.mean(dim=-1).transpose(1, 2)

    return _ImageCNN()


def Head(d: int, n_b: int):
    import torch.nn as nn

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d + n_b, 1)

        def forward(self, enc, b_feats):
            import torch

            pooled = enc.mean(dim=1)
            return self.linear(torch.cat([pooled, b_feats], dim=-1)).squeeze(-1)

    return _Head()
