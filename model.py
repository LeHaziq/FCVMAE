import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import MinkowskiEngine as ME
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "MinkowskiEngine is required for sparse 3D CNN execution. "
        "Install it before running these scripts."
    ) from exc


@dataclass
class MAEConfig:
    variant: str = "base"
    in_channels: int = 3
    out_channels: int = 3
    norm: str = "bn"
    act: str = "gelu"
    decoder_depth: int = 2


def _make_norm(norm: str, channels: int):
    if norm == "bn":
        return ME.MinkowskiBatchNorm(channels)
    if norm == "in":
        return ME.MinkowskiInstanceNorm(channels)
    raise ValueError(f"Unsupported norm={norm}")


def _make_act(act: str):
    if act == "relu":
        return ME.MinkowskiReLU(inplace=True)
    if act == "gelu":
        return ME.MinkowskiGELU()
    raise ValueError(f"Unsupported act={act}")


class SparseConvBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        norm: str = "bn",
        act: str = "gelu",
    ):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            dimension=3,
        )
        self.norm = _make_norm(norm, c_out)
        self.act = _make_act(act)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.act(self.norm(self.conv(x)))


class SparseResBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "bn", act: str = "gelu"):
        super().__init__()
        self.block1 = SparseConvBlock(channels, channels, norm=norm, act=act)
        self.block2 = SparseConvBlock(channels, channels, norm=norm, act=act)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class SparseUpBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: Tuple[int, int, int],
        norm: str = "bn",
        act: str = "gelu",
    ):
        super().__init__()
        self.deconv = ME.MinkowskiConvolutionTranspose(
            c_in,
            c_out,
            kernel_size=stride,
            stride=stride,
            dimension=3,
        )
        self.norm = _make_norm(norm, c_out)
        self.act = _make_act(act)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.act(self.norm(self.deconv(x)))


VARIANT_SPECS: Dict[str, Dict[str, List[int]]] = {
    "small": {"channels": [64, 128, 256, 384], "blocks": [1, 2, 2, 1]},
    "base": {"channels": [96, 192, 384, 576], "blocks": [2, 2, 4, 2]},
    "large": {"channels": [128, 256, 512, 768], "blocks": [2, 4, 6, 2]},
}


class SparseEncoder(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        if cfg.variant not in VARIANT_SPECS:
            raise ValueError(f"Unknown variant {cfg.variant}")
        spec = VARIANT_SPECS[cfg.variant]
        c = spec["channels"]
        b = spec["blocks"]

        self.stem = SparseConvBlock(cfg.in_channels, c[0], norm=cfg.norm, act=cfg.act)

        self.stage1 = nn.Sequential(*[SparseResBlock(c[0], cfg.norm, cfg.act) for _ in range(b[0])])
        self.down1 = SparseConvBlock(c[0], c[1], stride=(1, 2, 2), norm=cfg.norm, act=cfg.act)

        self.stage2 = nn.Sequential(*[SparseResBlock(c[1], cfg.norm, cfg.act) for _ in range(b[1])])
        self.down2 = SparseConvBlock(c[1], c[2], stride=(2, 2, 2), norm=cfg.norm, act=cfg.act)

        self.stage3 = nn.Sequential(*[SparseResBlock(c[2], cfg.norm, cfg.act) for _ in range(b[2])])
        self.down3 = SparseConvBlock(c[2], c[3], stride=(2, 2, 2), norm=cfg.norm, act=cfg.act)

        self.stage4 = nn.Sequential(*[SparseResBlock(c[3], cfg.norm, cfg.act) for _ in range(b[3])])

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        return x


class SparseDecoder(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        spec = VARIANT_SPECS[cfg.variant]
        c = spec["channels"]

        self.mask_token = nn.Parameter(torch.zeros(1, c[-1]))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.union = ME.MinkowskiUnion()
        self.up3 = SparseUpBlock(c[3], c[2], stride=(2, 2, 2), norm=cfg.norm, act=cfg.act)
        self.up2 = SparseUpBlock(c[2], c[1], stride=(2, 2, 2), norm=cfg.norm, act=cfg.act)
        self.up1 = SparseUpBlock(c[1], c[0], stride=(1, 2, 2), norm=cfg.norm, act=cfg.act)

        decoder_blocks = [SparseResBlock(c[0], cfg.norm, cfg.act) for _ in range(cfg.decoder_depth)]
        self.decode_refine = nn.Sequential(*decoder_blocks)
        self.pred_head = ME.MinkowskiConvolution(c[0], cfg.out_channels, kernel_size=1, dimension=3)

    def _mask_tensor(
        self,
        encoded: ME.SparseTensor,
        masked_coords: torch.Tensor,
    ) -> Optional[ME.SparseTensor]:
        if masked_coords is None or masked_coords.numel() == 0:
            return None
        feats = self.mask_token.expand(masked_coords.shape[0], -1).contiguous()
        # masked_coords must use the same tensor stride domain as encoded features.
        return ME.SparseTensor(
            features=feats,
            coordinates=masked_coords.int(),
            coordinate_manager=encoded.coordinate_manager,
            device=encoded.device,
        )

    def forward(self, encoded: ME.SparseTensor, masked_coords: Optional[torch.Tensor] = None) -> ME.SparseTensor:
        mask_st = self._mask_tensor(encoded, masked_coords)
        x = encoded if mask_st is None else self.union(encoded, mask_st)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.decode_refine(x)
        x = self.pred_head(x)
        return x


class SparseVideoMAE(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SparseEncoder(cfg)
        self.decoder = SparseDecoder(cfg)

    def forward(
        self,
        visible_st: ME.SparseTensor,
        masked_latent_coords: Optional[torch.Tensor] = None,
    ) -> ME.SparseTensor:
        encoded = self.encoder(visible_st)
        pred = self.decoder(encoded, masked_coords=masked_latent_coords)
        return pred


class SparseAUHead(nn.Module):
    def __init__(self, in_channels: int, num_aus: int):
        super().__init__()
        self.pool = ME.MinkowskiGlobalAvgPooling()
        self.classifier = nn.Linear(in_channels, num_aus)

    def forward(self, x: ME.SparseTensor) -> torch.Tensor:
        x = self.pool(x)
        return self.classifier(x.F)


class SparseAUModel(nn.Module):
    def __init__(self, mae_cfg: MAEConfig, num_aus: int):
        super().__init__()
        self.encoder = SparseEncoder(mae_cfg)
        c_last = VARIANT_SPECS[mae_cfg.variant]["channels"][-1]
        self.head = SparseAUHead(c_last, num_aus)

    def forward(self, x: ME.SparseTensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.head(feats)
        return logits
