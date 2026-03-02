from typing import Callable, Dict, List, Sequence, Tuple

import torch

import MinkowskiEngine as ME


def clip_to_patch_grid(clip: torch.Tensor, patch: Tuple[int, int, int], batch_index: int = 0):
    """
    Convert dense clip [C,T,H,W] to patch-grid features.

    Returns:
      feats: [N, C]
      coords: [N, 4] -> [b,t,y,x]
      grid_shape: (Tg, Hg, Wg)
    """
    c, t, h, w = clip.shape
    pt, ph, pw = patch
    if t % pt != 0 or h % ph != 0 or w % pw != 0:
        raise ValueError(
            f"Clip shape {(t, h, w)} must be divisible by patch size {(pt, ph, pw)}"
        )

    tg, hg, wg = t // pt, h // ph, w // pw
    patches = clip.view(c, tg, pt, hg, ph, wg, pw).mean(dim=(2, 4, 6))
    patches = patches.permute(1, 2, 3, 0).contiguous()  # [Tg,Hg,Wg,C]

    coords = []
    feats = []
    for ti in range(tg):
        for yi in range(hg):
            for xi in range(wg):
                coords.append([batch_index, ti, yi, xi])
                feats.append(patches[ti, yi, xi])

    coords_t = torch.tensor(coords, dtype=torch.int32)
    feats_t = torch.stack(feats, dim=0).float()
    return feats_t, coords_t, (tg, hg, wg)


def make_pretrain_patch_collate_fn(patch_size: Sequence[int]) -> Callable:
    patch = tuple(int(v) for v in patch_size)

    def collate(batch: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        feats_list: List[torch.Tensor] = []
        coords_list: List[torch.Tensor] = []
        grid_shapes: List[Tuple[int, int, int]] = []

        for b, clip in enumerate(batch):
            feats, coords, grid_shape = clip_to_patch_grid(clip, patch, batch_index=b)
            feats_list.append(feats)
            coords_list.append(coords)
            grid_shapes.append(grid_shape)

        return {
            "feats_list": feats_list,
            "coords_list": coords_list,
            "grid_shapes": grid_shapes,
        }

    return collate


def make_finetune_sparse_collate_fn(patch_size: Sequence[int]) -> Callable:
    patch = tuple(int(v) for v in patch_size)

    def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        feat_batches: List[torch.Tensor] = []
        coord_batches: List[torch.Tensor] = []
        label_batches: List[torch.Tensor] = []

        for b, (clip, labels) in enumerate(batch):
            feats, coords, _ = clip_to_patch_grid(clip, patch, batch_index=b)
            feat_batches.append(feats)
            coord_batches.append(coords)
            label_batches.append(labels)

        return {
            "features": torch.cat(feat_batches, dim=0),
            "coordinates": torch.cat(coord_batches, dim=0),
            "labels": torch.stack(label_batches, dim=0).float(),
        }

    return collate


def build_sparse_tensor(features: torch.Tensor, coordinates: torch.Tensor, device: torch.device) -> ME.SparseTensor:
    return ME.SparseTensor(
        features=features.to(device),
        coordinates=coordinates.to(device),
        device=device,
    )
