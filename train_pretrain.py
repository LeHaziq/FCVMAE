import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from disfa import DISFAClipDataset
from model import MAEConfig, SparseVideoMAE
from sparse_utils import build_sparse_tensor, make_pretrain_patch_collate_fn
from voxceleb2 import VoxCeleb2ClipDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_tube_block_mask(
    coords: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    mask_ratio: float,
    tube_prob: float = 0.7,
):
    """Hybrid tube/block masking on patch-grid coordinates for one sample."""
    tg, hg, wg = grid_shape
    total = coords.shape[0]

    if random.random() < tube_prob:
        spatial = [(y, x) for y in range(hg) for x in range(wg)]
        n_spatial_keep = max(1, int(len(spatial) * (1 - mask_ratio)))
        n_spatial_keep = min(n_spatial_keep, len(spatial))
        keep_spatial = set(random.sample(spatial, n_spatial_keep))
        keep = [idx for idx, c in enumerate(coords.tolist()) if (c[2], c[3]) in keep_spatial]
    else:
        n_keep = max(1, int(total * (1 - mask_ratio)))
        n_keep = min(n_keep, total)
        keep = sorted(random.sample(range(total), n_keep))

    keep_idx = torch.tensor(keep, dtype=torch.long)
    mask_bool = torch.ones(total, dtype=torch.bool)
    mask_bool[keep_idx] = False
    masked_idx = torch.nonzero(mask_bool, as_tuple=False).squeeze(1)
    return keep_idx, masked_idx


def masked_recon_loss(
    pred_st: ME.SparseTensor,
    target_feats: torch.Tensor,
    target_coords: torch.Tensor,
    huber_delta: float = 0.1,
):
    # Query decoded predictions at masked coordinates.
    query_coords = target_coords.to(device=pred_st.device, dtype=pred_st.dtype)
    pred = pred_st.features_at_coordinates(query_coords)
    # Compute loss in FP32 for stability when AMP is enabled.
    pred_f = pred.float()
    target_f = target_feats.float()
    recon = F.huber_loss(pred_f, target_f, delta=huber_delta)

    # Temporal-difference consistency surrogate.
    sort_key = (
        target_coords[:, 0].long() * 1_000_000_000
        + target_coords[:, 1].long() * 1_000_000
        + target_coords[:, 2].long() * 1_000
        + target_coords[:, 3].long()
    )
    sort_idx = torch.argsort(sort_key)
    p = pred_f[sort_idx]
    t = target_f[sort_idx]

    if p.shape[0] > 1:
        temporal = F.l1_loss(p[1:] - p[:-1], t[1:] - t[:-1])
    else:
        temporal = torch.tensor(0.0, device=pred.device)

    loss = recon + 0.1 * temporal
    return loss, {"recon": recon.item(), "temporal": float(temporal.item())}


def to_latent_mask_coords(masked_coords: torch.Tensor):
    # Encoder downsamples with strides (1,2,2) -> (2,2,2) -> (2,2,2).
    # Total stride wrt patch-grid: (4,8,8) for (t,y,x).
    out = masked_coords.clone()
    out[:, 1] = torch.div(out[:, 1], 4, rounding_mode="floor")
    out[:, 2] = torch.div(out[:, 2], 8, rounding_mode="floor")
    out[:, 3] = torch.div(out[:, 3], 8, rounding_mode="floor")
    out = torch.unique(out, dim=0)
    return out.int()


def cosine_mask_ratio(step: int, total_steps: int, start: float, end: float):
    alpha = min(1.0, step / max(total_steps, 1))
    return end + 0.5 * (start - end) * (1.0 + torch.cos(torch.tensor(alpha * 3.14159265))).item()


def build_pretrain_dataset(cfg: Dict):
    data_cfg = cfg["data"]
    pre_cfg = cfg["pretrain"]
    dataset_name = pre_cfg.get("dataset", "voxceleb2").lower()

    if dataset_name == "voxceleb2":
        ds = VoxCeleb2ClipDataset(
            frame_root=data_cfg["voxceleb2_frame_root"],
            clip_len=data_cfg["clip_len"],
            image_size=data_cfg["image_size"],
            clip_stride=data_cfg["clip_stride_pretrain"],
            speakers=(data_cfg.get("voxceleb2_speakers") or None),
            temporal_jitter=data_cfg.get("temporal_jitter", 0),
            augment=data_cfg.get("augment_pretrain", True),
            hflip_prob=data_cfg.get("hflip_prob", 0.5),
            color_jitter=data_cfg.get("color_jitter", 0.2),
            max_tracks=int(data_cfg.get("voxceleb2_max_tracks", 0) or 0),
            max_samples=int(data_cfg.get("voxceleb2_max_samples", 0) or 0),
        )
        return ds

    if dataset_name == "disfa":
        ds = DISFAClipDataset(
            frame_root=data_cfg["frame_root"],
            clip_len=data_cfg["clip_len"],
            image_size=data_cfg["image_size"],
            clip_stride=data_cfg["clip_stride_pretrain"],
            subjects=(data_cfg.get("pretrain_subjects") or None),
            temporal_jitter=data_cfg.get("temporal_jitter", 0),
            augment=data_cfg.get("augment_pretrain", True),
            hflip_prob=data_cfg.get("hflip_prob", 0.5),
            color_jitter=data_cfg.get("color_jitter", 0.2),
            return_labels=False,
        )
        return ds

    raise ValueError("Unsupported pretrain.dataset. Use 'voxceleb2' or 'disfa'.")


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mae_cfg = MAEConfig(
        variant=cfg["model"]["variant"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        norm=cfg["model"]["norm"],
        act=cfg["model"]["act"],
        decoder_depth=cfg["model"]["decoder_depth"],
    )
    model = SparseVideoMAE(mae_cfg).to(device)

    data_cfg = cfg["data"]
    try:
        ds = build_pretrain_dataset(cfg)
    except FileNotFoundError as exc:
        dataset_name = cfg["pretrain"].get("dataset", "voxceleb2").lower()
        if dataset_name == "voxceleb2":
            raw_root = cfg["data"].get("voxceleb2_raw_root", "/path/to/VoxCeleb2/raw/dev")
            frame_root = cfg["data"].get("voxceleb2_frame_root", "/path/to/VoxCeleb2/frames/dev")
            hint = (
                "VoxCeleb2 frame_root is missing. Extract frames first, e.g.\\n"
                f"python extract_voxceleb2_frames.py --input-root {raw_root} "
                f"--output-root {frame_root} --fps 25 --workers 8"
            )
            raise FileNotFoundError(f"{exc}\n{hint}") from exc
        raise

    loader = DataLoader(
        ds,
        batch_size=cfg["pretrain"]["batch_size"],
        shuffle=True,
        num_workers=cfg["pretrain"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=make_pretrain_patch_collate_fn(data_cfg["patch_size"]),
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["pretrain"]["lr"],
        weight_decay=cfg["pretrain"]["weight_decay"],
    )
    scaler = GradScaler(enabled=cfg["pretrain"]["amp"])

    epochs = cfg["pretrain"]["epochs"]
    out_dir = Path(cfg["pretrain"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    total_steps = epochs * max(1, len(loader))

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            ratio = cosine_mask_ratio(
                global_step,
                total_steps,
                cfg["pretrain"]["mask_ratio_start"],
                cfg["pretrain"]["mask_ratio_end"],
            )

            vis_feats = []
            vis_coords = []
            masked_targets = []
            masked_coords = []

            for feats, coords, grid_shape in zip(
                batch["feats_list"],
                batch["coords_list"],
                batch["grid_shapes"],
            ):
                keep_idx, masked_idx = sample_tube_block_mask(
                    coords,
                    grid_shape,
                    mask_ratio=ratio,
                    tube_prob=cfg["pretrain"].get("tube_prob", 0.7),
                )
                vis_feats.append(feats[keep_idx])
                vis_coords.append(coords[keep_idx])
                masked_targets.append(feats[masked_idx])
                masked_coords.append(coords[masked_idx])

            if not masked_targets:
                continue

            visible_st = build_sparse_tensor(
                torch.cat(vis_feats, dim=0),
                torch.cat(vis_coords, dim=0),
                device,
            )
            masked_targets_t = torch.cat(masked_targets, dim=0).to(device)
            masked_coords_t = torch.cat(masked_coords, dim=0).to(device)
            latent_mask_coords = to_latent_mask_coords(masked_coords_t)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["pretrain"]["amp"]):
                pred_st = model(visible_st, masked_latent_coords=latent_mask_coords)
                loss, logs = masked_recon_loss(pred_st, masked_targets_t, masked_coords_t)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if global_step % cfg["pretrain"]["log_interval"] == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"mask_ratio={ratio:.3f} loss={loss.item():.4f} "
                    f"recon={logs['recon']:.4f} temporal={logs['temporal']:.4f}"
                )

            global_step += 1

        ckpt = out_dir / f"pretrain_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "epoch": epoch,
            },
            ckpt,
        )
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
