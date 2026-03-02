import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from disfa import DISFAClipDataset
from model import MAEConfig, SparseAUModel
from sparse_utils import build_sparse_tensor, make_finetune_sparse_collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--pretrained", type=str, default="")
    return parser.parse_args()


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def f1_scores(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)

    per_au_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    macro_f1 = per_au_f1.mean()

    tp_micro = tp.sum()
    fp_micro = fp.sum()
    fn_micro = fn.sum()
    micro_f1 = (2 * tp_micro + eps) / (2 * tp_micro + fp_micro + fn_micro + eps)

    return per_au_f1, macro_f1, micro_f1


def average_precision_per_au(logits: torch.Tensor, labels: torch.Tensor):
    probs = torch.sigmoid(logits)
    aps = []
    for k in range(labels.shape[1]):
        y_true = labels[:, k]
        y_score = probs[:, k]
        order = torch.argsort(y_score, descending=True)
        y_true = y_true[order]

        tp_cum = torch.cumsum(y_true, dim=0)
        fp_cum = torch.cumsum(1 - y_true, dim=0)
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / (y_true.sum() + 1e-8)

        ap = torch.trapz(precision, recall)
        aps.append(ap)

    aps = torch.stack(aps)
    return aps, aps.mean()


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

    num_aus = len(cfg["finetune"]["au_list"])
    model = SparseAUModel(mae_cfg, num_aus=num_aus).to(device)

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location="cpu")
        if "encoder" in ckpt:
            enc_state = ckpt["encoder"]
        else:
            enc_state = {
                k.replace("encoder.", "", 1): v
                for k, v in ckpt["model"].items()
                if k.startswith("encoder.")
            }
        missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
        print(f"loaded encoder, missing={len(missing)} unexpected={len(unexpected)}")

    data_cfg = cfg["data"]
    ft_cfg = cfg["finetune"]

    ds = DISFAClipDataset(
        frame_root=data_cfg["frame_root"],
        label_root=data_cfg["label_root"],
        au_list=ft_cfg["au_list"],
        label_threshold=data_cfg.get("label_threshold", 1.0),
        clip_len=data_cfg["clip_len"],
        image_size=data_cfg["image_size"],
        clip_stride=data_cfg["clip_stride_finetune"],
        subjects=(data_cfg.get("finetune_subjects") or None),
        temporal_jitter=data_cfg.get("temporal_jitter", 0),
        augment=data_cfg.get("augment_finetune", True),
        hflip_prob=data_cfg.get("hflip_prob", 0.5),
        color_jitter=data_cfg.get("color_jitter", 0.1),
        return_labels=True,
    )

    loader = DataLoader(
        ds,
        batch_size=ft_cfg["batch_size"],
        shuffle=True,
        num_workers=ft_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=make_finetune_sparse_collate_fn(data_cfg["patch_size"]),
    )

    pos_weight = torch.tensor(ft_cfg["pos_weight"], device=device)
    if pos_weight.numel() != num_aus:
        raise ValueError(f"pos_weight length ({pos_weight.numel()}) must match AU count ({num_aus})")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    params = [
        {"params": model.encoder.parameters(), "lr": ft_cfg["lr_encoder"]},
        {"params": model.head.parameters(), "lr": ft_cfg["lr_head"]},
    ]
    optim = torch.optim.AdamW(params, weight_decay=ft_cfg["weight_decay"])
    scaler = GradScaler(enabled=ft_cfg["amp"])

    if ft_cfg["freeze_encoder_epochs"] > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False

    out_dir = Path(ft_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(ft_cfg["epochs"]):
        if epoch == ft_cfg["freeze_encoder_epochs"]:
            for p in model.encoder.parameters():
                p.requires_grad = True

        model.train()
        all_logits = []
        all_labels = []

        for batch in loader:
            sparse_in = build_sparse_tensor(batch["features"], batch["coordinates"], device)
            labels = batch["labels"].to(device)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=ft_cfg["amp"]):
                logits = model(sparse_in)
                if ft_cfg["label_smoothing"] > 0:
                    smooth = ft_cfg["label_smoothing"]
                    labels_s = labels * (1 - smooth) + 0.5 * smooth
                else:
                    labels_s = labels
                loss = criterion(logits, labels_s)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        logits_epoch = torch.cat(all_logits, dim=0)
        labels_epoch = torch.cat(all_labels, dim=0)

        per_au_f1, macro_f1, micro_f1 = f1_scores(logits_epoch, labels_epoch)
        ap_per_au, map_score = average_precision_per_au(logits_epoch, labels_epoch)

        print(
            f"epoch={epoch} macro_f1={macro_f1.item():.4f} "
            f"micro_f1={micro_f1.item():.4f} mAP={map_score.item():.4f}"
        )
        print(f"per_au_f1={per_au_f1.tolist()}")
        print(f"per_au_ap={ap_per_au.tolist()}")

        ckpt = out_dir / f"finetune_epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
