import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SUBJECT_RE = re.compile(r"SN\d{3}")


def _natural_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def _extract_frame_id(path: Path, fallback: int) -> int:
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1])
    return fallback


def _discover_subject_from_path(path: Path) -> str:
    m = SUBJECT_RE.search(str(path))
    if m:
        return m.group(0)
    return path.parent.name


def index_disfa_frames(frame_root: str, subjects: Optional[Sequence[str]] = None) -> Dict[str, List[Tuple[int, Path]]]:
    root = Path(frame_root)
    if not root.exists():
        raise FileNotFoundError(f"DISFA frame_root does not exist: {frame_root}")

    allowed = set(subjects) if subjects else None
    grouped: Dict[str, List[Path]] = {}

    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in IMG_EXTS:
            continue
        subject = _discover_subject_from_path(file_path)
        if allowed is not None and subject not in allowed:
            continue
        grouped.setdefault(subject, []).append(file_path)

    indexed: Dict[str, List[Tuple[int, Path]]] = {}
    for subject, files in grouped.items():
        files = sorted(files, key=_natural_key)
        indexed[subject] = [(_extract_frame_id(p, i), p) for i, p in enumerate(files)]

    if not indexed:
        raise RuntimeError(
            "No DISFA frames found. Ensure frame_root contains extracted image frames "
            "with subject identifiers (e.g., SN001) in their paths."
        )
    return indexed


def _parse_au_file(path: Path) -> Dict[int, float]:
    values: Dict[int, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                continue
            try:
                frame = int(float(parts[0]))
                intensity = float(parts[1])
            except ValueError:
                continue
            values[frame] = intensity
    return values


def load_disfa_au_maps(
    label_root: str,
    subjects: Sequence[str],
    au_list: Sequence[int],
) -> Dict[str, Dict[int, Dict[int, float]]]:
    root = Path(label_root)
    if not root.exists():
        raise FileNotFoundError(f"DISFA label_root does not exist: {label_root}")

    subject_maps: Dict[str, Dict[int, Dict[int, float]]] = {}
    for subject in subjects:
        subject_maps[subject] = {}
        subject_dir = root / subject

        for au in au_list:
            candidates: List[Path] = []
            if subject_dir.exists():
                candidates.extend(sorted(subject_dir.glob(f"*au{au}*.txt")))
                candidates.extend(sorted(subject_dir.glob(f"*AU{au}*.txt")))

            if not candidates:
                candidates.extend(sorted(root.glob(f"**/{subject}*au{au}*.txt")))
                candidates.extend(sorted(root.glob(f"**/{subject}*AU{au}*.txt")))

            subject_maps[subject][au] = _parse_au_file(candidates[0]) if candidates else {}

    return subject_maps


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # PIL -> torch without numpy dependency.
    width, height = img.size
    channels = len(img.getbands())
    storage = torch.ByteStorage.from_buffer(img.tobytes())
    tensor = torch.ByteTensor(storage).view(height, width, channels)
    return tensor.permute(2, 0, 1).contiguous().float().div(255.0)


def _apply_clip_augment(clip: torch.Tensor, hflip_prob: float, color_jitter: float) -> torch.Tensor:
    # clip is [C,T,H,W]; temporal coherence by sharing params across time.
    if random.random() < hflip_prob:
        clip = torch.flip(clip, dims=[3])

    if color_jitter > 0:
        b = 1.0 + random.uniform(-color_jitter, color_jitter)
        c = 1.0 + random.uniform(-color_jitter, color_jitter)
        clip = torch.clamp(clip * b, 0.0, 1.0)

        mean = clip.mean(dim=(2, 3), keepdim=True)
        clip = torch.clamp((clip - mean) * c + mean, 0.0, 1.0)

    return clip


class DISFAClipDataset(Dataset):
    """
    Loads fixed-length clips from extracted DISFA frame folders.

    Expected dataset resources:
      - frame_root: recursively contains frame images with subject id (SNxxx) in path.
      - label_root (optional for AU mode): contains AU text files.
    """

    def __init__(
        self,
        frame_root: str,
        clip_len: int,
        image_size: int,
        clip_stride: int,
        subjects: Optional[Sequence[str]] = None,
        temporal_jitter: int = 0,
        augment: bool = False,
        hflip_prob: float = 0.0,
        color_jitter: float = 0.0,
        return_labels: bool = False,
        label_root: Optional[str] = None,
        au_list: Optional[Sequence[int]] = None,
        label_threshold: float = 1.0,
    ):
        self.clip_len = clip_len
        self.image_size = image_size
        self.clip_stride = clip_stride
        self.temporal_jitter = temporal_jitter
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

        self.return_labels = return_labels
        self.au_list = list(au_list or [])
        self.label_threshold = label_threshold

        self.subject_frames = index_disfa_frames(frame_root, subjects=subjects)
        self.subjects = sorted(self.subject_frames.keys())

        if self.return_labels:
            if label_root is None:
                raise ValueError("label_root is required when return_labels=True")
            if not self.au_list:
                raise ValueError("au_list is required when return_labels=True")
            self.subject_au_maps = load_disfa_au_maps(label_root, self.subjects, self.au_list)
        else:
            self.subject_au_maps = {}

        self.samples: List[Tuple[str, int]] = []
        for subject in self.subjects:
            num_frames = len(self.subject_frames[subject])
            if num_frames < self.clip_len:
                continue
            max_start = num_frames - self.clip_len
            for start in range(0, max_start + 1, self.clip_stride):
                self.samples.append((subject, start))

        if not self.samples:
            raise RuntimeError(
                "No valid clips could be generated. Check clip_len, clip_stride, and frame availability."
            )

    def __len__(self):
        return len(self.samples)

    def _sample_start(self, subject: str, base_start: int) -> int:
        if self.temporal_jitter <= 0:
            return base_start

        max_start = len(self.subject_frames[subject]) - self.clip_len
        jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
        return min(max(base_start + jitter, 0), max_start)

    def _load_clip(self, subject: str, start: int) -> Tuple[torch.Tensor, List[int]]:
        entries = self.subject_frames[subject][start : start + self.clip_len]
        frame_ids = [fid for fid, _ in entries]

        frames: List[torch.Tensor] = []
        for _, path in entries:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            frames.append(_pil_to_tensor(img))

        clip = torch.stack(frames, dim=1)  # [C,T,H,W]
        return clip, frame_ids

    def _clip_labels(self, subject: str, frame_ids: List[int]) -> torch.Tensor:
        labels = torch.zeros(len(self.au_list), dtype=torch.float32)
        subject_map = self.subject_au_maps.get(subject, {})

        for i, au in enumerate(self.au_list):
            au_map = subject_map.get(au, {})
            peak = 0.0
            for frame_id in frame_ids:
                # Robust to 0-based/1-based convention mismatch.
                v = max(au_map.get(frame_id, 0.0), au_map.get(frame_id + 1, 0.0))
                peak = max(peak, v)
            labels[i] = 1.0 if peak >= self.label_threshold else 0.0

        return labels

    def __getitem__(self, idx: int):
        subject, start = self.samples[idx]
        start = self._sample_start(subject, start)

        clip, frame_ids = self._load_clip(subject, start)
        if self.augment:
            clip = _apply_clip_augment(clip, self.hflip_prob, self.color_jitter)

        if self.return_labels:
            labels = self._clip_labels(subject, frame_ids)
            return clip, labels

        return clip
