import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = {"dev", "test", "train", "val", "validation"}
ID_RE = re.compile(r"^id\d+")


def _natural_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    width, height = img.size
    channels = len(img.getbands())
    storage = torch.ByteStorage.from_buffer(img.tobytes())
    tensor = torch.ByteTensor(storage).view(height, width, channels)
    return tensor.permute(2, 0, 1).contiguous().float().div(255.0)


def _apply_clip_augment(clip: torch.Tensor, hflip_prob: float, color_jitter: float) -> torch.Tensor:
    # clip is [C,T,H,W]; apply same params across all frames.
    if random.random() < hflip_prob:
        clip = torch.flip(clip, dims=[3])

    if color_jitter > 0:
        b = 1.0 + random.uniform(-color_jitter, color_jitter)
        c = 1.0 + random.uniform(-color_jitter, color_jitter)
        clip = torch.clamp(clip * b, 0.0, 1.0)
        mean = clip.mean(dim=(2, 3), keepdim=True)
        clip = torch.clamp((clip - mean) * c + mean, 0.0, 1.0)

    return clip


def index_voxceleb2_tracks(
    frame_root: str,
    speakers: Optional[Sequence[str]] = None,
    min_frames: int = 1,
    max_tracks: int = 0,
) -> Dict[str, List[Path]]:
    """
    Index VoxCeleb2 frame tracks.

    Expected structure (flexible):
      <frame_root>/<speaker>/<video>/<track>/frame_XXXX.jpg
    or any nested structure where each leaf directory holds frame images.

    Returns:
      track_map: {track_id: sorted_frame_paths}
      track_id is relative path from frame_root.
    """
    root = Path(frame_root)
    if not root.exists():
        raise FileNotFoundError(f"VoxCeleb2 frame_root does not exist: {frame_root}")

    speaker_allow = set(speakers) if speakers else None
    grouped: Dict[str, List[Path]] = {}

    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in IMG_EXTS:
            continue

        rel_parent = file_path.parent.relative_to(root)
        parts = rel_parent.parts
        if len(parts) >= 2 and parts[0].lower() in SPLIT_NAMES and ID_RE.match(parts[1]):
            speaker = parts[1]
        else:
            speaker = parts[0] if parts else file_path.parent.name
        if speaker_allow is not None and speaker not in speaker_allow:
            continue

        track_id = str(rel_parent)
        grouped.setdefault(track_id, []).append(file_path)

    filtered: Dict[str, List[Path]] = {}
    for track_id, files in grouped.items():
        files_sorted = sorted(files, key=_natural_key)
        if len(files_sorted) >= min_frames:
            filtered[track_id] = files_sorted

    if not filtered:
        raise RuntimeError(
            "No VoxCeleb2 frame tracks found. Ensure frame_root contains image files in nested track directories."
        )

    track_ids = sorted(filtered.keys())
    if max_tracks > 0 and len(track_ids) > max_tracks:
        track_ids = track_ids[:max_tracks]

    return {tid: filtered[tid] for tid in track_ids}


class VoxCeleb2ClipDataset(Dataset):
    def __init__(
        self,
        frame_root: str,
        clip_len: int,
        image_size: int,
        clip_stride: int,
        speakers: Optional[Sequence[str]] = None,
        temporal_jitter: int = 0,
        augment: bool = True,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.2,
        max_tracks: int = 0,
        max_samples: int = 0,
    ):
        self.clip_len = clip_len
        self.image_size = image_size
        self.clip_stride = clip_stride
        self.temporal_jitter = temporal_jitter
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

        self.track_frames = index_voxceleb2_tracks(
            frame_root=frame_root,
            speakers=speakers,
            min_frames=clip_len,
            max_tracks=max_tracks,
        )
        self.track_ids = sorted(self.track_frames.keys())

        self.samples: List[Tuple[str, int]] = []
        for track_id in self.track_ids:
            n_frames = len(self.track_frames[track_id])
            max_start = n_frames - self.clip_len
            for start in range(0, max_start + 1, self.clip_stride):
                self.samples.append((track_id, start))

        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        if not self.samples:
            raise RuntimeError(
                "No valid VoxCeleb2 clips generated. Check clip_len/clip_stride and extracted frame counts."
            )

    def __len__(self):
        return len(self.samples)

    def _sample_start(self, track_id: str, base_start: int) -> int:
        if self.temporal_jitter <= 0:
            return base_start

        n_frames = len(self.track_frames[track_id])
        max_start = n_frames - self.clip_len
        jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
        return min(max(base_start + jitter, 0), max_start)

    def _load_clip(self, track_id: str, start: int) -> torch.Tensor:
        frame_paths = self.track_frames[track_id][start : start + self.clip_len]

        frames: List[torch.Tensor] = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            frames.append(_pil_to_tensor(img))

        return torch.stack(frames, dim=1)  # [C,T,H,W]

    def __getitem__(self, idx: int):
        track_id, base_start = self.samples[idx]
        start = self._sample_start(track_id, base_start)

        clip = self._load_clip(track_id, start)
        if self.augment:
            clip = _apply_clip_augment(clip, self.hflip_prob, self.color_jitter)

        return clip
