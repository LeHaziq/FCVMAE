import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"}
SPLIT_NAMES = {"dev", "test", "train", "val", "validation"}
ID_RE = re.compile(r"^id\d+")


@dataclass
class Task:
    video_path: Path
    out_dir: Path


@dataclass
class Result:
    video_path: str
    out_dir: str
    status: str
    frame_count: int
    message: str = ""


def parse_args():
    p = argparse.ArgumentParser(description="Extract VoxCeleb2 raw videos into frame folders.")
    p.add_argument("--input-root", type=str, required=True, help="Root containing raw VoxCeleb2 videos.")
    p.add_argument("--output-root", type=str, required=True, help="Root to write extracted frames.")
    p.add_argument("--fps", type=float, default=25.0, help="Target FPS. Set <=0 to keep source FPS.")
    p.add_argument("--resize-width", type=int, default=0, help="Optional output width (0 disables resize).")
    p.add_argument("--resize-height", type=int, default=0, help="Optional output height (0 disables resize).")
    p.add_argument("--jpeg-q", type=int, default=2, help="FFmpeg JPEG quality 2(best)-31(worst).")
    p.add_argument("--frame-pattern", type=str, default="frame_%06d.jpg", help="Output frame filename pattern.")
    p.add_argument("--workers", type=int, default=8, help="Parallel ffmpeg workers.")
    p.add_argument("--max-videos", type=int, default=0, help="Optional cap for number of videos.")
    p.add_argument("--speakers", type=str, default="", help="Comma-separated speaker ids to include (e.g., id00012,id00017).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames.")
    p.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip videos whose output dir already has image frames (default: true).",
    )
    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-extract even if output dir already has frames.",
    )
    p.add_argument("--dry-run", action="store_true", help="List planned tasks without extracting.")
    p.add_argument("--summary-json", type=str, default="", help="Optional path to write extraction summary JSON.")
    return p.parse_args()


def _speaker_from_rel_parts(parts: Sequence[str]) -> str:
    if not parts:
        return ""
    if len(parts) >= 2 and parts[0].lower() in SPLIT_NAMES and ID_RE.match(parts[1]):
        return parts[1]
    return parts[0]


def discover_videos(input_root: Path, speakers: Optional[Sequence[str]], max_videos: int) -> List[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")

    speaker_set = set(speakers) if speakers else None
    videos: List[Path] = []

    for p in input_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = p.relative_to(input_root)
        speaker = _speaker_from_rel_parts(rel.parts)
        if speaker_set is not None and speaker not in speaker_set:
            continue
        videos.append(p)

    videos = sorted(videos)
    if max_videos > 0:
        videos = videos[:max_videos]
    return videos


def has_extracted_frames(out_dir: Path) -> bool:
    if not out_dir.exists() or not out_dir.is_dir():
        return False
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        if any(out_dir.glob(ext)):
            return True
    return False


def count_extracted_frames(out_dir: Path) -> int:
    total = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        total += sum(1 for _ in out_dir.glob(ext))
    return total


def build_tasks(videos: Sequence[Path], input_root: Path, output_root: Path) -> List[Task]:
    tasks: List[Task] = []
    for video in videos:
        rel = video.relative_to(input_root)
        out_dir = output_root / rel.with_suffix("")
        tasks.append(Task(video_path=video, out_dir=out_dir))
    return tasks


def _make_filter_chain(fps: float, resize_width: int, resize_height: int) -> str:
    filters: List[str] = []
    if fps and fps > 0:
        filters.append(f"fps={fps:g}")
    if resize_width > 0 and resize_height > 0:
        filters.append(f"scale={resize_width}:{resize_height}")
    return ",".join(filters)


def _clear_existing_frames(out_dir: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in out_dir.glob(ext):
            p.unlink()


def run_one(task: Task, args) -> Result:
    out_dir = task.out_dir

    if args.skip_existing and not args.overwrite and has_extracted_frames(out_dir):
        n = count_extracted_frames(out_dir)
        return Result(str(task.video_path), str(out_dir), "skipped", n, "frames already exist")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        _clear_existing_frames(out_dir)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    cmd.append("-y" if args.overwrite else "-n")
    cmd.extend(["-i", str(task.video_path)])

    vf = _make_filter_chain(args.fps, args.resize_width, args.resize_height)
    if vf:
        cmd.extend(["-vf", vf])

    cmd.extend(["-qscale:v", str(args.jpeg_q), str(out_dir / args.frame_pattern)])

    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return Result(str(task.video_path), str(out_dir), "error", 0, f"failed to launch ffmpeg: {exc}")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
        return Result(str(task.video_path), str(out_dir), "error", 0, msg)

    frame_count = count_extracted_frames(out_dir)
    return Result(str(task.video_path), str(out_dir), "ok", frame_count)


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg before running extraction.")


def main():
    args = parse_args()
    ensure_ffmpeg()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()] if args.speakers else None

    videos = discover_videos(
        input_root=input_root,
        speakers=speakers,
        max_videos=args.max_videos,
    )
    if not videos:
        raise RuntimeError("No videos found with supported extensions under input_root.")

    tasks = build_tasks(videos, input_root, output_root)

    print(f"discovered_videos={len(videos)}")
    print(f"output_root={output_root}")

    if args.dry_run:
        for t in tasks[:20]:
            print(f"DRY_RUN {t.video_path} -> {t.out_dir}")
        if len(tasks) > 20:
            print(f"... ({len(tasks) - 20} more)")
        return

    results: List[Result] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut_map = {ex.submit(run_one, task, args): task for task in tasks}
        total = len(fut_map)
        for i, fut in enumerate(as_completed(fut_map), start=1):
            res = fut.result()
            results.append(res)
            if i % 25 == 0 or i == total:
                ok = sum(1 for r in results if r.status == "ok")
                skipped = sum(1 for r in results if r.status == "skipped")
                err = sum(1 for r in results if r.status == "error")
                print(f"progress={i}/{total} ok={ok} skipped={skipped} error={err}")

    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    err = [r for r in results if r.status == "error"]

    total_frames = sum(r.frame_count for r in ok)
    print(f"done ok={len(ok)} skipped={len(skipped)} error={len(err)} extracted_frames={total_frames}")

    if err:
        print("first_errors:")
        for r in err[:10]:
            print(f"- {r.video_path}: {r.message}")

    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "counts": {
                "ok": len(ok),
                "skipped": len(skipped),
                "error": len(err),
                "extracted_frames": total_frames,
            },
            "results": [asdict(r) for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"summary_json={out_path}")


if __name__ == "__main__":
    main()
