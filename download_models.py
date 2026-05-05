"""
download_models.py — MediaPipe Model Downloader
================================================
Downloads FaceLandmarker and HandLandmarker .task files into ./models/.
Can be run standalone or imported by main.py.

Exports:
    FACE_MODEL  — Path to face_landmarker.task
    HAND_MODEL  — Path to hand_landmarker.task
    ensure_models() — download both if missing
"""

import urllib.request
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"

_URLS = {
    FACE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    HAND_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def _progress_hook(filename: str):
    def hook(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = min(done / total_size * 100, 100)
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            sys.stdout.write(
                f"\r   [{bar}] {pct:5.1f}%  "
                f"{done / 1_048_576:.1f}/{total_size / 1_048_576:.1f} MB"
            )
            sys.stdout.flush()
            if pct >= 100:
                print()
        else:
            sys.stdout.write(f"\r   {done / 1_048_576:.1f} MB …")
            sys.stdout.flush()
    return hook


def _download_one(path: Path, url: str) -> None:
    if path.exists():
        print(f"  ✓ {path.name} ready ({path.stat().st_size / 1_048_576:.1f} MB)")
        return
    print(f"  ⬇ Downloading {path.name}…")
    try:
        urllib.request.urlretrieve(url, path, reporthook=_progress_hook(path.name))
        print(f"  ✓ Saved {path.name}")
    except Exception as exc:
        if path.exists():
            path.unlink()
        print(f"\n  ✗ Failed: {exc}")
        print(f"    Manual URL: {url}")
        raise


def ensure_models() -> None:
    """Download both models if missing. Safe to call multiple times."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in _URLS.items():
        _download_one(path, url)


def main() -> None:
    print("\n" + "=" * 55)
    print("  MediaPipe Model Downloader")
    print("=" * 55 + "\n")
    try:
        ensure_models()
        print("\n  🎉 All models ready!\n")
    except Exception:
        print("\n  ⚠ Some models failed. Place them manually in ./models/\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
