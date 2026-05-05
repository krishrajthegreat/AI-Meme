"""
media_loader.py — Meme Media Loader
=====================================
Handles loading static images and animated GIFs, frame cycling, and
aspect-ratio-aware resizing for the split-screen meme panel.

Exports:
    load_image(path)         → np.ndarray | None
    load_gif_frames(path)    → list[np.ndarray]
    get_current_frame(frames, idx) → (frame, next_idx)
    resize_meme(frame, h, w) → np.ndarray
    make_blank_frame(h, w, text) → np.ndarray
"""

from pathlib import Path
import cv2
import numpy as np


def make_blank_frame(height: int = 480, width: int = 640,
                     text: str = "No meme loaded") -> np.ndarray:
    """Black frame with centred grey text — used as fallback."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        sc, th = 0.7, 2
        (tw, th2), _ = cv2.getTextSize(text, font, sc, th)
        cv2.putText(frame, text, ((width - tw) // 2, (height + th2) // 2),
                    font, sc, (128, 128, 128), th)
    return frame


def load_image(path) -> np.ndarray | None:
    """Load a static image (.jpg/.jpeg/.png). Returns None on failure."""
    p = Path(path)
    if not p.exists():
        print(f"⚠  Image not found: {p}")
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠  Failed to decode: {p}")
    return img


def load_gif_frames(path) -> list[np.ndarray]:
    """Read all frames of an animated GIF into a list. Returns [] on failure."""
    p = Path(path)
    if not p.exists():
        print(f"⚠  GIF not found: {p}")
        return []
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print(f"⚠  Could not open GIF: {p}")
        return []
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if not frames:
        print(f"⚠  GIF has 0 frames: {p}")
    return frames


def get_current_frame(frames: list[np.ndarray],
                      frame_index: int) -> tuple[np.ndarray, int]:
    """Return (current_frame_copy, next_index) with auto-wrap."""
    if not frames:
        return make_blank_frame(), 0
    idx = frame_index % len(frames)
    return frames[idx].copy(), (idx + 1) % len(frames)


def resize_meme(frame: np.ndarray,
                target_height: int,
                target_width: int) -> np.ndarray:
    """Scale to target_height, then centre or crop to target_width."""
    if frame is None:
        return make_blank_frame(target_height, target_width)
    h, w = frame.shape[:2]
    if h == 0:
        return make_blank_frame(target_height, target_width)
    scale = target_height / h
    nw = int(w * scale)
    resized = cv2.resize(frame, (nw, target_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if nw >= target_width:
        xs = (nw - target_width) // 2
        canvas = resized[:, xs:xs + target_width]
    else:
        xo = (target_width - nw) // 2
        canvas[:, xo:xo + nw] = resized
    return canvas
