"""
AI Meme Emote Detector — main.py
=================================
Real-time gesture → meme overlay using MediaPipe Tasks API.

Imports from:
    download_models  — model auto-download + paths
    gestures         — all 16 detect_* functions
    media_loader     — image/GIF loading, resizing, frame cycling
    debug            — toggleable visual overlays (press 'd')
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ── Project module imports ───────────────────────────────────────────────

from download_models import ensure_models, FACE_MODEL, HAND_MODEL

from gestures import GestureEngine

from media_loader import (
    load_image, load_gif_frames, get_current_frame,
    resize_meme, make_blank_frame,
)

from debug import DebugOverlay

# ── Gesture → Meme Map ──────────────────────────────────────────────────

# Required new meme files in ./images/:
#   selfpointing-meme.jpg  — e.g. the "I'm watching you" or "pointing at you" meme
#   jerry-meme.jpg         — e.g. Jerry from Tom & Jerry with open arms/spread pose
# (Currently mapped to existing files as fallback)
GESTURE_MEME_MAP = {
    "smirk":           "tsundere.gif",
    "speed":           "speed.jpg",
    "patrick":         "patrick.jpeg",
    "thinking":        "monkey.jpeg",
    "shush":           "mourinho.jpeg",
    "facepalm":        "jonah.jpg",
    "self_point":      "guy-pointing-at-himself.jpg",
    "selfpointing":    "jerry.gif",
    "shaq_t":          "shaq.jpg",
    "sad":             "black.gif",
    "raised_hand":     "pique.jpeg",
    "pointing":        "black.gif",
    "understandable":  "understandable.jpg",
    "jerry":           "jerry.gif",
    "hat":             "son.jpg",
}

GESTURE_HUD = {
    "shaq_t":          ("SHAQ T!",           (128, 0, 128)),
    "jerry":           ("Jerry \U0001f42d",  (0, 200, 255)),
    "patrick":         ("PATRICK!",          (128, 128, 255)),
    "thinking":        ("Thinking...",       (255, 200, 0)),
    "shush":           ("SHUSH!",            (255, 255, 255)),
    "facepalm":        ("FACEPALM",          (100, 100, 255)),
    "self_point":      ("ME!",               (0, 200, 200)),
    "selfpointing":    ("Self Pointing \U0001f449", (0, 220, 180)),
    "raised_hand":     ("RAISED HAND",       (0, 255, 200)),
    "pointing":        ("LOOKING UP",        (255, 128, 0)),
    "understandable":  ("PEACE",             (100, 255, 100)),
    "smirk":           ("Smirking",          (0, 165, 255)),
    "sad":             ("CLASPED",           (255, 100, 100)),
    "speed":           ("SPEED!",            (0, 100, 255)),
    "hat":             ("HAT",               (255, 180, 0)),
}

GESTURE_HOLD_DELAY = 1.0      # gesture must be held this long before meme switches
MEME_MIN_DISPLAY = 3.0        # once shown, a meme stays at least this long
IMAGES_DIR = "./images"


# ── MemePlayer (delegates to media_loader) ───────────────────────────────

class MemePlayer:
    """Loads and serves meme images / animated GIFs frame-by-frame."""

    def __init__(self, images_folder: str = IMAGES_DIR):
        from pathlib import Path
        self.folder = Path(images_folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.gesture = None
        self.media_type = None        # "image" | "gif" | None
        self.image = None             # static image
        self.gif_frames = []          # pre-loaded GIF frames
        self.gif_idx = 0
        self.th, self.tw = 480, 640
        self.blank = make_blank_frame(self.th, self.tw, "")

    def set_dims(self, h: int, w: int):
        self.th, self.tw = h, w
        self.blank = make_blank_frame(h, w, "")

    def load(self, gesture: str) -> bool:
        """Load meme for *gesture*. Returns False on failure."""
        if gesture == self.gesture:
            return True
        self.image = None
        self.gif_frames = []
        self.gif_idx = 0
        self.gesture = gesture

        if gesture is None or gesture not in GESTURE_MEME_MAP:
            self.media_type = None
            return False

        fp = self.folder / GESTURE_MEME_MAP[gesture]
        if not fp.exists():
            print(f"⚠  Missing meme: {fp.name}")
            self.media_type = None
            return False

        if fp.suffix.lower() == ".gif":
            self.gif_frames = load_gif_frames(fp)
            if not self.gif_frames:
                self.media_type = None
                return False
            self.media_type = "gif"
        else:
            self.image = load_image(fp)
            if self.image is None:
                self.media_type = None
                return False
            self.media_type = "image"
        return True

    def frame(self) -> np.ndarray:
        """Return the next display-ready frame (sized to th×tw)."""
        raw = None
        if self.media_type == "gif" and self.gif_frames:
            raw, self.gif_idx = get_current_frame(self.gif_frames, self.gif_idx)
        elif self.media_type == "image" and self.image is not None:
            raw = self.image.copy()
        if raw is None:
            return self.blank.copy()
        return resize_meme(raw, self.th, self.tw)

    def release(self):
        self.gif_frames = []
        self.image = None


# ── MediaPipe Init (Tasks API only) ─────────────────────────────────────

def create_face_landmarker() -> vision.FaceLandmarker:
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
    )


def create_hand_landmarker() -> vision.HandLandmarker:
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )


# ── HUD Drawing ─────────────────────────────────────────────────────────

def draw_labels(frame, labels):
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 40
    for text, color in labels:
        (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
        cv2.rectangle(frame, (5, y - th - 5), (15 + tw, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, y), font, 0.9, color, 2)
        y += 45


# ── Main Loop ───────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🎭  AI Meme Emote Detector")
    print("=" * 55)

    # 1. Models
    try:
        ensure_models()
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        print("   Run  python download_models.py  manually.")
        return

    # 2. Detectors
    print("\n🔧 Initialising detectors…")
    face_det = create_face_landmarker()
    hand_det = create_hand_landmarker()
    print("✓ Ready.\n")

    # 3. Meme player
    player = MemePlayer(IMAGES_DIR)

    # 4. Webcam — try camera 0 then 1   (ISSUE 4 fix)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠  Camera 0 unavailable, trying camera 1…")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ No webcam found. Connect a camera and try again.")
        return

    # 5. Test frame for dimensions      (ISSUE 4 fix: else branch)
    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        player.set_dims(h, w)
        print(f"📷 Webcam: {w}×{h}")
    else:
        print("⚠  Test frame failed — using 640×480 default")
        player.set_dims(480, 640)

    # 6. Debug overlay + gesture engine (no idle preload — start blank)
    dbg = DebugOverlay()
    engine = GestureEngine(window_size=5)
    print("\u25b6 Running \u2014 press 'q' to quit, 'd' to toggle debug.")
    print("  (Hold still for 1 sec \u2014 calibrating blendshape baseline\u2026)\n")

    # State — no idle fallback; just hold last meme
    cur_gesture = None          # nothing loaded yet
    pend_gesture = None
    pend_start_t = 0.0
    meme_shown_t = time.time()
    last_active_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠  Frame read failed — retrying…")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_r = face_det.detect(mp_img)
        hand_r = hand_det.detect(mp_img)

        face_lm = face_r.face_landmarks[0] if face_r.face_landmarks else None
        bs = face_r.face_blendshapes[0] if face_r.face_blendshapes else None
        hands = hand_r.hand_landmarks if hand_r.hand_landmarks else []

        # ── Gesture detection (engine handles smoothing + conflicts) ──
        active, conf = engine.update(face_r, hand_r)

        labels = []
        if active != "idle" and active in GESTURE_HUD:
            label, color = GESTURE_HUD[active]
            labels.append((f"{label} {conf:.0%}", color))

        if labels:
            draw_labels(frame, labels)

        # Debug overlay
        dbg.draw(frame, face_lm, hands, bs)

        # ── Track last real gesture for debounce ────────────────────
        now = time.time()

        if active != "idle":
            last_active_t = now

        # ── Gesture switching with debounce ───────────────────────────
        # Only switch meme for REAL gestures (not idle)
        # If engine returns idle, keep showing whatever was last displayed
        if active != "idle" and active != cur_gesture:
            if pend_gesture != active:
                pend_gesture = active
                pend_start_t = now
        elif active == cur_gesture:
            pend_gesture = None
        # If active == "idle", do nothing — hold last meme

        if pend_gesture is not None and pend_gesture != cur_gesture:
            held_long_enough = (now - pend_start_t) >= GESTURE_HOLD_DELAY
            meme_displayed_long_enough = cur_gesture is None or (now - meme_shown_t) >= MEME_MIN_DISPLAY
            if held_long_enough and meme_displayed_long_enough:
                player.load(pend_gesture)
                cur_gesture = pend_gesture
                meme_shown_t = now
                pend_gesture = None

        # ── Display ──────────────────────────────────────────────────
        meme = player.frame()
        cv2.putText(frame, "Press 'q' quit | 'd' debug",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Meme Mirror", np.hstack((frame, meme)))

        # ── Key handling (ISSUE 5 fix: guard key == -1) ──────────────
        key = cv2.waitKey(1)
        if key != -1:
            dbg.handle_key(key)
            if key & 0xFF == ord("q"):
                break

    # ── Cleanup ──────────────────────────────────────────────────────
    player.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
