"""
debug.py — Visual Debug Overlays
==================================
Toggle-able face mesh, hand skeleton, and blendshape readout.

Usage in main.py:
    from debug import DebugOverlay
    dbg = DebugOverlay()
    # in loop:
    dbg.draw(frame, face_lm, hands, bs)
    dbg.handle_key(key)   # key from cv2.waitKey
"""

import cv2
import numpy as np


class DebugOverlay:
    """Toggle-able debug visualisation layer (press 'd')."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._frame_count = 0

    def toggle(self):
        self.enabled = not self.enabled
        print(f"\n🔧 Debug overlay: {'ON' if self.enabled else 'OFF'}\n")

    def handle_key(self, key: int):
        """Pass the raw cv2.waitKey() return value. Toggles on 'd'."""
        if key == -1:          # ← ISSUE 5 fix: guard against no-key
            return
        if key & 0xFF == ord("d"):
            self.toggle()

    def draw(self, frame, face_landmarks=None, hand_landmarks_list=None,
             blendshapes=None):
        """Draw enabled overlays onto *frame* (in-place). No-op when off."""
        if not self.enabled:
            return
        if face_landmarks:
            self._draw_face(frame, face_landmarks)
        if hand_landmarks_list:
            self._draw_hands(frame, hand_landmarks_list)
        if blendshapes:
            self._print_blendshapes(blendshapes)
        h, w = frame.shape[:2]
        cv2.putText(frame, "DEBUG", (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ── Face Mesh (468 points) ───────────────────────────────────────

    @staticmethod
    def _draw_face(frame, face_landmarks):
        h, w = frame.shape[:2]
        for lm in face_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
        key_indices = {1: "nose", 13: "lip_u", 14: "lip_l", 61: "L_mth",
                       291: "R_mth", 152: "chin", 234: "L_ear", 454: "R_ear"}
        for idx in key_indices:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)
                cv2.putText(frame, str(idx), (px + 5, py - 5),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 255), 1)

    # ── Hand Skeleton ────────────────────────────────────────────────

    @staticmethod
    def _draw_hands(frame, hand_landmarks_list):
        h, w = frame.shape[:2]
        conns = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
        ]
        colors = [(255,50,50),(50,255,50),(50,50,255),(255,255,50)]
        for hi, hand in enumerate(hand_landmarks_list):
            c = colors[hi % len(colors)]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in conns:
                cv2.line(frame, pts[a], pts[b], c, 2)
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 4, c, -1)
                cv2.putText(frame, str(i), (pt[0]+4, pt[1]-4),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255), 1)

    # ── Blendshape Print (every 15 frames, score > 0.1) ─────────────

    def _print_blendshapes(self, blendshapes):
        self._frame_count += 1
        if self._frame_count % 15 != 0:
            return
        active = sorted(
            [(b.category_name, b.score) for b in blendshapes if b.score > 0.1],
            key=lambda x: x[1], reverse=True,
        )
        if not active:
            return
        print(f"── Blendshapes (#{self._frame_count}) ──")
        for name, score in active:
            print(f"  {name:<28s} {score:.3f}  {'█' * int(score * 20)}")
        print()
