"""
gestures.py — Confidence-Scored Gesture Engine
================================================
Returns float 0.0–1.0 per gesture.  GestureEngine handles rolling-window
smoothing, baseline normalisation, conflict resolution & priority order.

Usage:
    from gestures import GestureEngine
    engine = GestureEngine()
    # in loop:
    name, conf = engine.update(face_result, hand_result)
"""

import numpy as np
from collections import deque

# ── Helpers ──────────────────────────────────────────────────────────────

def bs_val(bs, name):
    if not bs:
        return 0.0
    if isinstance(bs, dict):
        return max(bs.get(name, 0.0), 0.0)
    for b in bs:
        if b.category_name == name:
            return b.score
    return 0.0

def _d(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def _c(v):
    return max(0.0, min(1.0, v))

def _ext(hand, tip, pip):
    """True if finger tip is above PIP (extended). y goes down."""
    return hand[tip].y < hand[pip].y

# ── Per-gesture thresholds ───────────────────────────────────────────────

THRESHOLDS = {
    "patrick": 0.55, "understandable": 0.60,
    "smirk": 0.60, "speed": 0.48,
    "thinking": 0.45, "shush": 0.65, "facepalm": 0.58,
    "self_point": 0.60, "raised_hand": 0.65, "hat": 0.58,
    "pointing": 0.58, "sad": 0.58,
    "shaq_t": 0.55, "jerry": 0.55,
    "selfpointing": 0.55,
}

# Conflict groups: within each group, only the highest-confidence wins
# Hard rules applied AFTER group resolution in GestureEngine:
#   - shaq_t hard override at 0.50 — immediately wins, skips all others
CONFLICT_GROUPS = [
    ["patrick", "pointing"],
    ["thinking", "shush", "facepalm"],
    ["shaq_t", "raised_hand", "hat"],
    ["jerry", "self_point", "selfpointing"],  # mutual exclusion for hand-shape gestures
]

# Priority order: shaq_t first with hard override, raised_hand near last
PRIORITY = [
    "shaq_t",           # 1. T-shape — hard override at 0.50
    "sad",              # 2. clasped hands low
    "patrick",          # 3. jaw wide open
    "pointing",         # 4. head tilt up
    "facepalm", "thinking", "shush",  # 5-7. hand near face
    "selfpointing",     # 8. index finger at camera (foreshortened)
    "jerry",            # 9. open spread hand — all fingers out
    "self_point",       # 10. finger pointing down at self
    "understandable",   # 11. peace/V sign
    "smirk",            # 12. asymmetric smile
    "speed",            # 13. squint + pucker
    "raised_hand",      # 14. open palm — last hand gesture
    "hat",              # 15. hand on head
]

# ═════════════════════════════════════════════════════════════════════════
# FACE-ONLY DETECTORS
# ═════════════════════════════════════════════════════════════════════════

# Smirk: asymmetric smile — one mouth corner significantly higher than
# the other. Must NOT be frowning or squinting.  [UNCHANGED]
def detect_smirk(bs):
    l = bs_val(bs, "mouthSmileLeft"); r = bs_val(bs, "mouthSmileRight")
    asym = abs(l - r); peak = max(l, r)
    if asym < 0.10 or peak < 0.15:
        return 0.0
    if max(bs_val(bs, "mouthFrownLeft"), bs_val(bs, "mouthFrownRight")) > 0.20:
        return 0.0
    return _c(asym / 0.35 * 0.6 + peak / 0.45 * 0.4)

# CHANGE 4 — Speed: lowered thresholds for consistent triggering.
# Temporal smoothing (2-frame hold at 0.40+) handled in GestureEngine.
def detect_speed(bs):
    sq_l = bs_val(bs, "eyeSquintLeft")
    sq_r = bs_val(bs, "eyeSquintRight")
    lips = max(bs_val(bs, "mouthPucker"), bs_val(bs, "mouthFunnel"))
    if sq_l < 0.20 or sq_r < 0.20:
        return 0.0
    if lips < 0.18:
        return 0.0
    if bs_val(bs, "jawOpen") > 0.25:
        return 0.0
    return _c(sq_l / 0.50 * 0.35 + sq_r / 0.50 * 0.35 + lips / 0.45 * 0.30)

# Patrick: jaw wide open — surprise/shock face with no hands visible.
# [UNCHANGED]
def detect_patrick(bs):
    jaw = bs_val(bs, "jawOpen")
    if jaw < 0.35:
        return 0.0
    if min(bs_val(bs, "eyeSquintLeft"), bs_val(bs, "eyeSquintRight")) > 0.35:
        return 0.0
    return _c((jaw - 0.30) / 0.40)

# Pointing: head tilted upward — face-only, NO hands required.
# [UNCHANGED]
def detect_pointing(face_lm, bs):
    """Head tilted upward — face-only, no hands required."""
    if not face_lm or not bs:
        return 0.0
    nose = face_lm[4]
    forehead = face_lm[10]
    chin = face_lm[152]
    if forehead.y <= 0.001:
        return 0.0
    ratio = nose.y / forehead.y
    if ratio > 0.90:
        return 0.0
    chin_gap = chin.y - nose.y
    if chin_gap < 0.08:
        return 0.0
    if bs_val(bs, "jawOpen") > 0.30:
        return 0.0
    ratio_score = _c((0.90 - ratio) / 0.15)
    gap_score = _c((chin_gap - 0.08) / 0.10)
    return _c(ratio_score * 0.6 + gap_score * 0.4)

# ═════════════════════════════════════════════════════════════════════════
# HAND + FACE DETECTORS  [thinking, shush, facepalm — UNCHANGED]
# ═════════════════════════════════════════════════════════════════════════

# CHANGE 1 — Thinking: stripped to minimum — only proximity + hand detected.
# Debug prints every sub-condition so we can see what blocks it.
def detect_thinking(face_lm, hands, bs):
    if not face_lm or not hands or len(hands) < 1:
        return 0.0
    PROX = 0.20
    targets = [face_lm[291], face_lm[61], face_lm[13]]  # R cheek, L cheek, nose
    for hand in hands:
        wrist = hand[0]
        tip8 = hand[8]
        tip12 = hand[12]
        # Compute all diagnostics
        d_291 = _d(tip8, face_lm[291])
        d_61  = _d(tip8, face_lm[61])
        d_13  = _d(tip8, face_lm[13])
        wrist_ok = 0.15 <= wrist.y <= 0.85
        idx_ext = tip8.y < hand[5].y   # tip above MCP (loose)
        mid_ext = tip12.y < hand[9].y
        best_d = min(d_291, d_61, d_13)
        prox_ok = best_d < PROX
        print(f"  [thinking] d291={d_291:.3f} d61={d_61:.3f} d13={d_13:.3f}"
              f" best={best_d:.3f} prox={'PASS' if prox_ok else 'FAIL'}"
              f" wrist_y={wrist.y:.3f} wrist={'PASS' if wrist_ok else 'FAIL'}"
              f" idx_ext={'PASS' if idx_ext else 'FAIL'} mid_ext={'PASS' if mid_ext else 'FAIL'}")
        # Only two hard requirements: proximity + hand exists
        if not prox_ok:
            continue
        # Score based on proximity alone
        conf = _c(1.0 - best_d / PROX)
        print(f"  [thinking MATCH] conf={conf:.2f}")
        return conf
    return 0.0

def detect_shush(face_lm, hands, bs):
    if not face_lm or not hands:
        return 0.0
    ears = [face_lm[234], face_lm[454]]
    for hand in hands:
        for ti in [4, 8, 12, 16, 20]:
            for ear in ears:
                d = _d(hand[ti], ear)
                if d < 0.10:
                    ext_cnt = sum(1 for t, p in [(8,6),(12,10),(16,14),(20,18)]
                                  if _ext(hand, t, p))
                    if ext_cnt < 3:
                        continue
                    return _c((1.0 - d / 0.10) * 0.7 + ext_cnt / 5 * 0.3)
    return 0.0

def detect_facepalm(face_lm, hands, bs):
    """Hand covering face — palm center near nose, fingers above wrist."""
    if not face_lm or not hands or len(hands) < 1:
        return 0.0
    nose = face_lm[4] if len(face_lm) > 4 else face_lm[1]
    for hand in hands:
        wrist = hand[0]
        cx = (hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5
        cy = (hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5
        if abs(cx - nose.x) > 0.18 or abs(cy - nose.y) > 0.18:
            continue
        up_cnt = sum(1 for idx in [8, 12, 16, 20] if hand[idx].y < wrist.y)
        if up_cnt < 3:
            continue
        dx = abs(cx - nose.x)
        dy = abs(cy - nose.y)
        prox = _c(1.0 - (dx + dy) / 0.36)
        cover = _c(up_cnt / 4)
        return _c(prox * 0.6 + cover * 0.4)
    return 0.0

# self_point: index finger pointing DOWNWARD toward body center.
# Thumb must be TUCKED (lm4→1m2 < 0.08). All 3 other fingers curled.
def detect_self_point(face_lm, hands, bs):
    """Index finger pointing down at own chest. Thumb tucked."""
    if not hands or len(hands) < 1:
        return 0.0
    for hand in hands:
        wrist = hand[0]
        tip = hand[8]
        pip = hand[6]
        thumb_tip = hand[4]
        thumb_base = hand[2]

        thumb_dist = _d(thumb_tip, thumb_base)
        mid_curled = hand[12].y > hand[9].y
        ring_curled = hand[16].y > hand[13].y
        pinky_curled = hand[20].y > hand[17].y
        curl_cnt = sum([mid_curled, ring_curled, pinky_curled])
        idx_down = tip.y > pip.y
        idx_centered = 0.25 <= tip.x <= 0.75

        if thumb_dist >= 0.08:
            continue
        if not idx_down:
            continue
        if curl_cnt < 3:
            continue
        if not idx_centered:
            continue
        if wrist.y < 0.40:
            continue

        down_score = _c((tip.y - pip.y) / 0.08)
        center_score = _c(1.0 - abs(tip.x - 0.5) / 0.25)
        curl_score = _c(curl_cnt / 3)
        conf = down_score * 0.40 + center_score * 0.25 + curl_score * 0.35
        return _c(conf)
    return 0.0


# selfpointing: index finger pointing AT CAMERA (foreshortened in 2D).
# When pointing at camera, index tip-to-MCP distance is unusually short.
# Other fingers curled, thumb extended.
def detect_selfpointing(hands):
    """Detects when the user points their index finger directly toward the camera.
    Uses 2D foreshortening: when pointing at the camera, the index finger appears
    unusually short in the 2D normalized landmark space (distance from tip to MCP < 0.10).
    Other fingers must be curled and thumb must be extended."""
    if not hands or len(hands) != 1:
        return 0.0
    hand = hands[0]
    index_tip = hand[8]
    index_mcp = hand[5]
    middle_tip = hand[12]; middle_mcp = hand[9]
    ring_tip = hand[16];   ring_mcp = hand[13]
    pinky_tip = hand[20];  pinky_mcp = hand[17]
    thumb_tip = hand[4];   thumb_ip = hand[3]

    # Index finger 2D length (foreshortened when pointing at camera)
    index_2d_len = np.sqrt((index_tip.x - index_mcp.x)**2 +
                           (index_tip.y - index_mcp.y)**2)
    # Other fingers curled (tips at or below MCP, with small tolerance)
    mid_curled   = middle_tip.y > middle_mcp.y - 0.02
    ring_curled  = ring_tip.y   > ring_mcp.y   - 0.02
    pinky_curled = pinky_tip.y  > pinky_mcp.y  - 0.02
    # Thumb raised (not tucked)
    thumb_up = thumb_tip.y < thumb_ip.y

    foreshort = index_2d_len < 0.10
    all_curled = mid_curled and ring_curled and pinky_curled

    print(f"  [selfpointing] idx_2d={index_2d_len:.3f} fore={'PASS' if foreshort else 'FAIL'}"
          f" curl={'PASS' if all_curled else 'FAIL'} thumb={'UP' if thumb_up else 'DN'}")

    if not (foreshort and all_curled and thumb_up):
        return 0.0

    # Confidence: how foreshortened (shorter = more confident)
    fore_score = _c((0.10 - index_2d_len) / 0.06)
    return _c(fore_score * 0.60 + 1.0 * 0.40)  # curl+thumb already passed as binary

# Hat: one hand placed ON TOP of head — wrist above forehead, palm
# center near the top of the head. Does NOT require all fingers extended
# (differentiator from raised_hand which needs 4+ fingers open).
def detect_hat(face_lm, hands, bs):
    """Hand resting on top of head — like wearing/holding a hat."""
    if not face_lm or not hands or len(hands) < 1:
        return 0.0
    forehead = face_lm[10]   # top of forehead
    nose = face_lm[4]
    for hand in hands:
        wrist = hand[0]
        # Palm center
        cx = (hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5
        cy = (hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5

        # Hand must be ABOVE the nose (near top of head)
        if cy > nose.y:
            continue

        # Palm center must be horizontally near the head (within 0.20 of forehead x)
        if abs(cx - forehead.x) > 0.20:
            continue

        # Palm center must be vertically near/above forehead (within 0.15)
        if cy > forehead.y + 0.05:
            continue

        # Score based on proximity to forehead
        dx = abs(cx - forehead.x)
        dy = abs(cy - forehead.y)
        prox_score = _c(1.0 - (dx + dy) / 0.30)
        above_score = _c((nose.y - cy) / 0.15)
        return _c(prox_score * 0.5 + above_score * 0.5)
    return 0.0

# CHANGE 2 — Raised hand: tightened to require high wrist, fully open palm,
# thumb spread, and fingers spread apart. Prevents stealing other gestures.
def detect_raised_hand(face_lm, hands, bs):
    if not face_lm or not hands or len(hands) < 1:
        return 0.0
    for hand in hands:
        wrist = hand[0]

        # Hand must be HIGH in frame: wrist y < 0.45 (upper half)
        if wrist.y > 0.45:
            continue

        # ALL 4 fingertips must be clearly extended: tip y < mcp y by >= 0.04
        fully_ext = True
        for tip_i, mcp_i in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            if hand[mcp_i].y - hand[tip_i].y < 0.04:
                fully_ext = False
                break
        if not fully_ext:
            continue

        # Thumb must be spread out: lm 4 x further from lm 17 x than lm 2 x
        thumb_spread = abs(hand[4].x - hand[17].x)
        thumb_base = abs(hand[2].x - hand[17].x)
        if thumb_spread <= thumb_base:
            continue

        # Palm must face forward: fingertip x-spread (lm 8 to lm 20) > 0.08
        tip_xs = [hand[i].x for i in (8, 12, 16, 20)]
        x_spread = max(tip_xs) - min(tip_xs)
        if x_spread < 0.08:
            continue

        # All fingertips above wrist
        tips = [hand[i] for i in (4, 8, 12, 16, 20)]
        if not all(wrist.y > t.y for t in tips):
            continue

        ext_cnt = sum(1 for t, p in [(8,6),(12,10),(16,14),(20,18)]
                      if _ext(hand, t, p))
        height_score = _c((0.45 - wrist.y) / 0.30)
        spread_score = _c(x_spread / 0.12)
        return _c(ext_cnt / 4 * 0.35 + height_score * 0.35 + spread_score * 0.30)
    return 0.0

# Understandable / Peace: V-sign.  [UNCHANGED]
def detect_understandable(hands):
    """Peace/V sign — index + middle extended in V, ring + pinky curled."""
    if not hands or len(hands) < 1:
        return 0.0
    for hand in hands:
        idx_tip = hand[8]; idx_pip = hand[6]
        mid_tip = hand[12]; mid_pip = hand[10]
        if not (idx_pip.y - idx_tip.y >= 0.04):
            continue
        if not (mid_pip.y - mid_tip.y >= 0.04):
            continue
        if _ext(hand, 16, 14):
            continue
        if _ext(hand, 20, 18):
            continue
        v_spread = abs(idx_tip.x - mid_tip.x)
        if v_spread < 0.03:
            continue
        ext_score = _c((idx_pip.y - idx_tip.y) / 0.08) * 0.5 + \
                    _c((mid_pip.y - mid_tip.y) / 0.08) * 0.5
        spread_score = _c(v_spread / 0.06)
        return _c(ext_score * 0.5 + spread_score * 0.5)
    return 0.0

# ═════════════════════════════════════════════════════════════════════════
# TWO-HAND DETECTORS
# ═════════════════════════════════════════════════════════════════════════

# CHANGE 1 — shaq_t: strict positioning with 5-point horizontal check,
# vertical hand with fingers-up, face-level requirement, debug prints.
# Hard override at 0.50 in GestureEngine — beats everything.
def detect_shaq_t(hands):
    """Timeout T — horizontal hand on top, vertical hand underneath."""
    if len(hands) != 2:
        return 0.0

    def hand_metrics(h):
        """Return (y_spread, x_spread) of lm 0,5,9,13,17 and tip-wrist gap."""
        pts = [h[0], h[5], h[9], h[13], h[17]]
        xs = [p.x for p in pts]; ys = [p.y for p in pts]
        y_sp = max(ys) - min(ys)
        x_sp = max(xs) - min(xs)
        avg_y = sum(ys) / len(ys)
        tip_gap = h[0].y - h[12].y  # positive = wrist below fingertip
        return y_sp, x_sp, avg_y, tip_gap

    m0 = hand_metrics(hands[0])
    m1 = hand_metrics(hands[1])
    # Debug: always print when 2 hands detected
    print(f"  [shaq_t debug] hand0: y_sp={m0[0]:.3f} x_sp={m0[1]:.3f} avg_y={m0[2]:.3f} tip_gap={m0[3]:.3f}"
          f" | hand1: y_sp={m1[0]:.3f} x_sp={m1[1]:.3f} avg_y={m1[2]:.3f} tip_gap={m1[3]:.3f}")

    # Try both assignments: hand0=horiz + hand1=vert, or vice versa
    for hi, vi in [(0, 1), (1, 0)]:
        h_ysp, h_xsp, h_avg_y, _ = hand_metrics(hands[hi])
        v_ysp, v_xsp, v_avg_y, v_gap = hand_metrics(hands[vi])

        # Horizontal: y-spread of lm 0,5,9,13,17 < 0.12, x-spread > 0.10
        if h_ysp >= 0.12 or h_xsp <= 0.10:
            continue

        # Vertical: wrist (lm 0) below middle fingertip (lm 12) by >= 0.08
        if v_gap < 0.08:
            continue

        # Horizontal hand avg_y must be ABOVE vertical hand's wrist y
        v_wrist_y = hands[vi][0].y
        if h_avg_y >= v_wrist_y:
            continue

        # Horizontal hand must be near face level: avg_y between 0.15 and 0.60
        if h_avg_y < 0.15 or h_avg_y > 0.60:
            continue

        horiz_score = _c(h_xsp / 0.16) * 0.5 + _c(1.0 - h_ysp / 0.12) * 0.5
        vert_score = _c(v_gap / 0.12)
        stack_score = _c((v_wrist_y - h_avg_y) / 0.15)
        conf = _c(horiz_score * 0.35 + vert_score * 0.35 + stack_score * 0.30)
        print(f"  [shaq_t MATCH] conf={conf:.2f} h_ysp={h_ysp:.3f} h_xsp={h_xsp:.3f} v_gap={v_gap:.3f}")
        return conf

    return 0.0

# Sad: both hands clasped together low in frame.  [UNCHANGED]
def detect_sad(face_lm, hands, bs):
    """Both hands clasped low in lap — wrists close and low, fingers curled."""
    if len(hands) != 2:
        return 0.0
    w0, w1 = hands[0][0], hands[1][0]
    if w0.y < 0.65 or w1.y < 0.65:
        return 0.0
    if abs(w0.x - w1.x) > 0.18:
        return 0.0
    if abs(w0.y - w1.y) > 0.15:
        return 0.0
    for hand in hands:
        curl_cnt = sum(1 for t, p in [(8,6),(12,10),(16,14),(20,18)]
                       if hand[t].y > hand[p].y)
        if curl_cnt < 3:
            return 0.0
    low_score = _c((min(w0.y, w1.y) - 0.60) / 0.20)
    close_score = _c(1.0 - max(abs(w0.x - w1.x), abs(w0.y - w1.y)) / 0.18)
    return _c(low_score * 0.5 + close_score * 0.5)

# Jerry: fully open spread hand — all 5 fingers extended outward,
# thumb spread wide laterally. Named after the Tom & Jerry character pose.
def detect_jerry(hands):
    """Detects the 'Jerry' gesture — a fully open, spread hand with all five fingers
    extended outward and the thumb spread wide laterally. All fingertips must be
    above their MCP bases and the overall hand spread (thumb tip to pinky tip
    distance) must exceed 0.30 in normalized coords."""
    if not hands or len(hands) != 1:
        return 0.0
    hand = hands[0]
    wrist = hand[0]
    thumb_tip = hand[4];  thumb_mcp = hand[2]
    index_tip = hand[8];  index_mcp = hand[5]
    middle_tip = hand[12]; middle_mcp = hand[9]
    ring_tip = hand[16];  ring_mcp = hand[13]
    pinky_tip = hand[20]; pinky_mcp = hand[17]

    # All 4 fingers extended (tip above MCP by >= 0.04)
    idx_ext = index_tip.y < index_mcp.y - 0.04
    mid_ext = middle_tip.y < middle_mcp.y - 0.04
    ring_ext = ring_tip.y < ring_mcp.y - 0.04
    pinky_ext = pinky_tip.y < pinky_mcp.y - 0.04
    all_ext = idx_ext and mid_ext and ring_ext and pinky_ext

    # Thumb fully extended outward
    thumb_ext_y = thumb_tip.y < thumb_mcp.y
    thumb_spread_x = abs(thumb_tip.x - wrist.x) > 0.12
    thumb_ok = thumb_ext_y and thumb_spread_x

    # Hand spread (thumb tip to pinky tip)
    hand_spread = np.sqrt((thumb_tip.x - pinky_tip.x)**2 +
                          (thumb_tip.y - pinky_tip.y)**2)
    spread_ok = hand_spread > 0.30

    print(f"  [jerry] ext={['N','N','N','N']}"
          f" idx={'E' if idx_ext else 'C'} mid={'E' if mid_ext else 'C'}"
          f" ring={'E' if ring_ext else 'C'} pinky={'E' if pinky_ext else 'C'}"
          f" thumb={'OK' if thumb_ok else 'NO'} spread={hand_spread:.3f}"
          f" spread_ok={'Y' if spread_ok else 'N'}")

    if not (all_ext and thumb_ok and spread_ok):
        return 0.0

    # Confidence scoring
    ext_cnt = sum([idx_ext, mid_ext, ring_ext, pinky_ext])
    ext_score = _c(ext_cnt / 4)
    spread_score = _c((hand_spread - 0.25) / 0.15)
    thumb_score = _c(abs(thumb_tip.x - wrist.x) / 0.18)
    conf = _c(ext_score * 0.35 + spread_score * 0.35 + thumb_score * 0.30)
    print(f"  [jerry MATCH] conf={conf:.2f} spread={hand_spread:.3f}")
    return conf

# ═════════════════════════════════════════════════════════════════════════
# NORMALIZER + ENGINE
# ═════════════════════════════════════════════════════════════════════════

class BlendshapeNormalizer:
    """Captures neutral baseline from first N frames, then subtracts it."""
    def __init__(self, n=30):
        self.n = n
        self.count = 0
        self.sums = {}
        self.baseline = {}
        self.calibrated = False

    def feed(self, bs_list):
        """Accept raw blendshape list, return normalised dict."""
        if not bs_list:
            return {}
        raw = {b.category_name: b.score for b in bs_list}
        if self.count < self.n:
            for k, v in raw.items():
                self.sums[k] = self.sums.get(k, 0.0) + v
            self.count += 1
            if self.count == self.n:
                self.baseline = {k: v / self.n for k, v in self.sums.items()}
                self.calibrated = True
            return raw  # use raw during calibration
        return {k: max(v - self.baseline.get(k, 0.0), 0.0) for k, v in raw.items()}


class GestureEngine:
    """
    Wraps all detection with rolling-window smoothing, baseline normalisation,
    conflict resolution and priority ordering.
    """
    def __init__(self, window_size=5):
        self.norm = BlendshapeNormalizer(n=30)
        self.windows = {g: deque(maxlen=window_size) for g in PRIORITY}
        self.calibrating = True
        self.speed_above_count = 0  # CHANGE 4: temporal smoothing counter

    def update(self, face_result, hand_result):
        """Run all detectors, smooth, resolve conflicts, return (name, conf)."""
        face_lm = face_result.face_landmarks[0] if face_result.face_landmarks else None
        raw_bs = face_result.face_blendshapes[0] if face_result.face_blendshapes else None
        hands = hand_result.hand_landmarks if hand_result.hand_landmarks else []
        n_hands = len(hands)

        # normalise blendshapes
        bs = self.norm.feed(raw_bs) if raw_bs else {}
        if self.calibrating and self.norm.calibrated:
            self.calibrating = False
            print("✓ Blendshape baseline calibrated (30 frames)\n")

        # ── compute raw confidence for each gesture ──────────────────
        scores = {}

        # two-hand gestures
        if n_hands == 2:
            scores["shaq_t"] = detect_shaq_t(hands)
            scores["sad"] = detect_sad(face_lm, hands, bs)

        # hand+face gestures
        if n_hands >= 1 and face_lm:
            scores["thinking"] = detect_thinking(face_lm, hands, bs)
            scores["shush"] = detect_shush(face_lm, hands, bs)
            scores["facepalm"] = detect_facepalm(face_lm, hands, bs)
            scores["raised_hand"] = detect_raised_hand(face_lm, hands, bs)
            scores["hat"] = detect_hat(face_lm, hands, bs)

        # hand-only gestures
        if n_hands >= 1:
            scores["self_point"] = detect_self_point(face_lm, hands, bs)
            scores["understandable"] = detect_understandable(hands)
            scores["jerry"] = detect_jerry(hands)
            scores["selfpointing"] = detect_selfpointing(hands)

        # face-only gestures
        if bs:
            scores["smirk"] = detect_smirk(bs)
            scores["speed"] = detect_speed(bs)
            if n_hands == 0:
                scores["patrick"] = detect_patrick(bs)

        # face-only (pointing = head tilt up)
        if face_lm and bs:
            scores["pointing"] = detect_pointing(face_lm, bs)

        # ── push into rolling windows ────────────────────────────────
        for g in PRIORITY:
            self.windows[g].append(scores.get(g, 0.0))

        # ── compute smoothed (average of window) ────────────────────
        smoothed = {}
        for g in PRIORITY:
            w = self.windows[g]
            smoothed[g] = sum(w) / len(w) if w else 0.0

        # ── conflict resolution ──────────────────────────────────────
        for group in CONFLICT_GROUPS:
            best = max(group, key=lambda g: smoothed.get(g, 0.0))
            for g in group:
                if g != best:
                    smoothed[g] = 0.0

        # ── CHANGE 1: shaq_t hard override — if >= 0.50 return immediately ──
        if smoothed.get("shaq_t", 0) >= 0.50:
            smoothed["raised_hand"] = 0.0
            return "shaq_t", smoothed["shaq_t"]

        # ── raised_hand suppression: if shaq_t has any score, kill raised_hand
        if smoothed.get("shaq_t", 0) > 0.0:
            smoothed["raised_hand"] = 0.0

        # ── speed temporal smoothing (2-frame hold) ─────────────────
        if smoothed.get("speed", 0) >= 0.40:
            self.speed_above_count += 1
        else:
            self.speed_above_count = 0
        if self.speed_above_count >= 2 and smoothed.get("speed", 0) < THRESHOLDS["speed"]:
            smoothed["speed"] = THRESHOLDS["speed"]

        # ── priority walk: first that meets its threshold wins ──────
        for g in PRIORITY:
            if smoothed[g] >= THRESHOLDS[g]:
                return g, smoothed[g]

        return "idle", 0.0
