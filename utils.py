"""
utils.py — PostureGuard AI
Small helper functions will be kept here
"""

import numpy as np
import cv2


def calculate_angle(a, b, c):
    """
    Calculates angle with three points (in degrees)
    a = first point (e.g. Ear)
    b = middle point / vertex (e.g. Shoulder)
    c = end point (e.g. Hip)
    """
    a = np.array(a)  # first point
    b = np.array(b)  # vertex / middle point
    c = np.array(c)  # last point

    # create vector
    ba = a - b
    bc = c - b

    # calculate cosine with dot product and magnitude
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # fix range

    # convert to degree
    angle = np.degrees(np.arccos(cosine_angle))
    return round(angle, 2)

def calculate_angle_with_vertical(p1, p2):
    """
    Calculates angle (degree) relative to vertical axis. Will be close to 0 degree if straight.
    p1 = top point (e.g. Ear)
    p2 = bottom point (e.g. Shoulder / Hip)
    """
    import math
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dy == 0:
        return 90.0
        
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return round(angle, 2)


def get_landmark_coords(landmarks, landmark_index, frame_width, frame_height):
    """
    Extracts pixel coordinates from MediaPipe landmarks
    """
    lm = landmarks[landmark_index]
    x = int(lm.x * frame_width)
    y = int(lm.y * frame_height)
    return (x, y)


def draw_sci_fi_panel(frame, x, y, w, h, bg_color=(10, 10, 15), border_color=(0, 255, 255), alpha=0.5):
    """
    Draws Sci-Fi / Jarvis style Translucent panel.
    Will have technological corner brackets around it.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Sci-Fi Corner Brackets ──
    t = 2          # thickness
    c_len = 15     # corner dash length
    
    # Top-Left
    cv2.line(frame, (x, y), (x + c_len, y), border_color, t)
    cv2.line(frame, (x, y), (x, y + c_len), border_color, t)
    # Top-Right
    cv2.line(frame, (x + w, y), (x + w - c_len, y), border_color, t)
    cv2.line(frame, (x + w, y), (x + w, y + c_len), border_color, t)
    # Bottom-Left
    cv2.line(frame, (x, y + h), (x + c_len, y + h), border_color, t)
    cv2.line(frame, (x, y + h), (x, y + h - c_len), border_color, t)
    # Bottom-Right
    cv2.line(frame, (x + w, y + h), (x + w - c_len, y + h), border_color, t)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - c_len), border_color, t)

    return frame

def draw_cyber_text(frame, text, position, font_scale=0.6, color=(255, 255, 255), thickness=1):
    """
    Anti-aliased crisp text
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_hud_bar(frame, x, y, width, max_val, current_val, color, title):
    """
    Digital energy/progress bar for angles
    """
    # Background Track
    cv2.rectangle(frame, (x, y), (x + width, y + 5), (40, 40, 50), -1)
    
    # Fill Value
    fill_ratio = max(0.0, min(1.0, current_val / max_val))
    fill_w = int(width * fill_ratio)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + 5), color, -1)
        # Glow edge
        cv2.circle(frame, (x + fill_w, y + 2), 3, (255, 255, 255), -1)

    # Title
    draw_cyber_text(frame, f"{title}: {current_val} DEG", (x, y - 8), font_scale=0.45, color=(200, 200, 200))


def get_posture_color(status):
    """
    Returns color according to Posture status
    GOOD  → Green
    WARNING → Yellow
    BAD   → Red
    """
    colors = {
        "GOOD":    (0, 255, 0),    # Green
        "BAD":     (0, 0, 255),    # Red
    }
    return colors.get(status, (255, 255, 255))
