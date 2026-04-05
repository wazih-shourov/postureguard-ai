"""
posture_analyzer.py — PostureGuard AI
Core logic for detecting Pose Landmarks using MediaPipe Tasks API (v0.10.30+)
and performing posture analysis is here
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from utils import calculate_angle, calculate_angle_with_vertical, get_posture_color

# ── Landmark index constants (MediaPipe 33 landmarks)
NOSE            = 0
LEFT_EAR        = 7
RIGHT_EAR       = 8
LEFT_SHOULDER   = 11
RIGHT_SHOULDER  = 12
LEFT_HIP        = 23
RIGHT_HIP       = 24
LEFT_KNEE       = 25
RIGHT_KNEE      = 26


class PostureAnalyzer:
    def __init__(self, model_path="assets/pose_landmarker_full.task"):
        # ── MediaPipe Tasks API setup
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.8,
            output_segmentation_masks=True
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

        # Connections for drawing skeleton
        self.POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS

        # ── Calibration System ──
        self._calibration_frames = []
        self._calibration_done   = False
        self._CALIBRATION_COUNT  = 40
        self.baseline            = None

        # ── Status Temporal Smoothing ──
        self._recent_statuses = []
        self._SMOOTH_WINDOW   = 10

        # ── Landmark EMA Smoothing ──
        #
        #   smoothed = alpha * raw + (1 - alpha) * smoothed_prev
        #
        # Lower alpha  → More smoothing (lag increases slightly)
        # Higher alpha → Closer to raw (less smoothing)
        # alpha = 0.25 → Follows body movement, but absorbs noise
        #
        # Dead-zone: Pixel movement smaller than this will be completely ignored
        # → Skeleton will not move at all if the user remains still
        self._EMA_ALPHA   = 0.25   # smoothing factor
        self._DEAD_ZONE   = 1.5    # pixels
        self._smoothed_lm = None   # numpy array (N, 4): [x_px, y_px, z, visibility]

    # ──────────────────────────────────────────────────────
    # Core frame processing
    # ──────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Takes BGR frame and processes it with MediaPipe.
        Returns PoseLandmarkerResult.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.frame_timestamp_ms += 33  # ~30 fps

        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        return result

    # ──────────────────────────────────────────────────────
    # EMA Landmark Smoothing
    # ──────────────────────────────────────────────────────

    def _update_ema(self, result, frame_w, frame_h):
        """
        Smooths MediaPipe raw landmarks using EMA and updates self._smoothed_lm.

        Dead-zone filter:
          If any landmark raw position moves less than DEAD_ZONE pixels from the previous
          smoothed position, that landmark is not updated.
          As a result, the skeleton remains completely frozen if the user is still.

        self._smoothed_lm: numpy array shape (N, 4)
            col-0 = x pixel
            col-1 = y pixel
            col-2 = z (normalized)
            col-3 = visibility
        """
        if not result.pose_landmarks:
            return  # No pose, buffer unchanged

        raw_lms = result.pose_landmarks[0]
        n = len(raw_lms)

        # Raw → numpy array
        raw_arr = np.array(
            [[lm.x * frame_w, lm.y * frame_h, lm.z, lm.visibility]
             for lm in raw_lms],
            dtype=np.float32,
        )  # shape (N, 4)

        if self._smoothed_lm is None or self._smoothed_lm.shape[0] != n:
            # First frame or landmark count changed — initialize with raw
            self._smoothed_lm = raw_arr.copy()
            return

        # ── EMA ──
        ema = self._EMA_ALPHA * raw_arr + (1.0 - self._EMA_ALPHA) * self._smoothed_lm

        # ── Dead-Zone Filter (X, Y only) ──
        delta_xy = np.abs(raw_arr[:, :2] - self._smoothed_lm[:, :2])  # (N, 2)
        frozen   = np.all(delta_xy < self._DEAD_ZONE, axis=1)          # (N,) bool mask

        # keep x, y of frozen landmarks to their previous smoothed value
        ema[frozen, :2] = self._smoothed_lm[frozen, :2]

        self._smoothed_lm = ema

    # ──────────────────────────────────────────────────────
    # Skeleton Drawing  (smoothed landmarks ব্যবহার করে)
    # ──────────────────────────────────────────────────────

    def draw_skeleton(self, frame, result):
        """
        Draws Sci-Fi skeleton with EMA-smoothed + dead-zone filtered landmarks.
        Skeleton won't move at all when standing still.
        When body moves, skeleton will accurately follow.
        """
        h, w = frame.shape[:2]

        # update smoothed buffer every frame
        self._update_ema(result, w, h)

        if self._smoothed_lm is None:
            return frame

        slm = self._smoothed_lm  # (N, 4)

        # ── Connections (Bones) ──
        for connection in self.POSE_CONNECTIONS:
            s_idx = connection.start
            e_idx = connection.end

            if s_idx < len(slm) and e_idx < len(slm):
                if slm[s_idx, 3] > 0.5 and slm[e_idx, 3] > 0.5:
                    sx, sy = int(slm[s_idx, 0]), int(slm[s_idx, 1])
                    ex, ey = int(slm[e_idx, 0]), int(slm[e_idx, 1])

                    z_avg     = (slm[s_idx, 2] + slm[e_idx, 2]) / 2
                    thickness = int(max(2, 5 + (-z_avg * 10)))

                    cv2.line(frame, (sx, sy), (ex, ey), (255, 100, 0), thickness)

        # ── Joints ──
        for i in range(len(slm)):
            if slm[i, 3] > 0.5:
                cx, cy = int(slm[i, 0]), int(slm[i, 1])
                z      = slm[i, 2]
                radius = int(max(3, 8 + (-z * 15)))

                cv2.circle(frame, (cx, cy), radius + 4, (255, 200, 0), 1)   # outer ring
                cv2.circle(frame, (cx, cy), radius,     (0, 255, 255), -1)  # inner core

        return frame

    # ──────────────────────────────────────────────────────
    # Key landmark extraction  (smoothed)
    # ──────────────────────────────────────────────────────

    def get_key_landmarks(self, result, frame_w, frame_h):
        """
        Extracts key landmark pixel coordinates from the smoothed landmark buffer.
        Does not use raw MediaPipe output — makes angle calculation smooth too.
        """
        if self._smoothed_lm is None:
            return None

        slm = self._smoothed_lm  # (N, 4)

        needed = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER]

        # Return None if any necessary (upper body) landmark is invisible
        if any(idx >= len(slm) or slm[idx, 3] < 0.3 for idx in needed):
            return None

        # Hip optional (legs/waist might not be visible in webcam)
        has_hip = False
        if RIGHT_HIP < len(slm) and slm[RIGHT_HIP, 3] >= 0.3:
            has_hip = True

        def px(idx):
            return (int(slm[idx, 0]), int(slm[idx, 1]))

        return {
            "NOSE":           px(NOSE),
            "LEFT_EAR":       px(LEFT_EAR),
            "RIGHT_EAR":      px(RIGHT_EAR),
            "LEFT_SHOULDER":  px(LEFT_SHOULDER),
            "RIGHT_SHOULDER": px(RIGHT_SHOULDER),
            "LEFT_HIP":       px(LEFT_HIP),
            "RIGHT_HIP":      px(RIGHT_HIP),
            "LEFT_KNEE":      px(LEFT_KNEE),
            "RIGHT_KNEE":     px(RIGHT_KNEE),
            "HAS_HIP":        has_hip,
        }

    # ──────────────────────────────────────────────────────
    # Angle Calculation
    # ──────────────────────────────────────────────────────

    def calculate_angles(self, coords):
        """
        1. Neck Inclination  → Vertical vs (Ear to Shoulder)
        2. Shoulder Tilt     → Left vs Right Shoulder height difference (pixels)
        3. Back Inclination  → Vertical vs (Shoulder to Hip)
        """
        if coords is None:
            return None

        try:
            neck_angle = calculate_angle_with_vertical(
                coords["RIGHT_EAR"],
                coords["RIGHT_SHOULDER"]
            )

            shoulder_tilt = abs(
                coords["LEFT_SHOULDER"][1] - coords["RIGHT_SHOULDER"][1]
            )

            if coords.get("HAS_HIP", False):
                back_angle = calculate_angle_with_vertical(
                    coords["RIGHT_SHOULDER"],
                    coords["RIGHT_HIP"]
                )
            else:
                back_angle = None

            # ── 4. Head Drop (Vertical distance: Nose vs Shoulders) ──
            mid_shoulder_y = (coords["LEFT_SHOULDER"][1] + coords["RIGHT_SHOULDER"][1]) / 2.0
            head_drop = mid_shoulder_y - coords["NOSE"][1]

            # ── 5. Face Width (Forward Lean / Z-Proximity) ──
            face_width = abs(coords["LEFT_EAR"][0] - coords["RIGHT_EAR"][0])

            return {
                "neck_angle":    neck_angle,
                "shoulder_tilt": shoulder_tilt,
                "back_angle":    back_angle,
                "head_drop":     head_drop,
                "face_width":    face_width,
            }

        except Exception as e:
            print(f"[Angle Error] {e}")
            return None

    # ──────────────────────────────────────────────────────
    # Posture Classification  (calibration + delta-based)
    # ──────────────────────────────────────────────────────

    def classify_posture(self, angles):
        """
        Calibration-based posture classification:

        Phase 1 — Calibration (first N frames):
          Learns User's personal posture baseline (median of raw angles).

        Phase 2 — Delta-based scoring:
          Not scored by raw angle, but by deviation (delta) from baseline.
          This provides accurate results even on side-view cameras.

        Weighted score:
          score = neck_delta * 1.5 + back_delta * 2.0 + tilt_delta * 0.5

        Thresholds:
          score < 15   → GOOD
          score >= 15  → BAD

        Temporal smoothing:
          Final status is determined by majority vote of last 10 frames.
        """
        if angles is None:
            return "UNKNOWN"

        # ── Phase 1: Calibration ──
        if not self._calibration_done:
            self._calibration_frames.append(angles)
            if len(self._calibration_frames) >= self._CALIBRATION_COUNT:
                import statistics
                valid_back = [f["back_angle"] for f in self._calibration_frames if f["back_angle"] is not None]
                b_baseline = statistics.median(valid_back) if valid_back else 0.0

                self.baseline = {
                    "neck_angle":    statistics.median(f["neck_angle"]    for f in self._calibration_frames),
                    "shoulder_tilt": statistics.median(f["shoulder_tilt"] for f in self._calibration_frames),
                    "back_angle":    b_baseline,
                    "head_drop":     statistics.median(f["head_drop"]     for f in self._calibration_frames),
                    "face_width":    statistics.median(f["face_width"]    for f in self._calibration_frames),
                }
                self._calibration_done = True
                print(f"[CALIBRATION DONE] Baseline: {self.baseline}")
            return "GOOD"  # Show GOOD during Calibration

        # ── Phase 2: Delta-based Classification ──
        # Side-view metrics
        baseline = self.baseline
        neck_delta = max(0.0, angles['neck_angle'] - baseline['neck_angle'])
        tilt_delta = abs(angles['shoulder_tilt'] - baseline['shoulder_tilt'])
        
        if angles["back_angle"] is not None and baseline["back_angle"] != 0.0:
            back_delta = max(0.0, angles['back_angle'] - baseline['back_angle'])
        else:
            back_delta = 0.0

        # Front-view metrics
        # Leaning forward = face width increases
        fw_delta = max(0.0, angles["face_width"] - self.baseline["face_width"])
        # Slouching down = head drop distance decreases
        hd_delta = max(0.0, self.baseline["head_drop"] - angles["head_drop"])

        # Weighted score combination (works seamlessly for both frontal and side views)
        side_score   = (neck_delta * 1.5) + (back_delta * 2.0) + (tilt_delta * 0.5)
        front_score  = (fw_delta * 0.8)   + (hd_delta * 0.8)
        
        score = side_score + front_score

        if score < 15:
            raw_status = "GOOD"
        else:
            raw_status = "BAD"

        # ── Temporal Smoothing (Majority Vote) ──
        self._recent_statuses.append(raw_status)
        if len(self._recent_statuses) > self._SMOOTH_WINDOW:
            self._recent_statuses.pop(0)

        final_status = max(set(self._recent_statuses), key=self._recent_statuses.count)
        return final_status

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    def get_calibration_status(self):
        """Returns Calibration progress percentage (0-100)"""
        if self._calibration_done:
            return 100
        return int(len(self._calibration_frames) / self._CALIBRATION_COUNT * 100)

    def release(self):
        """Release resources"""
        self.landmarker.close()
