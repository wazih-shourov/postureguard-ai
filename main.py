"""
main.py — PostureGuard AI (Part 1)
Entry point: Webcam open + MediaPipe skeleton overlay
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys
import cv2
import time
import pygame
from posture_analyzer import PostureAnalyzer
from utils import draw_sci_fi_panel, draw_cyber_text, draw_hud_bar, get_posture_color


def main():
    print("=" * 50)
    print("  PostureGuard AI - Starting...")
    print("  Press Q to quit")
    print("=" * 50)

    # Camera initialize - try MSMF backend first (best on Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("[INFO] MSMF backend failed, trying default backend...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam! Please check your camera.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[OK] Webcam is running!")

    # ── Pygame Mixer initialize (For Audio Alarm) ──
    pygame.mixer.init()
    pygame.mixer.music.load(r"assets/beep warning sound.mp3")

    # ── Alarm state variables ──
    bad_posture_start_time = None
    ALARM_GRACE_PERIOD = 5.0  # 5 seconds grace period
    is_alarm_playing = False

    # Create PostureAnalyzer object
    analyzer = PostureAnalyzer()

    # ===========================
    # Main Loop - process each frame
    # ===========================
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Cannot read frame!")
            break

        # Mirror frame (for natural view)
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # ── Detect pose using MediaPipe
        results = analyzer.process_frame(frame)

        # ── Draw skeleton
        frame = analyzer.draw_skeleton(frame, results)

        # ── Extract key landmarks
        coords = analyzer.get_key_landmarks(results, w, h)

        # ── Calculate angles
        angles = analyzer.calculate_angles(coords)

        # ── Classify posture
        status = analyzer.classify_posture(angles)

        # ── Get status color
        color = get_posture_color(status)

        # ──────────────────────────────────────────
        # ALARM & GRACE PERIOD LOGIC
        # ──────────────────────────────────────────
        show_alert_flash = False
        
        if status == "BAD":
            if bad_posture_start_time is None:
                bad_posture_start_time = time.time()
            
            elapsed = time.time() - bad_posture_start_time
            
            if elapsed >= ALARM_GRACE_PERIOD:
                # Play alarm if not already playing
                if not is_alarm_playing:
                    pygame.mixer.music.play(-1)  # -1 means loop indefinitely
                    is_alarm_playing = True
                
                # Enable visual flashing
                show_alert_flash = True
            else:
                # Countdown visibility (No audio yet)
                remaining = ALARM_GRACE_PERIOD - elapsed
                msg = f"WARNING! CORRECT POSTURE IN {remaining:.1f}S"
                draw_cyber_text(frame, msg, (w//2 - 180, 50), font_scale=0.7, color=(0, 165, 255), thickness=2)
        else:
            # RESET everything if posture is GOOD
            bad_posture_start_time = None
            if is_alarm_playing:
                pygame.mixer.music.stop()
                is_alarm_playing = False

        # ──────────────────────────────────────────
        # OVERLAY — ULTRA HIGH END SCI-FI HUD (Jarvis Style)
        # ──────────────────────────────────────────
        
        # 1. Main Status Panel (Top Left)
        draw_sci_fi_panel(frame, 20, 20, 350, 70, border_color=color, alpha=0.3)
        status_text = f"SYS_STATE: {status}"
        draw_cyber_text(frame, status_text, (40, 60), font_scale=1.0, color=color, thickness=2)
        
        # 2. Project Watermark Box (Top Right)
        draw_sci_fi_panel(frame, w - 300, 20, 280, 70, border_color=(255, 100, 0), alpha=0.2)
        draw_cyber_text(frame, "PROJECT: POSTURE GUARD AI", (w - 285, 45), font_scale=0.55, color=(0, 255, 255))
        draw_cyber_text(frame, "VER: X-MARK-IV | ONLINE", (w - 285, 75), font_scale=0.5, color=(200, 200, 200))

        # 3. Biometric Analytics Panel (Mid Left)
        if angles:
            cal_pct = analyzer.get_calibration_status()

            draw_sci_fi_panel(frame, 20, 110, 310, 230, border_color=(0, 200, 255), alpha=0.25)
            draw_cyber_text(frame, "[BIOMETRIC ANALYTICS]", (40, 140), font_scale=0.6, color=(0, 255, 255), thickness=1)

            if cal_pct < 100:
                # ── Calibration Progress Overlay ──
                cal_label = f"CALIBRATING... {cal_pct}%"
                draw_cyber_text(frame, cal_label, (40, 175), font_scale=0.7, color=(0, 255, 255), thickness=2)
                draw_cyber_text(frame, "Sit straight for best results", (40, 205), font_scale=0.45, color=(180, 180, 180))
                # Progress bar
                bar_w = 270
                cv2.rectangle(frame, (40, 220), (40 + bar_w, 228), (40, 40, 50), -1)
                fill = int(bar_w * cal_pct / 100)
                cv2.rectangle(frame, (40, 220), (40 + fill, 228), (0, 255, 200), -1)
                cv2.circle(frame, (40 + fill, 224), 4, (255, 255, 255), -1)
            else:
                y_bar = 180
                # ── Delta-based colors (deviation from baseline) ──
                baseline = analyzer.baseline
                n_delta = max(0, angles['neck_angle']    - baseline['neck_angle'])
                s_delta = abs(angles['shoulder_tilt']   - baseline['shoulder_tilt'])
                
                if angles["back_angle"] is not None and baseline["back_angle"] != 0.0:
                    b_delta = max(0, angles['back_angle'] - baseline['back_angle'])
                else:
                    b_delta = 0.0

                fw_delta = max(0, angles['face_width']  - baseline['face_width'])
                hd_delta = max(0, baseline['head_drop'] - angles['head_drop'])

                n_color = (0, 0, 255) if n_delta > 8  else (0, 255, 0)
                s_color = (0, 0, 255) if s_delta > 10 else (0, 255, 0)
                b_color = (0, 0, 255) if b_delta > 6  else (0, 255, 0)
                fw_color = (0, 0, 255) if fw_delta > 15 else (0, 255, 0)
                hd_color = (0, 0, 255) if hd_delta > 15 else (0, 255, 0)

                draw_hud_bar(frame, 40, y_bar, 270, 45, round(s_delta, 1), s_color, "SHOULDER IMBALANCE")
                y_bar += 50
                draw_hud_bar(frame, 40, y_bar, 270, 45, round(n_delta, 1), n_color, "NECK DEVIATION")
                y_bar += 50
                draw_hud_bar(frame, 40, y_bar, 270, 45, round(b_delta, 1), b_color, "SPINE DEVIATION")
                y_bar += 50
                draw_hud_bar(frame, 40, y_bar, 270, 45, round(fw_delta, 1), fw_color, "FORWARD LEAN")
                y_bar += 50
                draw_hud_bar(frame, 40, y_bar, 270, 45, round(hd_delta, 1), hd_color, "HEAD SLOUCH")

        # 4. Global Target Frame (Screen Bounds)
        if show_alert_flash and int(time.time() * 6) % 2 == 0:
            # Flashing RED effect on the targeted frame & Text
            draw_sci_fi_panel(frame, 5, 5, w - 10, h - 10, bg_color=(0,0,20), border_color=(0,0,255), alpha=0.3)
            draw_cyber_text(frame, "POSTURE ALERT!", (w//2 - 220, h//2), font_scale=2.0, color=(0, 0, 255), thickness=4)
        else:
            draw_sci_fi_panel(frame, 5, 5, w - 10, h - 10, bg_color=(0,0,0), border_color=color, alpha=0.0)

        # Bottom Hints
        draw_cyber_text(frame, "[SYS_COMMAND] Press 'Q' to terminating system...", (20, h - 30), font_scale=0.5, color=(150, 150, 150))

        # ── Display in Window
        cv2.imshow("PostureGuard AI", frame)

        # ── Exit on pressing 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[EXIT] Closing PostureGuard AI...")
            break


    # Cleanup
    cap.release()
    pygame.mixer.quit()
    analyzer.release()
    cv2.destroyAllWindows()
    print("[DONE] Session complete!")


if __name__ == "__main__":
    main()
