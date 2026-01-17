import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from vision.face_hand_detection import detect_face_hand
from vision.shadow_segmentation import segment_shadow_with_hand
from vision.light_estimation import estimate_light_direction
from visualization.overlay import draw_overlay
from visualization.matrix_plot import (
    create_shadow_intensity_matrix,
    create_light_direction_overlay
)
from physics.depth_estimation import estimate_depth_geometric

# ---------------- VIDEO SETUP ---------------- #
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('demo_output.avi', fourcc, 20.0, (640, 480))
cap = cv2.VideoCapture(0)

# ---------------- ANALYTICS ---------------- #
depth_history = []
shadow_area_history = []

# ---------------- PHYSICS CALIBRATION ---------------- #
CALIBRATION_DEPTH_CM = 2.0   # touching face
calibrated_k = None

# ---------------- SHADOW PHYSICS TUNING ---------------- #
MIN_SHADOW_AREA = 400        # below this → unreliable shadow
GAMMA = 1.7                  # response expansion
ALPHA = 1.2                  # scale factor

# ---------------- LIGHT STABILIZATION ---------------- #
light_dir_buffer = []
LOCK_FRAMES = 40

print("[INFO] Physics-Based Shadow Depth System Starting...")
print("[INFO] Hybrid Depth: Expanded Shadow Physics + Geometry Fallback")
print("[INFO] Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    face_box, hand_landmarks, hand_in_face = detect_face_hand(frame)

    action = "No Action"
    depth = None
    matrix_plot = None

    if face_box:
        x, y, w, h = face_box
        face_roi = frame[y:y+h, x:x+w]

        # -------- SHADOW SEGMENTATION -------- #
        shadow_mask, avg_intensity = segment_shadow_with_hand(
            face_roi, hand_in_face
        )

        shadow_area = np.sum(shadow_mask > 0)

        # -------- LIGHT DIRECTION (STABILIZED) -------- #
        raw_light_dir = estimate_light_direction(face_roi)
        if raw_light_dir is not None:
            light_dir_buffer.append(raw_light_dir)

        if len(light_dir_buffer) > LOCK_FRAMES:
            light_dir_buffer.pop(0)

        light_dir = None
        if len(light_dir_buffer) > 10:
            light_dir = np.mean(light_dir_buffer, axis=0)
            light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-6)

        # -------- SHADOW-CENTROID LIGHT CORRECTION -------- #
        ys, xs = np.where(shadow_mask > 0)
        if len(xs) > 50:
            shadow_cx = np.mean(xs)
            shadow_cy = np.mean(ys)
            face_cx, face_cy = w / 2, h / 2
            shadow_vec = np.array([face_cx - shadow_cx, face_cy - shadow_cy])
            light_dir = shadow_vec / (np.linalg.norm(shadow_vec) + 1e-6)

        if light_dir is not None:
            display_frame = create_light_direction_overlay(
                display_frame, face_box, light_dir
            )

        # -------- GEOMETRIC DEPTH -------- #
        depth_geom = estimate_depth_geometric(
            hand_landmarks, face_box, frame.shape
        )

        # -------- PHYSICS CALIBRATION (TOUCH ONLY) -------- #
        if calibrated_k is None and shadow_area > 800:
            calibrated_k = CALIBRATION_DEPTH_CM * np.sqrt(shadow_area)
            print(f"[CALIBRATED] k = {calibrated_k:.2f}")

        # -------- SHADOW PHYSICS DEPTH -------- #
        depth_shadow = None
        if calibrated_k is not None and shadow_area > MIN_SHADOW_AREA:
            raw_depth = calibrated_k / (np.sqrt(shadow_area) + 1e-6)

            # Non-linear expansion (KEY FIX)
            depth_shadow = ALPHA * (raw_depth ** GAMMA)
            depth_shadow = np.clip(depth_shadow, 1.5, 40.0)

        # -------- HYBRID DEPTH DECISION -------- #
        if depth_shadow is not None:
            depth = depth_shadow
            depth_source = "Physics (Expanded Shadow)"
        else:
            depth = depth_geom
            depth_source = "Geometry"

        # -------- ACTION CLASSIFICATION -------- #
        if depth:
            depth_history.append(depth)
            shadow_area_history.append(shadow_area)

            if depth < 3:
                action = "Touching Face / Eating"
            elif depth < 7:
                action = "Near Face"
            elif depth < 15:
                action = "Hand Approaching"
            else:
                action = "Hand Away"

            matrix_plot = create_shadow_intensity_matrix(
                face_roi, shadow_mask, round(depth, 2)
            )

            draw_overlay(display_frame, round(depth, 2), action)

            cv2.putText(display_frame, f"Depth Source: {depth_source}",
                        (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 200, 0), 2)

            cv2.putText(display_frame, f"Shadow Area: {shadow_area}px",
                        (30, 180), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

        cv2.rectangle(display_frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)

    else:
        cv2.putText(display_frame, "Waiting for face detection...",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

    cv2.imshow("Shadow-Depth Vision", display_frame)
    if matrix_plot is not None:
        cv2.imshow("Shadow Intensity Matrix", matrix_plot)

    out.write(display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("[INFO] Video saved as demo_output.avi")
print(f"[INFO] Frames with depth data: {len(depth_history)}")

# -------- ANALYTICS -------- #
if len(depth_history) > 10:
    from visualization.analytics import plot_depth_vs_shadow_relationship
    plot_depth_vs_shadow_relationship(depth_history, shadow_area_history)
