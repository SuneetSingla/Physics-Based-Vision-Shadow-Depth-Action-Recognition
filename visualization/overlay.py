import cv2

def draw_overlay(frame, depth, action):
    if depth:
        cv2.putText(frame, f"Distance: {depth} cm",
                    (30,40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.putText(frame, f"Action: {action}",
                (30,80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,255), 2)
