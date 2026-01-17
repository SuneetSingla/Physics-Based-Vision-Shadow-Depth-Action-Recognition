import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands

face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6
)

def detect_face_hand(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    face_results = face_detector.process(rgb)
    hand_results = hand_detector.process(rgb)

    face_box = None
    hand_landmarks = None
    hand_center_in_face = None

    if face_results.detections:
        det = face_results.detections[0]
        bbox = det.location_data.relative_bounding_box
        face_box = (
            int(bbox.xmin * w),
            int(bbox.ymin * h),
            int(bbox.width * w),
            int(bbox.height * h)
        )

    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        
        # Get hand center (wrist landmark)
        wrist = hand_landmarks.landmark[0]
        hand_x, hand_y = int(wrist.x * w), int(wrist.y * h)
        
        # Convert to face ROI coordinates if hand is near face
        if face_box:
            fx, fy, fw, fh = face_box
            if fx <= hand_x <= fx + fw and fy <= hand_y <= fy + fh:
                hand_center_in_face = (hand_x - fx, hand_y - fy)

    return face_box, hand_landmarks, hand_center_in_face