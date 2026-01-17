import numpy as np

def estimate_depth_geometric(hand_landmarks, face_box, frame_shape):
    """
    Geometric depth estimation using hand-face landmark distances
    More robust than shadow-based method for indoor lighting
    
    Returns depth in cm (calibrated)
    """
    if hand_landmarks is None or face_box is None:
        return None
    
    h, w = frame_shape[:2]
    
    # Get hand center (wrist)
    wrist = hand_landmarks.landmark[0]
    hand_x, hand_y = wrist.x * w, wrist.y * h
    
    # Get face center
    fx, fy, fw, fh = face_box
    face_center_x = fx + fw / 2
    face_center_y = fy + fh / 2
    
    # Calculate pixel distance
    pixel_distance = np.sqrt(
        (hand_x - face_center_x)**2 + 
        (hand_y - face_center_y)**2
    )
    
    # Estimate depth using face width as reference
    # Average human face width ≈ 14cm
    # We use face_box width as scaling factor
    face_width_cm = 14.0  # Average face width
    pixels_per_cm = fw / face_width_cm
    
    # Convert pixel distance to cm
    depth_cm = pixel_distance / pixels_per_cm
    
    return round(depth_cm, 2)


def estimate_depth_hybrid(shadow_area, avg_face_intensity, k, 
                          hand_landmarks, face_box, frame_shape, 
                          use_geometric=True):
    """
    Hybrid approach: Use geometric when shadow is unreliable
    """
    if use_geometric:
        return estimate_depth_geometric(hand_landmarks, face_box, frame_shape)
    
    # Fallback to shadow-based (original method)
    if shadow_area == 0 or k is None or avg_face_intensity == 0:
        return None
    
    A_norm = shadow_area / (avg_face_intensity + 1e-6)
    depth = k / (np.sqrt(A_norm) + 1e-6)
    
    return round(depth, 2)