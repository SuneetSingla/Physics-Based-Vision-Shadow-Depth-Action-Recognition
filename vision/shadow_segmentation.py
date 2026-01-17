import cv2
import numpy as np

def segment_shadow_with_hand(face_roi, hand_in_face_coords=None):
    """
    Detect shadow cast by hand on face using:
    1. Darkness detection (adaptive threshold)
    2. Hand position proximity (if hand detected)
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Detect dark regions (potential shadows)
    shadow_mask = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 4
    )
    
    # If hand detected, filter shadow near hand position
    if hand_in_face_coords:
        hx, hy = hand_in_face_coords
        h, w = shadow_mask.shape
        
        # Create distance-based weight (shadows closer to hand are more likely)
        y_coords, x_coords = np.ogrid[:h, :w]
        distance_map = np.sqrt((x_coords - hx)**2 + (y_coords - hy)**2)
        
        # Weight mask: closer to hand = higher weight
        weight_mask = np.exp(-distance_map / (w * 0.3))
        
        # Apply weighted filtering
        shadow_mask = (shadow_mask * weight_mask).astype(np.uint8)
        shadow_mask = cv2.threshold(shadow_mask, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(shadow_mask, 50, 150)
    penumbra_strength = cv2.Laplacian(edges, cv2.CV_64F).var()

    
    avg_intensity = np.mean(gray)
    
    return shadow_mask, avg_intensity