import cv2
import numpy as np

def estimate_light_direction(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    mean_x = np.mean(grad_x)
    mean_y = np.mean(grad_y)

    light_vector = np.array([mean_x, mean_y])
    norm = np.linalg.norm(light_vector) + 1e-6

    return light_vector / norm
