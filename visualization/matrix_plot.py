import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def create_shadow_intensity_matrix(face_roi, shadow_mask, depth=None):
    """
    Create 2D heatmap showing light intensity loss due to shadow
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) 
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Face Region')
    axes[0].axis('off')
    
    intensity_loss = gray.copy().astype(float)
    intensity_loss[shadow_mask > 0] = intensity_loss[shadow_mask > 0] * 0.3
    
    im = axes[1].imshow(intensity_loss, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f'Shadow Intensity Matrix\nDepth: {depth} cm' if depth else 'Shadow Intensity Matrix')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    
    return cv2.cvtColor(plot_img[:,:,:3], cv2.COLOR_RGB2BGR)


def create_light_direction_overlay(frame, face_box, light_direction):
    """
    Draw arrow showing detected light source direction
    """
    if face_box is None or light_direction is None:
        return frame
    
    fx, fy, fw, fh = face_box
    center = (fx + fw//2, fy + fh//2)
    
    # Scale direction vector for visibility
    arrow_len = 80
    end_point = (
        int(center[0] + light_direction[0] * arrow_len),
        int(center[1] + light_direction[1] * arrow_len)
    )
    
    cv2.arrowedLine(frame, center, end_point, (0, 255, 255), 3, tipLength=0.3)
    cv2.putText(frame, "Light Direction", 
                (center[0] - 50, center[1] - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame