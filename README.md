# Physics-Based Shadow-Depth Action Recognition

## Overview
Real-time hand-to-face distance estimation using shadow geometry and inverse square law.

## Features
- Shadow-based depth calculation
- Light source direction detection
- Shadow intensity matrix visualization
- Action classification (Touching Face/Eating/Near Face)

## Physics Model
```
Shadow Area (A) ∝ 1/Distance²
d = k / √(A_normalized)
where A_normalized = shadow_pixels / avg_face_intensity
```

## Installation
```bash
pip install -r requirements.txt
python main.py
```

## Usage
1. Press 'C' to calibrate (hold hand 10cm from face)
2. Move hand to see real-time depth estimation
3. Press ESC to exit and view analytics

## Demo
[YouTube Video Link]

## Results
[Include analytics_report.png]
```

#### **8. LinkedIn Post Template:**
```
🔬 Just completed an advanced Computer Vision project: Physics-Based Shadow-Depth Estimation!

Unlike traditional depth sensors, this system uses SHADOW GEOMETRY to calculate hand-to-face distance in real-time.

Key innovations:
✅ Inverse Square Law for depth calculation
✅ Shadow intensity matrix visualization
✅ Light source direction detection
✅ Action classification (<2cm = "Touching Face")

The math: depth = k / √(shadow_area_normalized)

Challenges solved:
- Calibration under varying lighting
- Shadow segmentation with hand-position filtering
- Real-time matrix plotting

Tech stack: Python, OpenCV, MediaPipe, NumPy, Matplotlib

Full demo & code: [GitHub Link]
Video walkthrough: [YouTube Link]

#ComputerVision #MachineLearning #Python #AI #DeepLearning

@console.success