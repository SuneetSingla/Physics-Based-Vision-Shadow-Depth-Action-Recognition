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
