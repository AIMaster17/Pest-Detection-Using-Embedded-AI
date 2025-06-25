# Pest Detection Deployment Guide for Raspberry Pi 4

## Overview
This guide explains how to deploy the YOLOv5 pest detection model on a Raspberry Pi 4.

## Prerequisites
- Raspberry Pi 4 (2GB+ RAM recommended)
- Raspberry Pi OS (Buster or newer)
- Pi Camera or USB webcam
- Python 3.7+

## Setup Instructions

### 1. Transfer Model Files
Transfer the exported model files to your Raspberry Pi:

```bash
scp -r ./exported_models pi@<raspberry_pi_ip>:/home/pi/pest_detection/
```

### 2. Install Dependencies
SSH into your Raspberry Pi and install the required dependencies:

```bash
ssh pi@<raspberry_pi_ip>
cd /home/pi/pest_detection
pip3 install numpy opencv-python onnxruntime
```

For better performance, you can install OpenCV from source with optimizations.

### 3. Run Inference
Use the provided sample script for inference:

```bash
python3 raspberry_pi_inference.py
```

### 4. Real-time Detection with Camera
Modify the script to use your Pi Camera or USB webcam:

```python
# Add this to the beginning of the script
import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Initialize camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

# Allow camera to warm up
time.sleep(0.1)

# Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Grab the raw NumPy array representing the image
    image = frame.array
    
    # Process image here (similar to processing in the sample script)
    # ...
    
    # Show the frame
    cv2.imshow("Pest Detection", image)
    
    # Clear the stream for the next frame
    rawCapture.truncate(0)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Performance Optimization Tips

1. **Reduce Resolution**: Lower the input image size for faster processing
2. **Use Quantized Models**: Consider TFLite quantized models for better performance
3. **Process Every Nth Frame**: Skip frames to maintain real-time performance
4. **Optimize Threading**: Use separate threads for capture and processing
5. **Overclock (carefully)**: Increase Pi performance with proper cooling

## Troubleshooting

- **Memory Issues**: Use `top` command to monitor RAM usage; reduce batch size if needed
- **Slow Inference**: Try exporting to TFLite or quantized models for faster inference
- **Camera Errors**: Ensure camera is enabled in `raspi-config`
- **Missing Libraries**: Install additional dependencies as needed

For advanced deployment, consider using Docker containers for easier management.