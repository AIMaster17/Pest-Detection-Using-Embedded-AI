import os
import sys
import yaml
import argparse
from pathlib import Path
import torch

# Add YOLOv5 directory to path
YOLOV5_PATH = r'yolov5'
sys.path.append(YOLOV5_PATH)

def parse_args():
    """
    Parse command line arguments for model export
    """
    parser = argparse.ArgumentParser(description='Export YOLOv5 pest detection model for Raspberry Pi deployment')
    parser.add_argument('--weights', type=str, default='runs/train/pest_detection/weights/best.pt', 
                       help='model.pt path (default: best.pt from standard training directory)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], 
                       help='image size for export (should match training size)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--include', nargs='+', default=['onnx', 'coreml'], 
                       help='export formats: torchscript, onnx, coreml, tflite, edgetpu, saved_model, pb, trt')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    
    return parser.parse_args()

def find_weights_file(weights_path):
    """
    Find the weights file in common locations if not found at specified path
    """
    if os.path.exists(weights_path):
        return weights_path
    
    # Try to find weights in common locations
    alt_weights = [
        "./runs/train/pest_detection/weights/best.pt", 
        "./runs/train/pest_detection/weights/last.pt",
        "./yolov5n.pt"
    ]
    
    for alt_weight in alt_weights:
        if os.path.exists(alt_weight):
            print(f"Found weights file at {alt_weight} instead. Using this path.")
            return alt_weight
    
    raise FileNotFoundError(f"Could not find model weights file! Please provide a valid path.")

def main():
    # Parse arguments
    args = parse_args()
    print(f"Arguments: {args}")
    
    # Verify weights file
    args.weights = find_weights_file(args.weights)
    print(f"Using weights: {args.weights}")
    
    # Create output directory for exported models
    output_dir = Path('./exported_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exported models will be saved to {output_dir}")
    
    # Import export module from YOLOv5
    sys.path.insert(0, YOLOV5_PATH)
    
    try:
        # Import the export function
        from export import run as export_run
        
        # Print export configuration
        print("\n=== Export Configuration ===")
        print(f"Model: {args.weights}")
        print(f"Image size: {args.img_size[0]}x{args.img_size[1]} pixels")
        print(f"Export formats: {args.include}")
        print(f"Half precision: {args.half}")
        print("=============================\n")
        
        # Create argument namespace for export
        class Opt:
            pass
        
        opt = Opt()
        # Transfer all arguments to opt
        opt.weights = args.weights
        opt.imgsz = args.img_size
        opt.include = args.include
        opt.device = args.device
        opt.half = args.half
        opt.inplace = args.inplace
        opt.dynamic = args.dynamic
        opt.simplify = args.simplify
        opt.opset = args.opset
        opt.verbose = args.verbose
        
        # Additional required parameters
        opt.train = False  # not training mode
        opt.data = None  # no data needed for export
        
        print("Starting model export...")
        # Run export
        export_run(
            weights=args.weights,
            imgsz=args.img_size,
            include=args.include,
            device=args.device,
            half=args.half,
            inplace=args.inplace,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
            verbose=args.verbose
        )
        
        # Move exported models to our output directory
        export_path = Path(args.weights).parent
        formats = {
            'onnx': '.onnx',
            'coreml': '.mlmodel',
            'torchscript': '.torchscript',
            'tflite': '.tflite',
            'edgetpu': '_edgetpu.tflite',
            'saved_model': '_saved_model',
            'pb': '_pb',
            'trt': '_engine.trt'
        }
        
        print("\nCopying exported models to output directory...")
        for fmt in args.include:
            if fmt in formats:
                ext = formats[fmt]
                base_name = Path(args.weights).stem
                src_file = Path(args.weights).parent / f"{base_name}{ext}"
                if os.path.exists(src_file):
                    import shutil
                    dst_file = output_dir / f"{base_name}_{fmt}{ext}"
                    shutil.copy(src_file, dst_file)
                    print(f"Copied {fmt} model to {dst_file}")
                else:
                    print(f"Warning: {fmt} model not found at {src_file}")
        
        print("\nRaspberry Pi 4 Deployment Instructions:")
        print("1. Transfer the exported models to your Raspberry Pi 4")
        print("   scp -r ./exported_models pi@your_pi_ip_address:/home/pi/pest_detection/")
        print("\n2. Install dependencies on Raspberry Pi:")
        print("   • For ONNX model:")
        print("     pip install onnxruntime opencv-python numpy")
        print("   • For CoreML model (if using macOS):")
        print("     pip install coremltools")
        print("\n3. Create a simple inference script on Raspberry Pi:")
        print("   • See the sample code snippet in the exported_models directory")
        
        # Create a sample inference script for Raspberry Pi
        with open(output_dir / "raspberry_pi_inference.py", "w") as f:
            f.write("""# Sample Raspberry Pi inference code for pest detection using ONNX model
import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path

# Configuration
MODEL_PATH = 'best_onnx.onnx'  # Path to your exported ONNX model
IMG_SIZE = (416, 416)          # Must match the model's expected input size
CONF_THRESHOLD = 0.25          # Confidence threshold
IOU_THRESHOLD = 0.45           # IoU threshold for NMS
 
# Load class names
CLASS_NAMES = ['pest']         # Update with your actual class names

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def preprocess_image(img_path, input_size):
    # Read image
    img = cv2.imread(img_path)
    # Resize and pad
    img, ratio, pad = letterbox(img, input_size, auto=False)
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Transpose to CHW format
    img = img.transpose(2, 0, 1)
    # Convert to float32 and normalize
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    # Add batch dimension
    img = np.expand_dims(img, 0)
    return img, ratio, pad, cv2.imread(img_path)

def postprocess_output(onnx_outputs, conf_thres=0.25, iou_thres=0.45):
    # Process ONNX outputs
    outputs = onnx_outputs[0]
    
    # Filter based on confidence threshold
    conf_mask = outputs[:, 4] >= conf_thres
    outputs = outputs[conf_mask]
    
    # If no detections pass threshold
    if len(outputs) == 0:
        return []
    
    # Get bounding boxes, confidence scores, and class IDs
    boxes = []
    scores = []
    class_ids = []
    
    for output in outputs:
        # Get box coordinates
        box = output[:4]
        # Get confidence
        conf = output[4]
        # Get class scores
        class_scores = output[5:]
        # Get class ID with highest score
        class_id = np.argmax(class_scores)
        # Get highest class score
        class_score = class_scores[class_id]
        # Final confidence
        confidence = conf * class_score
        
        # Filter by confidence again
        if confidence >= conf_thres:
            boxes.append(box)
            scores.append(confidence)
            class_ids.append(class_id)
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), 
        scores.tolist(), 
        conf_thres, 
        iou_thres
    )
    
    # Prepare results
    detections = []
    for i in indices:
        if isinstance(i, (list, tuple, np.ndarray)):
            i = i[0]  # Handle different OpenCV versions
        detections.append({
            'box': boxes[i],
            'confidence': float(scores[i]),
            'class_id': int(class_ids[i])
        })
    
    return detections

def draw_detections(img, detections, class_names):
    # Draw detections on image
    for det in detections:
        box = det['box']
        conf = det['confidence']
        class_id = det['class_id']
        
        # Convert box from center format to corner format
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_names[class_id]} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

def main():
    # Load ONNX model
    print(f"Loading ONNX model from {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    # Get image path (can be changed to camera input)
    image_folder = Path("test_images")
    if not image_folder.exists():
        print(f"Creating test_images directory at {image_folder}")
        image_folder.mkdir(exist_ok=True)
        print("Please place test images in this directory and run again.")
        return
    
    # Process each image in the folder
    image_files = [f for f in image_folder.glob("*.jpg") or image_folder.glob("*.png")]
    if not image_files:
        print("No images found in test_images directory.")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    for img_path in image_files:
        print(f"Processing {img_path}...")
        
        # Preprocess image
        input_img, ratio, pad, original_img = preprocess_image(str(img_path), IMG_SIZE)
        
        # Run inference
        start_time = time.time()
        output = session.run(None, {input_name: input_img})
        inference_time = time.time() - start_time
        
        # Postprocess output
        detections = postprocess_output(output, CONF_THRESHOLD, IOU_THRESHOLD)
        
        # Draw detections on image
        result_img = draw_detections(original_img.copy(), detections, CLASS_NAMES)
        
        # Save result
        result_path = f"result_{img_path.name}"
        cv2.imwrite(result_path, result_img)
        
        # Print results
        print(f"Processed {img_path.name} in {inference_time:.3f} seconds")
        print(f"Found {len(detections)} pests")
        print(f"Result saved to {result_path}")
        print("-------------------")
    
    print("All images processed!")

if __name__ == "__main__":
    main()
""")
        
        # Create a README file with deployment instructions
        with open(output_dir / "README_RASPBERRY_PI.md", "w") as f:
            f.write("""# Pest Detection Deployment Guide for Raspberry Pi 4

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

For advanced deployment, consider using Docker containers for easier management.""")
            
        print(f"Created deployment guide at {output_dir}/README_RASPBERRY_PI.md")
        
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()