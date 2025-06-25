# Sample Raspberry Pi inference code for pest detection using ONNX model
import cv2
import numpy as np
import onnxruntime as ort # type: ignore
import time
from pathlib import Path

# Configuration
MODEL_PATH = 'exported_models\best_onnx.onnx'  # Path to your exported ONNX model
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
