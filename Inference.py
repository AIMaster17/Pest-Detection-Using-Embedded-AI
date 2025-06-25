import cv2
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
import os

# Specify the directory containing test images
TEST_IMAGE_DIR = r"Dataset_Processed\images\test"  # Directory path

# Letterbox function to resize and pad image while maintaining aspect ratio
def letterbox(img, new_shape=(416, 416), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (left, top)

# Load class names from classes.txt
def load_class_names(classes_path):
    with open(classes_path, 'r') as f:
        lines = f.readlines()
        class_names = [line.strip().split(maxsplit=1)[-1] for line in lines]
    return class_names

# Preprocess the image
def preprocess_image(img_path, input_size):
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    img = original_img.copy()
    img, ratio, pad = letterbox(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img / 255.0
    img = np.expand_dims(img, 0)
    return img.astype(np.float32), ratio, pad, original_img, original_img.shape[:2]

# Postprocess the output
def postprocess_output(onnx_outputs, conf_thres, iou_thres=0.45):
    outputs = onnx_outputs[0]  # Shape: (batch_size, num_anchors * grid_size * grid_size, 4 + 1 + num_classes)
    num_classes = outputs.shape[2] - 5  # 4 (boxes) + 1 (objectness) = 5, rest are class scores
    boxes = outputs[:, :, :4]  # Extract box coordinates [x_center, y_center, width, height]
    confs = outputs[:, :, 4]  # Objectness score
    class_scores = outputs[:, :, 5:]  # Class confidence scores

    # Compute class-specific confidence scores
    class_confs = np.max(class_scores, axis=2)  # Max score across classes
    class_ids = np.argmax(class_scores, axis=2)  # Class with highest score

    # Flatten for NMS processing
    batch_size = outputs.shape[0]
    all_boxes = []
    all_scores = []
    all_class_ids = []
    for i in range(batch_size):
        scores = confs[i] * class_confs[i]  # Combine objectness and class confidence
        mask = scores >= conf_thres  # Use the passed conf_thres
        if np.any(mask):
            boxes_i = boxes[i][mask]  # Apply mask to boxes
            scores_i = scores[mask]
            class_ids_i = class_ids[i][mask]
            all_boxes.append(boxes_i)
            all_scores.append(scores_i)
            all_class_ids.append(class_ids_i)

    if not all_boxes:
        return []

    # Convert to numpy arrays and process with NMS
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    # Convert center format to corner format
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    detections = []
    for i in indices:
        if isinstance(i, (list, tuple, np.ndarray)):
            i = i[0]
        detections.append({
            'box': boxes[i],  # [x1, y1, x2, y2]
            'confidence': float(scores[i]),
            'class_id': int(class_ids[i])
        })
    return detections

# Scale boxes back to original image size
def scale_boxes(boxes, ratio, pad, img_shape):
    scaled_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = (x1 - pad[0]) / ratio[0]
        y1 = (y1 - pad[1]) / ratio[1]
        x2 = (x2 - pad[0]) / ratio[0]
        y2 = (y2 - pad[1]) / ratio[1]
        x1 = max(0, min(x1, img_shape[1]))
        y1 = max(0, min(y1, img_shape[0]))
        x2 = max(0, min(x2, img_shape[1]))
        y2 = max(0, min(y2, img_shape[0]))
        scaled_boxes.append([x1, y1, x2, y2])
    return scaled_boxes

# Run inference
def run_inference(session, image_path, class_names, conf_thres=0.25, iou_thres=0.45):
    input_img, ratio, pad, original_img, original_shape = preprocess_image(image_path, (416, 416))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_img})
    detections = postprocess_output(outputs, conf_thres, iou_thres)
    for det in detections:
        box = det['box']
        scaled_box = scale_boxes([box], ratio, pad, original_shape)[0]
        det['box'] = scaled_box
    return detections, original_img

# Draw detections on the image
def draw_detections(img, detections, class_names):
    for det in detections:
        box = det['box']
        conf = det['confidence']
        class_id = det['class_id']
        class_name = class_names[class_id]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

# List images in the directory and prompt user to select one
def select_image(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
    if not images:
        raise FileNotFoundError(f"No images found in directory: {directory}")
    print("Available images:")
    for i, img in enumerate(images, 1):
        print(f"{i}. {img}")
    while True:
        try:
            choice = int(input("Enter the number of the image to process (e.g., 1): ")) - 1
            if 0 <= choice < len(images):
                return os.path.join(directory, images[choice])
            else:
                print(f"Please enter a number between 1 and {len(images)}")
        except ValueError:
            print("Please enter a valid number")

# Parse command-line arguments (image path removed)
def parse_args():
    parser = argparse.ArgumentParser(description='Pest Detection Inference')
    parser.add_argument('--model', type=str, default='exported_models/best_onnx.onnx', help='Path to ONNX model')
    parser.add_argument('--classes', type=str, default='dataset/classes.txt', help='Path to classes.txt')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save output image')
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    class_names = load_class_names(args.classes)
    session = ort.InferenceSession(args.model)
    print(f"Using confidence threshold: {args.conf_thres}")  # Debug print
    image_path = select_image(TEST_IMAGE_DIR)
    print(f"Processing image: {image_path}")
    detections, original_img = run_inference(session, image_path, class_names, args.conf_thres, args.iou_thres)
    if not detections:
        print("No pests detected in the image.")
    for det in detections:
        class_name = class_names[det['class_id']]
        print(f"Detected {class_name} with confidence {det['confidence']:.2f}")
    result_img = draw_detections(original_img.copy(), detections, class_names)
    output_filename = f"Output/{Path(image_path).stem}_output.jpg"
    cv2.imwrite(output_filename, result_img)
    print(f"Output saved to {output_filename}")

if __name__ == "__main__":
    main()