import os
import sys
import yaml
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import time

# Add YOLOv5 directory to path
YOLOV5_PATH = r'yolov5'
sys.path.append(YOLOV5_PATH)

# Updated import - use scale_boxes instead of scale_coords
from utils.general import increment_path, check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.experimental import attempt_load

def parse_args():
    """
    Parse command line arguments for model inference
    """
    parser = argparse.ArgumentParser(description='Run inference with YOLOv5 model on pest images')
    parser.add_argument('--weights', type=str, default='runs/train/pest_detection/weights/best.pt', 
                       help='model.pt path (default: best.pt from standard training directory)')
    parser.add_argument('--source', type=str, default='./Dataset_Processed/images/test', 
                       help='source directory with images to run inference on')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], 
                       help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default='runs/detect', help='save to project/name')
    parser.add_argument('--name', default='pest_detection', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--data', type=str, default='./Dataset_Processed/data.yaml', help='data.yaml path for class names')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    
    return parser.parse_args()

def find_class_names(data_yaml_path):
    """
    Find class names from data.yaml file
    """
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            if 'names' in data_config:
                return data_config['names']
    
    # Try to find alternative data.yaml files
    alt_paths = ["./Output_Dir/data.yaml", "./Dataset_Processed/data.yaml", "./Dataset/data.yaml"]
    for alt_path in alt_paths:
        if os.path.exists(alt_path) and alt_path != data_yaml_path:
            try:
                with open(alt_path, 'r') as f:
                    data_config = yaml.safe_load(f)
                    if 'names' in data_config:
                        print(f"Found class names in {alt_path}")
                        return data_config['names']
            except:
                pass
    
    # If no class names found, return default names
    print("Warning: Could not find class names, using default names.")
    return ['pest']  # Default class name if none found

def main():
    # Parse arguments
    args = parse_args()
    print(f"Arguments: {args}")
    
    # Initialize device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {save_dir}")
    
    # Check if weights file exists
    if not os.path.exists(args.weights):
        print(f"Warning: Weights file {args.weights} not found!")
        # Try to find weights in common locations
        alt_weights = [
            "./runs/train/pest_detection/weights/best.pt", 
            "./runs/train/pest_detection/weights/last.pt",
            "./yolov5n.pt"
        ]
        for alt_weight in alt_weights:
            if os.path.exists(alt_weight):
                args.weights = alt_weight
                print(f"Found weights file at {alt_weight} instead. Using this path.")
                break
        else:
            raise FileNotFoundError(f"Could not find model weights file! Make sure the training was completed.")
    
    # Check if source directory exists
    if not os.path.exists(args.source):
        print(f"Warning: Source directory {args.source} not found!")
        # Try to find source directory in common locations
        alt_sources = [
            "./Dataset_Processed/images/test", 
            "./Dataset_Processed/images/val",
            "./Dataset/images/test",
            "./Dataset/images/val",
            "./Output_Dir/images/test"
        ]
        for alt_source in alt_sources:
            if os.path.exists(alt_source):
                args.source = alt_source
                print(f"Found source directory at {alt_source} instead. Using this path.")
                break
        else:
            raise FileNotFoundError(f"Could not find source directory! Provide a valid image directory.")
    
    # Get class names
    class_names = find_class_names(args.data)
    print(f"Class names: {class_names}")
    
    # Load model
    try:
        print(f"Loading model from {args.weights}...")
        model = attempt_load(args.weights, device=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(args.img_size[0], s=stride)  # check img_size
        
        # Get model info
        model_info = {
            "stride": stride,
            "image_size": imgsz,
            "num_classes": len(class_names),
            "model_type": args.weights.split('/')[-1].split('.')[0]
        }
        print(f"Model info: {model_info}")
        
        # Run inference
        print("\n=== Running Inference ===")
        print(f"Model: {args.weights}")
        print(f"Source: {args.source}")
        print(f"Image size: {imgsz} pixels")
        print(f"Confidence threshold: {args.conf_thres}")
        print(f"IoU threshold: {args.iou_thres}")
        print("=======================\n")
        
        # Create a TXT file to save detection results
        results_file = save_dir / "detection_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"# Pest Detection Results\n")
            f.write(f"# Model: {args.weights}\n")
            f.write(f"# Source: {args.source}\n")
            f.write(f"# Image size: {imgsz}\n")
            f.write(f"# Confidence threshold: {args.conf_thres}\n")
            f.write(f"# IoU threshold: {args.iou_thres}\n")
            f.write(f"# Format: image_path, class_name, confidence, x1, y1, x2, y2\n\n")
        
        # Process the images
        if os.path.isdir(args.source):
            # Process a directory of images
            image_files = [os.path.join(args.source, f) for f in os.listdir(args.source) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        else:
            # Process a single image
            image_files = [args.source]
        
        # Process each image
        total_detections = 0
        detection_time = 0
        results_summary = {}
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Create a new directory for each image
            img_name = os.path.basename(img_path)
            img_save_dir = save_dir / (img_name.split('.')[0])
            img_save_dir.mkdir(parents=True, exist_ok=True)
                
            # Preprocessing
            img_original = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz))
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dimension
            
            # Inference
            t0 = time.time()
            with torch.no_grad():
                pred = model(img, augment=args.augment)[0]
            
            # Apply NMS
            pred = non_max_suppression(
                pred, 
                args.conf_thres, 
                args.iou_thres, 
                classes=args.classes, 
                agnostic=args.agnostic_nms
            )
            t1 = time.time()
            inference_time = t1 - t0
            detection_time += inference_time
            
            # Process detections
            for i, det in enumerate(pred):
                # Get original image dimensions
                img_height, img_width = img_original.shape[:2]
                
                # Draw detections on a copy of the original image
                annotated_img = img_original.copy()
                annotator = Annotator(annotated_img, line_width=args.line_thickness, 
                                     example=str(class_names))
                
                detections = 0
                if len(det):
                    # Rescale boxes from img_size to original image size
                    # Use scale_boxes instead of scale_coords for newer YOLOv5 versions
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
                    
                    # Write results to file
                    with open(results_file, 'a') as f:
                        for *xyxy, conf, cls in reversed(det):
                            class_id = int(cls)
                            class_name = class_names[class_id]
                            
                            # Count detections by class
                            if class_name not in results_summary:
                                results_summary[class_name] = 0
                            results_summary[class_name] += 1
                            
                            # Format for file: image_path, class_name, confidence, x1, y1, x2, y2
                            bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            f.write(f"{img_path},{class_name},{conf:.4f},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
                            
                            # Draw bounding box
                            if not args.hide_labels and not args.hide_conf:
                                label = f"{class_name} {conf:.2f}"
                                annotator.box_label(xyxy, label, color=colors(class_id, True))
                            elif not args.hide_labels:
                                label = f"{class_name}"
                                annotator.box_label(xyxy, label, color=colors(class_id, True))
                            
                            detections += 1
                
                # Print number of detections for this image
                print(f"{img_path}: {detections} detections in {inference_time:.3f} seconds")
                total_detections += detections
                
                # Save the annotated image
                cv2.imwrite(str(img_save_dir / f"{img_name}_detected.jpg"), annotated_img)
                
                # Create a simple visualization with side-by-side comparison
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.title(f"Detected Pests ({detections})")
                plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(str(img_save_dir / f"{img_name}_comparison.jpg"))
                plt.close()
        
        # Print and save summary statistics
        avg_time = detection_time / len(image_files) if image_files else 0
        print(f"\nProcessing complete!")
        print(f"Total images processed: {len(image_files)}")
        print(f"Total detections: {total_detections}")
        print(f"Average detection time: {avg_time:.3f} seconds per image")
        print(f"Detection results saved to: {results_file}")
        print(f"Annotated images saved to: {save_dir}")
        
        # Print detection counts by class
        print("\nDetections by class:")
        for class_name, count in results_summary.items():
            print(f"  {class_name}: {count}")
        
        # Save summary to file
        with open(save_dir / "detection_summary.txt", 'w') as f:
            f.write(f"# Pest Detection Summary\n")
            f.write(f"Model: {args.weights}\n")
            f.write(f"Source: {args.source}\n")
            f.write(f"Images processed: {len(image_files)}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Average detection time: {avg_time:.3f} seconds per image\n\n")
            f.write("Detections by class:\n")
            for class_name, count in results_summary.items():
                f.write(f"  {class_name}: {count}\n")
        
        # Create a simple bar chart of detections by class
        if results_summary:
            plt.figure(figsize=(10, 6))
            classes = list(results_summary.keys())
            counts = list(results_summary.values())
            
            plt.bar(classes, counts, color='skyblue')
            plt.xlabel('Pest Class')
            plt.ylabel('Number of Detections')
            plt.title('Pest Detections by Class')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(str(save_dir / "detection_summary.png"))
            plt.close()
        
        # Print next steps
        print("\nNext steps:")
        print("1. Review the detection results and annotated images")
        print("2. Adjust confidence threshold if needed (--conf-thres)")
        print("3. Export the model for deployment:")
        print(f"   python {YOLOV5_PATH}/export.py --weights {args.weights} --include onnx coreml")
        print("4. Deploy to embedded device for real-time pest detection")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()