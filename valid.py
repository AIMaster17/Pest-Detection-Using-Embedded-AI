import os
import sys
import yaml
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Add YOLOv5 directory to path
YOLOV5_PATH = r'yolov5'
sys.path.append(YOLOV5_PATH)

from utils.general import increment_path, init_seeds
from utils.metrics import fitness
from utils.plots import plot_results
from utils.torch_utils import select_device
from utils.callbacks import Callbacks

def parse_args():
    """
    Parse command line arguments for validation
    """
    parser = argparse.ArgumentParser(description='Validate YOLOv5 model on pest detection dataset')
    parser.add_argument('--data', type=str, default='./Dataset_Processed/data.yaml', 
                       help='data.yaml path')
    parser.add_argument('--weights', type=str, default='runs/train/pest_detection/weights/best.pt', 
                       help='model.pt path (default: best.pt from standard training directory)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for validation')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], 
                       help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='val, test, or speed')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/validate', help='save to project/name')
    parser.add_argument('--name', default='pest_validation', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--verbose', action='store_true', help='print per-class results')
    parser.add_argument('--plots', action='store_true', default=True, help='generate plots of results')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results')
    
    return parser.parse_args()

def verify_dataset_structure(data_yaml_path):
    """
    Verify dataset structure and fix paths if needed.
    Similar to the function in train.py.
    """
    print(f"Verifying dataset structure at {data_yaml_path}...")
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get absolute path to project root (parent of yolov5 directory)
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(YOLOV5_PATH)))
    
    # Get base directory (where data.yaml is located)
    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    
    # Check if paths are absolute or relative
    val_path = data_config.get('val', '')
    test_path = data_config.get('test', '')
    
    # Ensure paths are absolute
    if not os.path.isabs(val_path):
        val_path = os.path.join(project_root, val_path)
    if not os.path.isabs(test_path):
        test_path = os.path.join(project_root, test_path)
    
    # Dictionary to track if paths exist
    paths_exist = {'val': os.path.exists(val_path),
                   'test': os.path.exists(test_path)}
    
    print(f"Checking validation paths: {paths_exist}")
    print(f"Val path: {val_path}")
    print(f"Test path: {test_path}")
    
    # Check and update paths if needed (simplified compared to train.py)
    for key, exists in paths_exist.items():
        if not exists:
            print(f"Warning: {key} path doesn't exist or is incorrect!")
            # Try alternative locations
            possible_paths = [
                os.path.join(project_root, f'Dataset_Processed/images/{key}'),
                os.path.join(project_root, f'Dataset/images/{key}'),
                os.path.join(project_root, f'Output_Dir/images/{key}')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data_config[key] = path
                    print(f"Fixed {key} path: {path}")
                    break
    
    # Save the updated data.yaml if paths were changed
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    # Check if images exist in directories
    for subset in ['val', 'test']:
        subset_path = data_config.get(subset, '')
        if os.path.exists(subset_path):
            img_files = [f for f in os.listdir(subset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(img_files)} images in {subset} directory.")
    
    return data_config

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
    
    # Check if data.yaml file exists
    if not os.path.exists(args.data):
        print(f"Warning: Data file {args.data} not found!")
        # Check alternative locations
        alt_paths = ["./Output_Dir/data.yaml", "./Dataset_Processed/data.yaml", "./Dataset/data.yaml"]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                args.data = alt_path
                print(f"Found data file at {alt_path} instead. Using this path.")
                break
        else:
            raise FileNotFoundError(f"Could not find data.yaml file! Make sure dataset structure is correct.")
    
    # Check if weights file exists
    if not os.path.exists(args.weights):
        print(f"Warning: Weights file {args.weights} not found!")
        # Try to find weights in common locations
        alt_weights = ["./runs/train/pest_detection/weights/best.pt", 
                      "./runs/train/pest_detection/weights/last.pt",
                      "./yolov5n.pt"]
        for alt_weight in alt_weights:
            if os.path.exists(alt_weight):
                args.weights = alt_weight
                print(f"Found weights file at {alt_weight} instead. Using this path.")
                break
        else:
            raise FileNotFoundError(f"Could not find model weights file! Make sure the training was completed.")
    
    # Verify and fix dataset structure
    data_config = verify_dataset_structure(args.data)
    print(f"Dataset configuration: {data_config}")
    print(f"Number of classes: {data_config.get('nc', 'not specified')}")
    
    # Save a copy of the data.yaml file in the save directory for reference
    with open(save_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    # Initialize random seeds for reproducibility
    init_seeds(1)
    
    # Import validation function from YOLOv5
    sys.path.insert(0, YOLOV5_PATH)
    
    try:
        # Import the val function properly
        from val import run as val_run
        
        # Print key validation configuration
        print("\n=== Validation Configuration ===")
        print(f"Model: {args.weights}")
        print(f"Image size: {args.img_size[0]}x{args.img_size[1]} pixels")
        print(f"Batch size: {args.batch_size}")
        print(f"Dataset: {args.data}")
        print(f"Confidence threshold: {args.conf_thres}")
        print(f"IoU threshold: {args.iou_thres}")
        print("============================\n")
        
        # Setup callbacks
        callbacks = Callbacks()
        
        # Create argument namespace for validation
        class Opt:
            pass
        
        opt = Opt()
        # Transfer all arguments to opt
        opt.data = args.data
        opt.weights = args.weights
        opt.batch_size = args.batch_size
        opt.imgsz = args.img_size[0]
        opt.conf_thres = args.conf_thres
        opt.iou_thres = args.iou_thres
        opt.task = args.task
        opt.device = device
        opt.workers = args.workers
        opt.save_txt = args.save_txt
        opt.save_hybrid = args.save_hybrid
        opt.save_conf = args.save_conf
        opt.save_json = args.save_json
        opt.project = args.project
        opt.name = args.name
        opt.exist_ok = args.exist_ok
        opt.verbose = args.verbose
        
        # Additional required parameters
        opt.save_dir = save_dir
        opt.plots = args.plots
        opt.augment = False  # No augmentation during validation
        opt.rect = True  # Rectangular validation
        opt.half = False  # FP16 validation
        opt.dnn = False  # Use DNN for ONNX
        opt.data_dict = data_config
        opt.single_cls = False
        opt.classes = None  # Filter by class, None = all classes
        opt.agnostic_nms = False  # Class-agnostic NMS
        opt.max_det = 1000  # Maximum detections per image
        
        print("Starting YOLOv5 validation...")
        # Run validation
        results, _, _ = val_run(
            data=args.data,
            weights=args.weights,
            batch_size=args.batch_size,
            imgsz=args.img_size[0],
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            device=device,
            save_json=args.save_json,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_hybrid=args.save_hybrid,
            verbose=args.verbose,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            task=args.task,
            plots=args.plots,
        )
        
        # Print the results
        mp, mr, map50, map = results[:4]  # mean precision, recall, mAP@0.5, mAP@0.5:0.95
        print(f"\nValidation Results:")
        print(f"Precision: {mp:.3f}")
        print(f"Recall: {mr:.3f}")
        print(f"mAP@0.5: {map50:.3f}")
        print(f"mAP@0.5:0.95: {map:.3f}")
        
        # Calculate and print fitness score
        fitness_score = fitness(np.array(results).reshape(1, -1))
        print(f"Fitness score: {fitness_score}")
        
        # Save validation summary
        with open(save_dir / 'validation_summary.txt', 'w') as f:
            f.write(f"Model: {args.weights}\n")
            f.write(f"Dataset: {args.data}\n")
            f.write(f"Number of classes: {data_config.get('nc', 'unknown')}\n")
            f.write(f"Image size: {args.img_size}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Confidence threshold: {args.conf_thres}\n")
            f.write(f"IoU threshold: {args.iou_thres}\n")
            f.write(f"Precision: {mp:.3f}\n")
            f.write(f"Recall: {mr:.3f}\n")
            f.write(f"mAP@0.5: {map50:.3f}\n")
            f.write(f"mAP@0.5:0.95: {map:.3f}\n")
            f.write(f"Fitness score: {fitness_score}\n")
        
        # Save precision-recall curves if plots are enabled
        if args.plots:
            print(f"Plots saved to {save_dir}")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print suggestions for next steps
    print("\nNext steps:")
    print(f"1. Test individual inference: python {YOLOV5_PATH}/detect.py --weights {args.weights} --source path/to/test/images")
    print(f"2. Export model to optimized formats: python {YOLOV5_PATH}/export.py --weights {args.weights} --include onnx coreml")
    print(f"3. Deploy to embedded devices using the exported model")

if __name__ == "__main__":
    main()