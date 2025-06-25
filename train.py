import os
import sys
import yaml
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import shutil

# Add YOLOv5 directory to path
YOLOV5_PATH = r'yolov5'
sys.path.append(YOLOV5_PATH)

from utils.general import increment_path, init_seeds
from utils.metrics import fitness
from utils.plots import plot_results
from utils.torch_utils import select_device
from utils.callbacks import Callbacks

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv5 model on custom pest dataset')
    parser.add_argument('--data', type=str, default='./Dataset_Processed/data.yaml', 
                       help='data.yaml path (use Dataset_Processed for preprocessed data)')
    parser.add_argument('--hyp', type=str, default=YOLOV5_PATH + '/data/hyps/hyp.scratch-low.yaml', 
                        help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], 
                       help='train, val image size (pixels) - matched to preprocessed dataset')
    parser.add_argument('--weights', type=str, default='yolov5n.pt', 
                       help='initial weights path - using nano model for efficiency')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='pest_detection', help='save to project/name')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--patience', type=int, default=30, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use augmentation during training (besides those in preprocessing)')
    return parser.parse_args()

def verify_dataset_structure(data_yaml_path):
    """
    Verify dataset structure and fix paths if needed.
    """
    print(f"Verifying dataset structure at {data_yaml_path}...")
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get absolute path to project root (parent of yolov5 directory)
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(YOLOV5_PATH)))
    
    # Get base directory (where data.yaml is located)
    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    
    # Print debug information
    print(f"Project root: {project_root}")
    print(f"Base directory: {base_dir}")
    
    # Check if paths are absolute or relative
    train_path = data_config.get('train', '')
    val_path = data_config.get('val', '')
    test_path = data_config.get('test', '')
    
    # Ensure all paths are absolute
    if not os.path.isabs(train_path):
        train_path = os.path.join(project_root, train_path)
    if not os.path.isabs(val_path):
        val_path = os.path.join(project_root, val_path)
    if not os.path.isabs(test_path):
        test_path = os.path.join(project_root, test_path)
    
    # Dictionary to track if paths exist
    paths_exist = {'train': os.path.exists(train_path), 
                   'val': os.path.exists(val_path),
                   'test': os.path.exists(test_path)}
    
    print(f"Checking paths: {paths_exist}")
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")
    print(f"Test path: {test_path}")
    
    # Check if any paths don't exist
    if not all(paths_exist.values()):
        print("Some dataset paths don't exist. Attempting to fix...")
        
        # Try to find the paths relative to the project root
        if not paths_exist['train']:
            # Try different possible locations
            possible_train_paths = [
                os.path.join(project_root, 'Dataset_Processed/images/train'),
                os.path.join(project_root, 'Dataset/images/train'),
                os.path.join(project_root, 'Output_Dir/images/train')
            ]
            
            for path in possible_train_paths:
                if os.path.exists(path):
                    data_config['train'] = path
                    print(f"Fixed train path: {path}")
                    break
            else:
                print("WARNING: Could not find train images directory!")
        
        if not paths_exist['val']:
            # Try different possible locations
            possible_val_paths = [
                os.path.join(project_root, 'Dataset_Processed/images/val'),
                os.path.join(project_root, 'Dataset/images/val'),
                os.path.join(project_root, 'Output_Dir/images/val')
            ]
            
            for path in possible_val_paths:
                if os.path.exists(path):
                    data_config['val'] = path
                    print(f"Fixed val path: {path}")
                    break
            else:
                # If val directory doesn't exist, create it and copy some train images
                print("Val directory doesn't exist. Creating it...")
                
                # Find the train directory first
                train_dir = data_config.get('train', '')
                if os.path.exists(train_dir):
                    # Create val directory structure
                    val_dir = os.path.join(os.path.dirname(train_dir[:-5]), 'val')
                    os.makedirs(val_dir, exist_ok=True)
                    
                    # Also create val labels directory
                    train_label_dir = os.path.join(project_root, 'Dataset_Processed/labels/train')
                    val_label_dir = os.path.join(project_root, 'Dataset_Processed/labels/val')
                    os.makedirs(val_label_dir, exist_ok=True)
                    
                    # Copy some train images to val
                    train_images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    
                    # Take 15% of train images for validation
                    num_val = max(1, int(len(train_images) * 0.15))
                    val_images = train_images[:num_val]
                    
                    print(f"Copying {len(val_images)} images from train to val...")
                    for img in val_images:
                        # Copy image
                        src_img = os.path.join(train_dir, img)
                        dst_img = os.path.join(val_dir, img)
                        shutil.copy(src_img, dst_img)
                        
                        # Copy corresponding label
                        img_id = os.path.splitext(img)[0]
                        src_label = os.path.join(train_label_dir, f"{img_id}.txt")
                        dst_label = os.path.join(val_label_dir, f"{img_id}.txt")
                        if os.path.exists(src_label):
                            shutil.copy(src_label, dst_label)
                    
                    data_config['val'] = val_dir
                    print(f"Created val directory: {val_dir}")
                else:
                    print("ERROR: Cannot create val directory without train directory!")
        
        if not paths_exist['test']:
            # Try different possible locations
            possible_test_paths = [
                os.path.join(project_root, 'Dataset_Processed/images/test'),
                os.path.join(project_root, 'Dataset/images/test'),
                os.path.join(project_root, 'Output_Dir/images/test')
            ]
            
            for path in possible_test_paths:
                if os.path.exists(path):
                    data_config['test'] = path
                    print(f"Fixed test path: {path}")
                    break
            else:
                print("Test directory doesn't exist. Using val directory for testing.")
                data_config['test'] = data_config['val']
        
        # Save the updated data.yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Updated {data_yaml_path} with corrected paths.")
    else:
        print("All dataset paths exist!")
    
    # Update data config to use absolute paths
    data_config['train'] = train_path
    data_config['val'] = val_path
    data_config['test'] = test_path
    
    # Save the updated data.yaml with absolute paths
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    # Check if images exist in directories
    for subset in ['train', 'val', 'test']:
        subset_path = data_config.get(subset, '')
        if os.path.exists(subset_path):
            img_files = [f for f in os.listdir(subset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(img_files)} images in {subset} directory.")
        else:
            print(f"Warning: {subset} directory doesn't exist.")
    
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
    
    # Load hyperparameters
    with open(args.hyp) as f:
        hyp = yaml.safe_load(f)
    print("Loaded hyperparameters")
    
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
            raise FileNotFoundError(f"Could not find data.yaml file! Make sure you've run dataset_prep.py and preprocess_dataset.py first.")
    
    # Verify and fix dataset structure
    data_config = verify_dataset_structure(args.data)
    print(f"Dataset configuration: {data_config}")
    print(f"Number of classes: {data_config.get('nc', 'not specified')}")
    
    # Save a copy of the data.yaml file in the save directory for reference
    with open(save_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    # Initialize random seeds for reproducibility
    init_seeds(1)
    
    # Import train module from YOLOv5
    sys.path.insert(0, YOLOV5_PATH)
    
    try:
        # Import the train function properly
        from train import train as yolo_train
        
        # Print key training configuration
        print("\n=== Training Configuration ===")
        print(f"Model: YOLOv5n (nano model - optimized for speed and efficiency)")
        print(f"Image size: {args.img_size[0]}x{args.img_size[1]} pixels")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Dataset: {args.data}")
        print(f"Using augmentation: {args.augment}")
        print("============================\n")
        
        # Setup callbacks
        callbacks = Callbacks()
        
        # Create an opt object with all the parameters
        class Opt:
            pass
        
        opt = Opt()
        # Transfer all arguments to opt
        opt.weights = args.weights
        opt.cfg = ''  # Empty string for default configuration
        opt.data = args.data
        opt.epochs = args.epochs
        opt.batch_size = args.batch_size
        opt.imgsz = args.img_size[0]
        opt.project = args.project
        opt.name = args.name
        opt.exist_ok = args.exist_ok
        opt.cache = args.cache
        opt.nosave = args.nosave
        opt.workers = args.workers
        opt.freeze = args.freeze
        opt.save_period = args.save_period
        opt.patience = args.patience
        
        # Additional required parameters
        opt.multi_scale = False
        opt.image_weights = False
        opt.single_cls = False
        opt.optimizer = 'SGD'
        opt.sync_bn = False
        opt.quad = False
        opt.cos_lr = False
        opt.label_smoothing = 0.0
        opt.noval = False
        opt.evolve = False
        opt.bucket = ''
        opt.resume = False
        opt.local_rank = -1
        opt.entity = None
        opt.upload_dataset = False
        opt.bbox_interval = -1
        opt.artifact_alias = 'latest'
        opt.noplots = False
        opt.seed = 1
        opt.save_dir = save_dir
        opt.augment = args.augment
        opt.rect = False  # Rectangular training
        opt.adam = False  # Use torch.optim.Adam() optimizer
        opt.linear_lr = False  # Use linear lr scheduling
        opt.noautoanchor = False  # Disable autoanchor
        
        print("Starting YOLOv5 training...")
        # Call train with the correct parameters
        results = yolo_train(hyp, opt, device, callbacks)
        
        # Plot results
        plot_results(file=save_dir / 'results.csv')
        print(f"Results plotted and saved to {save_dir}")
        
        # Print final results
        print("\nTraining complete!")
        print(f"Final results (P, R, mAP@.5, mAP@.5:.95): {results}")
        
        # Calculate and display fitness score
        final_fitness = fitness(np.array(results).reshape(1, -1))
        print(f"Fitness score: {final_fitness}")
        
        # Save training summary
        with open(save_dir / 'training_summary.txt', 'w') as f:
            f.write(f"Model: YOLOv5n (nano)\n")
            f.write(f"Dataset: {args.data}\n")
            f.write(f"Number of classes: {data_config.get('nc', 'unknown')}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Image size: {args.img_size}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Final results (P, R, mAP@.5, mAP@.5:.95): {results}\n")
            f.write(f"Fitness score: {final_fitness}\n")
            f.write(f"Training completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Calculate and display training duration
            training_end_time = time.time()
            training_duration = training_end_time - (training_start_time if 'training_start_time' in locals() else training_end_time)
            f.write(f"Training duration: {time.strftime('%H:%M:%S', time.gmtime(training_duration))}\n")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Additional debug information
        print("\nChecking YOLOv5 train.py function signature:")
        try:
            import inspect
            from train import train
            print(inspect.signature(train))
            print("Available parameters for YOLOv5 train function:")
            print(inspect.getfullargspec(train))
        except Exception as debug_e:
            print(f"Could not inspect train function: {debug_e}")
        
        sys.exit(1)
    
    # Tips for next steps
    print("\nNext steps:")
    print(f"1. Evaluate your model using: python {YOLOV5_PATH}/val.py --weights {save_dir}/weights/best.pt --data {args.data}")
    print(f"2. Test inference: python {YOLOV5_PATH}/detect.py --weights {save_dir}/weights/best.pt --source path/to/test/images")
    print(f"3. Export to optimized formats: python {YOLOV5_PATH}/export.py --weights {save_dir}/weights/best.pt --include onnx coreml")
    print("4. Deploy to Raspberry Pi or other edge devices using the exported model")
    print("\nFor real-time pest detection, YOLOv5n is an excellent choice due to its efficiency!")

if __name__ == "__main__":
    training_start_time = time.time()
    main()