import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import shutil
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preprocess_dataset(
    input_dir="Output_Dir", 
    output_dir="Dataset_Processed",
    img_size=416,
    augment_train=True,
    augmentation_factor=2,  # How many augmented images to create per original image
):
    """
    Preprocess the IP102 dataset that was already converted to YOLO format.
    
    This function:
    1. Resizes all images to specified size (default 416x416)
    2. Normalizes pixel values to 0-1 range
    3. Applies augmentation to training set
    4. Creates a balanced dataset structure ready for training
    
    Args:
        input_dir: Path to the YOLOv5 formatted dataset (from dataset_prep.py)
        output_dir: Path to save the processed dataset
        img_size: Target image size (both width and height)
        augment_train: Whether to apply augmentation to training set
        augmentation_factor: Number of augmented versions to create per original image
    """
    print(f"Preprocessing dataset: resizing to {img_size}x{img_size}, normalizing, and augmenting...")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    for folder in ['images/train', 'images/val', 'images/test', 
                   'labels/train', 'labels/val', 'labels/test']:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    # Copy data.yaml file and update paths
    data_yaml_path = os.path.join(input_dir, 'data.yaml')
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths in data.yaml
        data_config['train'] = os.path.join(output_dir, 'images/train')
        data_config['val'] = os.path.join(output_dir, 'images/val')
        data_config['test'] = os.path.join(output_dir, 'images/test')
        
        # Write updated data.yaml
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    else:
        print(f"Warning: {data_yaml_path} not found. Please create data.yaml manually.")
    
    # Process each subset (train, val, test)
    for subset in ['train', 'val', 'test']:
        print(f"Processing {subset} set...")
        
        # Get list of image files in this subset
        subset_dir = os.path.join(input_dir, f'images/{subset}')
        if not os.path.exists(subset_dir):
            print(f"Warning: {subset_dir} not found. Skipping...")
            continue
            
        image_files = [f for f in os.listdir(subset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {subset} images"):
            img_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(subset_dir, img_file)
            label_path = os.path.join(input_dir, f'labels/{subset}/{img_id}.txt')
            
            # Skip if image doesn't have corresponding label file
            if not os.path.exists(label_path):
                print(f"Warning: No label file for {img_file}. Skipping...")
                continue
                
            # Read labels
            with open(label_path, 'r') as f:
                label_lines = f.readlines()
            
            # Parse YOLO format labels
            bboxes = []
            class_ids = []
            
            for line in label_lines:
                parts = line.strip().split()
                if len(parts) == 5:  # YOLO format: class_id x_center y_center width height
                    try:
                        # Handle potential float class IDs by converting to int
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate bounding box parameters
                        if width <= 0 or height <= 0:
                            print(f"Warning: Invalid bbox dimensions in {label_path}: width={width}, height={height}")
                            continue
                            
                        bboxes.append([x_center, y_center, width, height])
                        class_ids.append(class_id)
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing label {line.strip()} in {label_path}: {e}")
                        continue
            
            # Read and preprocess the image
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}. Skipping...")
                    continue
                    
                # Get original dimensions
                original_h, original_w = img.shape[:2]
                
                # 1. Resize the image to target size
                img_resized = cv2.resize(img, (img_size, img_size))
                
                # 2. Normalize pixel values to 0-1 range (float32)
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                # Save the original preprocessed image
                output_img_path = os.path.join(output_dir, f'images/{subset}/{img_id}.jpg')
                # Convert back to uint8 for saving with OpenCV
                cv2.imwrite(output_img_path, (img_normalized * 255).astype(np.uint8))
                
                # Copy the original label file
                output_label_path = os.path.join(output_dir, f'labels/{subset}/{img_id}.txt')
                shutil.copy(label_path, output_label_path)
                
                # 3. Apply augmentation to training set (only if it has valid bounding boxes)
                if subset == 'train' and augment_train and len(bboxes) > 0:
                    for aug_idx in range(augmentation_factor):
                        aug_img = img_normalized.copy()
                        aug_bboxes = bboxes.copy()
                        aug_class_ids = class_ids.copy()
                        
                        # Apply a series of augmentations with OpenCV
                        
                        # 3.1. Random horizontal flip (50% chance)
                        if random.random() < 0.5:
                            aug_img = cv2.flip(aug_img, 1)  # 1 = horizontal flip
                            # Update bounding boxes: x_center = 1 - x_center
                            for i in range(len(aug_bboxes)):
                                aug_bboxes[i][0] = 1.0 - aug_bboxes[i][0]
                        
                        # 3.2. Random vertical flip (30% chance)
                        if random.random() < 0.3:
                            aug_img = cv2.flip(aug_img, 0)  # 0 = vertical flip
                            # Update bounding boxes: y_center = 1 - y_center
                            for i in range(len(aug_bboxes)):
                                aug_bboxes[i][1] = 1.0 - aug_bboxes[i][1]
                        
                        # 3.3. Random rotation (90 degrees, 25% chance)
                        if random.random() < 0.25:
                            rot_angle = random.choice([90, 180, 270])
                            if rot_angle == 90:
                                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
                                # Update bounding boxes for 90 degrees rotation
                                for i in range(len(aug_bboxes)):
                                    x, y, w, h = aug_bboxes[i]
                                    aug_bboxes[i] = [1.0 - y, x, h, w]
                            elif rot_angle == 180:
                                aug_img = cv2.rotate(aug_img, cv2.ROTATE_180)
                                # Update bounding boxes for 180 degrees rotation
                                for i in range(len(aug_bboxes)):
                                    x, y, w, h = aug_bboxes[i]
                                    aug_bboxes[i] = [1.0 - x, 1.0 - y, w, h]
                            elif rot_angle == 270:
                                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                # Update bounding boxes for 270 degrees rotation
                                for i in range(len(aug_bboxes)):
                                    x, y, w, h = aug_bboxes[i]
                                    aug_bboxes[i] = [y, 1.0 - x, h, w]
                        
                        # 3.4. Random brightness adjustment (70% chance)
                        if random.random() < 0.7:
                            # Convert to HSV for easier brightness adjustment
                            hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV)
                            # Random brightness factor between 0.7 and 1.3
                            brightness_factor = random.uniform(0.7, 1.3)
                            # Apply to V channel (brightness)
                            hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 1.0)
                            # Convert back to RGB
                            aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                        
                        # 3.5. Random contrast adjustment (50% chance)
                        if random.random() < 0.5:
                            contrast_factor = random.uniform(0.7, 1.3)
                            aug_img = np.clip(aug_img * contrast_factor, 0, 1.0)
                        
                        # Validate bounding boxes after augmentation
                        valid_bboxes = []
                        valid_class_ids = []
                        
                        for i, bbox in enumerate(aug_bboxes):
                            x_center, y_center, width, height = bbox
                            
                            # Check if bbox is still valid after augmentation
                            if (0 < x_center < 1 and 0 < y_center < 1 and 
                                width > 0 and height > 0 and 
                                x_center - width/2 > 0 and x_center + width/2 < 1 and
                                y_center - height/2 > 0 and y_center + height/2 < 1):
                                valid_bboxes.append(bbox)
                                valid_class_ids.append(aug_class_ids[i])
                            else:
                                # Skip invalid bounding boxes
                                continue
                        
                        # Only save augmented image if it has valid bounding boxes
                        if valid_bboxes:
                            # Save augmented image
                            aug_img_id = f"{img_id}_aug{aug_idx+1}"
                            aug_img_path = os.path.join(output_dir, f'images/{subset}/{aug_img_id}.jpg')
                            
                            # Convert back to uint8 for saving with OpenCV
                            cv2.imwrite(aug_img_path, (aug_img * 255).astype(np.uint8))
                            
                            # Save augmented labels
                            aug_label_path = os.path.join(output_dir, f'labels/{subset}/{aug_img_id}.txt')
                            with open(aug_label_path, 'w') as f:
                                for i, bbox in enumerate(valid_bboxes):
                                    class_id = valid_class_ids[i]
                                    # YOLO format: class_id x_center y_center width height
                                    f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Print summary
    train_images = len(os.listdir(os.path.join(output_dir, 'images/train')))
    val_images = len(os.listdir(os.path.join(output_dir, 'images/val'))) if os.path.exists(os.path.join(output_dir, 'images/val')) else 0
    test_images = len(os.listdir(os.path.join(output_dir, 'images/test'))) if os.path.exists(os.path.join(output_dir, 'images/test')) else 0
    
    print("\nPreprocessing complete!")
    print(f"Processed dataset saved to: {output_dir}")
    print(f"Training images: {train_images}")
    print(f"Validation images: {val_images}")
    print(f"Test images: {test_images}")
    print(f"Total images: {train_images + val_images + test_images}")
    print(f"Image size: {img_size}x{img_size}")

def visualize_preprocessed_samples(dataset_dir, output_dir="sample_visualization", num_samples=5):
    """
    Visualize a few preprocessed samples with their bounding boxes to verify preprocessing
    
    Args:
        dataset_dir: Path to the preprocessed dataset
        output_dir: Path to save the visualizations
        num_samples: Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get some random samples from training set
    train_img_dir = os.path.join(dataset_dir, 'images/train')
    train_label_dir = os.path.join(dataset_dir, 'labels/train')
    
    image_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print("No images found in the training set.")
        return
    
    # Load class names
    class_names = []
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            if 'names' in data_config:
                class_names = data_config['names']
    
    # Pick random samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for sample_img in samples:
        img_id = os.path.splitext(sample_img)[0]
        img_path = os.path.join(train_img_dir, sample_img)
        label_path = os.path.join(train_label_dir, f"{img_id}.txt")
        
        # Read image and labels
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping...")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        
        # Read and draw bounding boxes
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            # Fix for floating point class IDs
                            class_id = int(float(parts[0]))
                            x_center = float(parts[1]) * w
                            y_center = float(parts[2]) * h
                            bbox_width = float(parts[3]) * w
                            bbox_height = float(parts[4]) * h
                            
                            # Calculate coordinates for rectangle
                            x1 = x_center - bbox_width / 2
                            y1 = y_center - bbox_height / 2
                            
                            # Create rectangle patch
                            rect = patches.Rectangle(
                                (x1, y1), bbox_width, bbox_height,
                                linewidth=2, edgecolor='r', facecolor='none'
                            )
                            ax.add_patch(rect)
                            
                            # Add class label
                            class_name = f"Class {class_id}"
                            if class_id < len(class_names):
                                class_name = class_names[class_id]
                            
                            plt.text(
                                x1, y1 - 5, class_name,
                                color='white', fontsize=12,
                                bbox=dict(facecolor='red', alpha=0.7)
                            )
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing label {line.strip()} in {label_path}: {e}")
                            continue
        
        plt.title(f"Sample: {img_id}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{img_id}_visualization.jpg"))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Path to your YOLOv5 formatted dataset (from dataset_prep.py)
    input_dir = r"Output_Dir"
    
    # Path to save the processed dataset
    output_dir = r"Dataset_Processed"
    
    # Process the dataset
    preprocess_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        img_size=416,  # Standard YOLOv5 input size
        augment_train=True,
        augmentation_factor=2  # Create 2 augmented versions per original image
    )
    
    # Visualize some samples (optional but helpful for verification)
    visualize_preprocessed_samples(
        dataset_dir=output_dir,
        output_dir="sample_visualization",
        num_samples=5
    )