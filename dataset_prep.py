import os
import json
import shutil
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

def clean_xml(xml_file):
    """Clean up potentially malformed XML files.
    
    This function attempts to fix common XML issues by:
    1. Reading the file content
    2. Finding the main closing tag
    3. Keeping only content up to and including that closing tag
    4. Writing the cleaned content to a temporary file
    
    Args:
        xml_file (str): Path to the XML file that needs cleaning
        
    Returns:
        str: Path to the cleaned XML file (could be the original if no cleaning needed)
    """
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the main closing tag (usually </annotation>)
        main_tag = "annotation"  # Adjusted based on the XML structure you showed
        close_pos = content.find(f"</{main_tag}>")
        
        if close_pos > 0:
            # Keep only content up to and including the closing tag
            cleaned_content = content[:close_pos + len(f"</{main_tag}>")]
            
            # Write back the cleaned content to a temporary file
            temp_file = xml_file + ".temp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            return temp_file
        
        return xml_file
    except Exception as e:
        print(f"Error cleaning XML file {xml_file}: {e}")
        return xml_file

def prepare_ip102_dataset(source_dir, output_dir, split=(0.7, 0.15, 0.15)):
    """
    Prepares the IP102 dataset for YOLOv5 training.
    
    This function:
    1. Creates necessary output directories
    2. Loads class names from classes.txt
    3. Processes XML labels to extract bounding boxes
    4. Converts labels to YOLOv5 format
    5. Splits data into train/val/test sets
    6. Saves processed dataset in YOLOv5-compatible format
    
    Args:
        source_dir: Path to the downloaded IP102 dataset
        output_dir: Path to save the processed dataset
        split: Train/Val/Test split ratios (default: 70/15/15)
    """
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    for folder in ['images/train', 'images/val', 'images/test', 
                  'labels/train', 'labels/val', 'labels/test']:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    # Load class names - MODIFIED to handle numeric class IDs
    class_names = []
    id_to_index = {}  # Maps original IDs (like "101") to zero-based class indices
    
    try:
        with open(os.path.join(source_dir, 'classes.txt'), 'r') as f:
            for line in f.readlines():
                # Split by first whitespace to separate ID from class name
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    class_id = parts[0]  # Keep as string for easier comparison
                    class_name = parts[1]
                    class_names.append(class_name)
                    # Map the original ID to a zero-based index for YOLO format
                    id_to_index[class_id] = len(class_names) - 1
        
        print(f"Loaded {len(class_names)} classes from classes.txt")
        print(f"Class ID mapping: {id_to_index}")
    except Exception as e:
        print(f"Error loading classes: {e}")
        return
    
    # Save class names in YOLOv5 format
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"train: {os.path.join(output_dir, 'images/train')}\n")
        f.write(f"val: {os.path.join(output_dir, 'images/val')}\n")
        f.write(f"test: {os.path.join(output_dir, 'images/test')}\n")
        f.write(f"nc: {len(class_names)}\n")  # number of classes
        f.write(f"names: {[c for c in class_names]}\n")
    
    # Process labels
    print("Looking for XML labels...")
    
    image_to_labels = {}
    
    # Find the labels directory
    labels_dir = os.path.join(source_dir, 'labels')
    if not os.path.exists(labels_dir):
        labels_dir = os.path.join(source_dir, 'xml')  # Try another common name
    
    # If still not found, look for XML files directly in source_dir
    if not os.path.exists(labels_dir):
        labels_dir = source_dir
    
    # Check which directories exist
    images_dir = os.path.join(source_dir, 'images')
    if not os.path.exists(images_dir):
        images_dir = source_dir  # If no specific images directory, use source_dir
    
    print(f"Found images directory: {images_dir}")
    print(f"Found labels directory: {labels_dir}")
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} image files")
    
    # Keep track of success/failure statistics
    success_count = 0
    error_count = 0
    unknown_class_count = 0
    
    # Process each image and look for corresponding XML
    for image_file in tqdm(image_files, desc="Processing labels"):
        img_id = os.path.splitext(image_file)[0]
        
        # Look for corresponding XML file
        xml_file = os.path.join(labels_dir, f"{img_id}.xml")
        
        if os.path.exists(xml_file):
            try:
                # Try to clean up the XML file first
                cleaned_xml = clean_xml(xml_file)
                
                # Parse XML labels
                tree = ET.parse(cleaned_xml)
                root = tree.getroot()
                
                # Remove temp file if we created one
                if cleaned_xml != xml_file and os.path.exists(cleaned_xml):
                    os.remove(cleaned_xml)
                
                # Get image dimensions
                size_elem = root.find('./size')
                if size_elem is not None:
                    width = int(size_elem.find('./width').text)
                    height = int(size_elem.find('./height').text)
                else:
                    # If size not found in XML, skip this file
                    print(f"Warning: No size information in {xml_file}")
                    error_count += 1
                    continue
                
                labels = []
                
                for obj in root.findall('./object'):
                    name_elem = obj.find('./name')
                    if name_elem is None:
                        continue
                    
                    # Get the class ID from the XML (expecting numeric ID like "101")
                    class_id_str = name_elem.text
                    
                    # Skip if class ID not in our mapping
                    if class_id_str not in id_to_index:
                        print(f"Warning: Unknown class ID '{class_id_str}' in {xml_file}")
                        unknown_class_count += 1
                        continue
                    
                    # Convert to zero-based index
                    yolo_class_id = id_to_index[class_id_str]
                    
                    # Get bounding box coordinates (in format xmin, ymin, xmax, ymax)
                    bbox_elem = obj.find('./bndbox')
                    if bbox_elem is None:
                        continue
                        
                    xmin = float(bbox_elem.find('./xmin').text)
                    ymin = float(bbox_elem.find('./ymin').text)
                    xmax = float(bbox_elem.find('./xmax').text)
                    ymax = float(bbox_elem.find('./ymax').text)
                    
                    # Convert to YOLO format: x_center, y_center, width, height (normalized)
                    x_center = (xmin + xmax) / 2.0 / width
                    y_center = (ymin + ymax) / 2.0 / height
                    bbox_width = (xmax - xmin) / width
                    bbox_height = (ymax - ymin) / height
                    
                    # Ensure values are within bounds [0, 1]
                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    bbox_width = min(max(bbox_width, 0.0), 1.0)
                    bbox_height = min(max(bbox_height, 0.0), 1.0)
                    
                    labels.append({
                        'filename': image_file,
                        'bbox': [x_center, y_center, bbox_width, bbox_height],
                        'class_id': yolo_class_id
                    })
                
                if labels:
                    image_to_labels[img_id] = labels
                    success_count += 1
                    
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                error_count += 1
                continue
        else:
            # Uncomment this if you want to see which images have no labels
            # print(f"Warning: No labels found for {image_file}")
            pass
    
    print(f"Successfully processed {success_count} images with labels")
    print(f"Failed to process {error_count} images due to errors")
    print(f"Encountered {unknown_class_count} unknown class IDs")
    
    if not image_to_labels:
        print("No valid labels found! Please check your dataset structure.")
        return
    
    # Shuffle and split data
    all_image_ids = list(image_to_labels.keys())
    random.shuffle(all_image_ids)
    
    num_images = len(all_image_ids)
    train_size = int(num_images * split[0])
    val_size = int(num_images * split[1])
    
    train_ids = all_image_ids[:train_size]
    val_ids = all_image_ids[train_size:train_size + val_size]
    test_ids = all_image_ids[train_size + val_size:]
    
    # Function to copy images and create label files
    def process_subset(subset_ids, subset_name):
        """Process a subset of the dataset (train, val, or test).
        
        This function:
        1. Copies images to the appropriate output directory
        2. Creates YOLO-format label files
        
        Args:
            subset_ids (list): List of image IDs to process
            subset_name (str): Name of the subset ('train', 'val', or 'test')
        """
        print(f"Processing {subset_name} set...")
        processed_count = 0
        
        for img_id in tqdm(subset_ids):
            if img_id not in image_to_labels:
                continue
                
            labels = image_to_labels[img_id]
            
            if not labels:
                continue
                
            # Get first labels to determine filename
            filename = labels[0]['filename']
            src_img_path = os.path.join(images_dir, filename)
            
            # If the above path doesn't exist, try direct source directory
            if not os.path.exists(src_img_path):
                src_img_path = os.path.join(source_dir, filename)
            
            # If still doesn't exist, try with just the ID
            if not os.path.exists(src_img_path):
                possible_extensions = ['.jpg', '.jpeg', '.png']
                for ext in possible_extensions:
                    potential_path = os.path.join(images_dir, img_id + ext)
                    if os.path.exists(potential_path):
                        src_img_path = potential_path
                        filename = img_id + ext
                        break
            
            # Copy image
            if os.path.exists(src_img_path):
                dst_img_path = os.path.join(output_dir, f'images/{subset_name}', filename)
                shutil.copy(src_img_path, dst_img_path)
                
                # Create label file
                label_path = os.path.join(output_dir, f'labels/{subset_name}', 
                                         os.path.splitext(filename)[0] + '.txt')
                
                with open(label_path, 'w') as f:
                    for ann in labels:
                        # YOLO format: class_id x_center y_center width height
                        bbox = ann['bbox']
                        f.write(f"{ann['class_id']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                
                processed_count += 1
            else:
                print(f"Warning: Image not found - {filename}")
        
        return processed_count
    
    # Process each subset
    train_processed = process_subset(train_ids, 'train')
    val_processed = process_subset(val_ids, 'val')
    test_processed = process_subset(test_ids, 'test')
    
    print(f"Dataset preparation complete. Data stored in {output_dir}")
    print(f"Training set: {train_processed} images")
    print(f"Validation set: {val_processed} images")
    print(f"Test set: {test_processed} images")
    
    if train_processed + val_processed + test_processed == 0:
        print("WARNING: No images were processed. Please check your dataset structure and paths.")
    elif train_processed < 10 or val_processed == 0 or test_processed == 0:
        print("WARNING: Very few images were processed. This may not be sufficient for training.")

if __name__ == "__main__":
    # Example usage
    source_dir = "Dataset"  # Path to downloaded IP102 dataset
    output_dir = "Output_Dir"  # Path to save processed dataset
    
    prepare_ip102_dataset(source_dir, output_dir)