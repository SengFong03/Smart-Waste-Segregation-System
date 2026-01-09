import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from datetime import datetime
import io
import zipfile
import tempfile
import os
from collections import defaultdict
import time

# Import database classes
from database import WasteDetectionDB
from game_database import GameDatabase

# Page configuration
st.set_page_config(
    page_title="AI-Enhanced Waste Segregation for Sustainable Cities",
    page_icon="♻️",
    layout="wide"
)

# Load models (cached for performance)
@st.cache_resource
def load_models():
    waste_model = YOLO('models/best.pt')
    # Load a pre-trained YOLO model for person detection
    person_model = YOLO('yolov8n.pt')  # Lightweight model for person detection
    return waste_model, person_model

# Load model (backward compatibility)
@st.cache_resource
def load_model():
    return load_models()[0]

# Initialize databases
@st.cache_resource
def get_detection_database():
    return WasteDetectionDB()

@st.cache_resource
def get_game_database():
    return GameDatabase()

# Waste bin mapping for game mode
BIN_TYPES = {
    "recyclable": ["plastic", "metal", "glass", "paper", "cardboard"],
    "organic": ["biodegradable"],
    "general": []  # Everything else goes here
}

def get_correct_bin(waste_type):
    """Determine the correct bin for a waste type"""
    waste_type_lower = waste_type.lower()
    
    for bin_type, waste_categories in BIN_TYPES.items():
        if waste_type_lower in waste_categories:
            return bin_type
    
    return "general"  # Default bin

# Prediction function with confidence threshold
def predict_waste(image, model, confidence_threshold):
    results = model(image, conf=confidence_threshold)
    return results[0]

# Helper function to detect humans in the frame
def detect_humans(image, person_model, confidence_threshold=0.3):
    """Detect humans in the image and return their bounding boxes"""
    results = person_model(image, conf=confidence_threshold)
    human_boxes = []
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            # Class 0 is 'person' in COCO dataset
            if class_id == 0:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                human_boxes.append([x1, y1, x2, y2])
    
    return human_boxes

# Helper function to check if detection overlaps with human
def overlaps_with_human(detection_box, human_boxes, overlap_threshold=0.3):
    """Check if a detection box significantly overlaps with any human box"""
    if not human_boxes:
        return False
    
    dx1, dy1, dx2, dy2 = detection_box
    detection_area = (dx2 - dx1) * (dy2 - dy1)
    
    for hx1, hy1, hx2, hy2 in human_boxes:
        # Calculate intersection
        ix1 = max(dx1, hx1)
        iy1 = max(dy1, hy1)
        ix2 = min(dx2, hx2)
        iy2 = min(dy2, hy2)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            overlap_ratio = intersection_area / detection_area
            
            if overlap_ratio > overlap_threshold:
                return True
    
    return False

# Simple object tracker class
class SimpleTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid, class_name, confidence):
        self.objects[self.next_id] = {
            'centroid': centroid,
            'class_name': class_name,
            'confidence': confidence,
            'last_seen': time.time()
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            # Register all detections as new objects
            for detection in detections:
                centroid, class_name, confidence = detection
                self.register(centroid, class_name, confidence)
        else:
            # Match existing objects with new detections
            object_centroids = np.array([obj['centroid'] for obj in self.objects.values()])
            object_ids = list(self.objects.keys())
            
            detection_centroids = np.array([det[0] for det in detections])
            
            # Compute distances between object and detection centroids
            D = np.linalg.norm(object_centroids[:, np.newaxis] - detection_centroids, axis=2)
            
            # Find minimum distances and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_idxs = set()
            used_col_idxs = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue
                
                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    centroid, class_name, confidence = detections[col]
                    
                    # Update the object
                    self.objects[object_id] = {
                        'centroid': centroid,
                        'class_name': class_name,
                        'confidence': confidence,
                        'last_seen': time.time()
                    }
                    self.disappeared[object_id] = 0
                    
                    used_row_idxs.add(row)
                    used_col_idxs.add(col)
            
            # Handle unmatched detections and objects
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
            
            if D.shape[0] >= D.shape[1]:
                # More objects than detections
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than objects
                for col in unused_col_idxs:
                    centroid, class_name, confidence = detections[col]
                    self.register(centroid, class_name, confidence)
        
        return self.objects

# Video transformer class for real-time processing with human filtering and tracking
class WasteDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.waste_model, self.person_model = load_models()
        self.confidence_threshold = 0.5
        self.tracker = SimpleTracker(max_disappeared=15, max_distance=150)
        self.enable_human_filtering = True
        self.enable_tracking = True
        self.human_detection_confidence = 0.3
        self.frame_count = 0
        self.frame_skip = 1
    
    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold
    
    def set_human_filtering(self, enabled):
        self.enable_human_filtering = enabled
    
    def set_tracking(self, enabled):
        self.enable_tracking = enabled
    
    def set_frame_skip(self, skip):
        self.frame_skip = skip
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Frame skipping for performance
        if self.frame_count % self.frame_skip != 0:
            return img  # Return original frame without processing
        
        # Convert BGR to RGB for YOLO
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect humans first (if enabled)
        human_boxes = []
        if self.enable_human_filtering:
            human_boxes = detect_humans(img_rgb, self.person_model, self.human_detection_confidence)
        
        # Run waste detection
        results = self.waste_model(img_rgb, conf=self.confidence_threshold)
        
        # Filter detections and prepare for tracking
        valid_detections = []
        filtered_boxes = []
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detection_box = [x1, y1, x2, y2]
                
                # Check if detection overlaps with human (if filtering enabled)
                if self.enable_human_filtering and overlaps_with_human(detection_box, human_boxes, 0.3):
                    continue  # Skip this detection
                
                # Valid detection - prepare for tracking
                class_id = int(box.cls[0])
                class_name = self.waste_model.names[class_id]
                confidence = float(box.conf[0])
                
                # Calculate centroid for tracking
                centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
                
                valid_detections.append([centroid, class_name, confidence])
                filtered_boxes.append(box)
        
        # Update tracker (if enabled)
        tracked_objects = {}
        if self.enable_tracking:
            tracked_objects = self.tracker.update(valid_detections)
        
        # Create annotated image
        annotated_img = img_rgb.copy()
        
        if self.enable_tracking and tracked_objects:
            # Draw tracked objects with IDs
            for obj_id, obj_data in tracked_objects.items():
                centroid = obj_data['centroid']
                class_name = obj_data['class_name']
                confidence = obj_data['confidence']
                
                # Find corresponding detection box for this tracked object
                best_match_idx = -1
                best_distance = float('inf')
                
                for idx, detection in enumerate(valid_detections):
                    det_centroid = detection[0]
                    distance = np.sqrt((centroid[0] - det_centroid[0])**2 + (centroid[1] - det_centroid[1])**2)
                    if distance < best_distance:
                        best_distance = distance
                        best_match_idx = idx
                
                if best_match_idx >= 0 and best_distance < 50:
                    # Draw bounding box from corresponding detection
                    box = filtered_boxes[best_match_idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with ID
                    label = f"ID:{obj_id} {class_name} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_img, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            # Draw regular detections without tracking
            for i, box in enumerate(filtered_boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                class_name = self.waste_model.names[class_id]
                confidence = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Optionally draw human boxes for debugging
        if self.enable_human_filtering and human_boxes:
            for hbox in human_boxes:
                hx1, hy1, hx2, hy2 = [int(x) for x in hbox]
                cv2.rectangle(annotated_img, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)
                cv2.putText(annotated_img, "Person", (hx1, hy1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Convert back to BGR for display
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        
        return annotated_img_bgr

# Simple capture transformer for preview mode
class CapturePreviewTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture_frame = None
        self.should_capture = False
    
    def capture_image(self):
        self.should_capture = True
    
    def get_captured_frame(self):
        return self.capture_frame
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.should_capture:
            # Store the frame for processing
            self.capture_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.should_capture = False
        
        return img  # Return unprocessed frame for preview

# Game capture transformer for game mode
class GameCaptureTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture_frame = None
        self.should_capture = False
    
    def capture_image(self):
        self.should_capture = True
    
    def get_captured_frame(self):
        return self.capture_frame
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.should_capture:
            # Store the frame for processing
            self.capture_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.should_capture = False
        
        return img  # Return unprocessed frame for preview

# Function to process image on demand (for memory optimization)
def process_image_on_demand(file_object, model, confidence_threshold):
    """Process a single image on demand when needed for display"""
    try:
        image = Image.open(file_object)
        results = predict_waste(image, model, confidence_threshold)
        
        if len(results.boxes) > 0:
            return results.plot(), np.array(image)
        else:
            return np.array(image), np.array(image)
    except Exception:
        return None, None

# Function to create downloadable ZIP of annotated images
def create_annotated_images_zip(image_results, model, confidence_threshold):
    """Create a ZIP file containing all annotated images"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, result in image_results.items():
            try:
                # Get annotated image
                annotated_img = None
                
                if result['annotated_image'] is not None:
                    # Image was already processed and stored
                    annotated_img = result['annotated_image']
                elif result['file_object'] is not None:
                    # Memory optimization mode - process on demand
                    annotated_img, _ = process_image_on_demand(result['file_object'], model, confidence_threshold)
                
                if annotated_img is not None:
                    # Convert numpy array to PIL Image
                    if isinstance(annotated_img, np.ndarray):
                        pil_img = Image.fromarray(annotated_img)
                    else:
                        pil_img = annotated_img
                    
                    # Save image to bytes buffer
                    img_buffer = io.BytesIO()
                    
                    # Determine format from filename extension
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in ['.jpg', '.jpeg']:
                        pil_img.save(img_buffer, format='JPEG', quality=95)
                    elif file_ext == '.png':
                        pil_img.save(img_buffer, format='PNG')
                    else:
                        # Default to PNG for other formats
                        pil_img.save(img_buffer, format='PNG')
                        filename = os.path.splitext(filename)[0] + '.png'
                    
                    # Add to ZIP with "annotated_" prefix
                    base_name, ext = os.path.splitext(filename)
                    annotated_filename = f"annotated_{base_name}{ext}"
                    
                    zip_file.writestr(annotated_filename, img_buffer.getvalue())
                    
            except Exception as e:
                # Skip problematic images but continue with others
                continue
    
    zip_buffer.seek(0)
    return zip_buffer

# Function to create downloadable ZIP of original images (filtered results)
def create_filtered_images_zip(filtered_results, original_image_results):
    """Create a ZIP file containing only the filtered original images"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename in filtered_results.keys():
            try:
                result = original_image_results[filename]
                
                # Get original image
                original_img = None
                
                if result['original_image'] is not None:
                    # Image was stored during processing
                    original_img = result['original_image']
                elif result['file_object'] is not None:
                    # Memory optimization mode - read original file
                    original_img = np.array(Image.open(result['file_object']))
                
                if original_img is not None:
                    # Convert numpy array to PIL Image
                    if isinstance(original_img, np.ndarray):
                        pil_img = Image.fromarray(original_img)
                    else:
                        pil_img = original_img
                    
                    # Save image to bytes buffer
                    img_buffer = io.BytesIO()
                    
                    # Determine format from filename extension
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in ['.jpg', '.jpeg']:
                        pil_img.save(img_buffer, format='JPEG', quality=95)
                    elif file_ext == '.png':
                        pil_img.save(img_buffer, format='PNG')
                    else:
                        # Default to PNG for other formats
                        pil_img.save(img_buffer, format='PNG')
                        filename = os.path.splitext(filename)[0] + '.png'
                    
                    zip_file.writestr(filename, img_buffer.getvalue())
                    
            except Exception as e:
                # Skip problematic images but continue with others
                continue
    
    zip_buffer.seek(0)
    return zip_buffer

# Function to convert absolute coordinates to normalized YOLO format
def convert_to_yolo_format(bbox, image_width, image_height, class_id):
    """
    Convert absolute bounding box coordinates to normalized YOLO format
    
    Args:
        bbox: [x1, y1, x2, y2] - absolute pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        class_id: Integer class ID
    
    Returns:
        str: YOLO format line "class_id x_center y_center width height"
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center coordinates
    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)
    
    # Calculate normalized width and height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    
    # Ensure coordinates are within valid range [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# Function to create YOLO training dataset
def create_yolo_training_dataset(image_results, model, confidence_threshold):
    """
    Create a YOLO format training dataset with images and label files
    
    Args:
        image_results: Dictionary of image processing results
        model: YOLO model for class name mapping
        confidence_threshold: Minimum confidence for including detections
    
    Returns:
        io.BytesIO: ZIP file containing the training dataset
    """
    zip_buffer = io.BytesIO()
    
    # Class name to ID mapping
    class_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Create data.yaml content
    data_yaml_content = f"""# YOLO Dataset Configuration
# Generated from Streamlit Waste Classification App

path: ./training_dataset
train: images
val: images
test: images

nc: {len(class_names)}
names: {[name.lower() for name in class_names]}

# Class mapping:
{chr(10).join([f"# {idx}: {name.lower()}" for idx, name in enumerate(class_names)])}
"""
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add data.yaml file
        zip_file.writestr("training_dataset/data.yaml", data_yaml_content)
        
        # Create README file with instructions
        readme_content = """# YOLO Training Dataset

This dataset was exported from the Streamlit Waste Classification App.

## Structure:
- `images/`: Contains original images
- `labels/`: Contains YOLO format annotation files (.txt)
- `data.yaml`: Dataset configuration file

## Usage:
1. Extract this ZIP file
2. Use with YOLOv8 training:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='training_dataset/data.yaml', epochs=50)
   ```

## Label Format:
Each .txt file contains one line per object:
`class_id x_center y_center width height`

All coordinates are normalized (0-1 range relative to image dimensions).

## Classes:
0: biodegradable
1: cardboard
2: glass
3: metal
4: paper
5: plastic
"""
        zip_file.writestr("training_dataset/README.md", readme_content)
        
        successful_exports = 0
        failed_exports = 0
        
        for filename, result in image_results.items():
            try:
                # Get original image
                original_img = None
                if result['original_image'] is not None:
                    original_img = result['original_image']
                elif result['file_object'] is not None:
                    # Memory optimization mode - read original file
                    original_img = np.array(Image.open(result['file_object']))
                
                if original_img is None:
                    failed_exports += 1
                    continue
                
                # Convert numpy array to PIL Image if needed
                if isinstance(original_img, np.ndarray):
                    pil_img = Image.fromarray(original_img)
                else:
                    pil_img = original_img
                
                # Get image dimensions
                image_width, image_height = pil_img.size
                
                # Save original image to dataset
                img_buffer = io.BytesIO()
                
                # Determine format and save image
                base_name = os.path.splitext(filename)[0]
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in ['.jpg', '.jpeg']:
                    pil_img.save(img_buffer, format='JPEG', quality=95)
                    image_filename = f"{base_name}.jpg"
                elif file_ext == '.png':
                    pil_img.save(img_buffer, format='PNG')
                    image_filename = f"{base_name}.png"
                else:
                    # Convert to JPG for consistency
                    if pil_img.mode in ['RGBA', 'LA', 'P']:
                        # Convert transparent images to RGB
                        rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                        if pil_img.mode == 'P':
                            pil_img = pil_img.convert('RGBA')
                        rgb_img.paste(pil_img, mask=pil_img.split()[-1] if pil_img.mode in ['RGBA', 'LA'] else None)
                        pil_img = rgb_img
                    pil_img.save(img_buffer, format='JPEG', quality=95)
                    image_filename = f"{base_name}.jpg"
                
                # Add image to ZIP
                zip_file.writestr(f"training_dataset/images/{image_filename}", img_buffer.getvalue())
                
                # Create label file content
                label_lines = []
                
                # Process detections if available
                if result.get('detections') and len(result['detections']) > 0:
                    for detection in result['detections']:
                        if (detection.get('waste_type') and 
                            detection.get('waste_type') not in ['No detection'] and 
                            'Error' not in str(detection.get('waste_type', '')) and
                            detection.get('confidence', 0) >= confidence_threshold):
                            
                            # Map class name to ID
                            waste_type = detection['waste_type'].upper()
                            if waste_type in class_name_to_id:
                                class_id = class_name_to_id[waste_type]
                                
                                # Get bounding box coordinates
                                bbox = None
                                if 'bbox' in detection and detection['bbox']:
                                    bbox = detection['bbox']
                                elif all(key in detection for key in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']):
                                    bbox = [detection['bbox_x1'], detection['bbox_y1'], 
                                           detection['bbox_x2'], detection['bbox_y2']]
                                
                                if bbox and len(bbox) == 4:
                                    # Validate bbox coordinates
                                    x1, y1, x2, y2 = bbox
                                    if all(isinstance(coord, (int, float)) for coord in bbox) and x2 > x1 and y2 > y1:
                                        # Convert to YOLO format
                                        yolo_line = convert_to_yolo_format(bbox, image_width, image_height, class_id)
                                        label_lines.append(yolo_line)
                                    # If bbox validation fails, skip this detection
                                # If no valid bbox found, skip this detection
                
                # Create label file (even if empty for consistency)
                label_filename = f"{base_name}.txt"
                label_content = '\n'.join(label_lines) if label_lines else ''
                zip_file.writestr(f"training_dataset/labels/{label_filename}", label_content)
                
                successful_exports += 1
                
            except Exception as e:
                failed_exports += 1
                continue
        
        # Add export statistics file with debugging info
        stats_content = f"""# Export Statistics

Total Images Processed: {len(image_results)}
Successfully Exported: {successful_exports}
Failed Exports: {failed_exports}
Export Success Rate: {(successful_exports / len(image_results) * 100):.1f}%

Confidence Threshold Used: {confidence_threshold:.2f}
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Class Distribution (with valid bboxes):
"""
        
        # Count detections by class with detailed analysis
        class_counts = {name: 0 for name in class_names}
        total_detections = 0
        valid_bbox_count = 0
        
        for filename, result in image_results.items():
            if result.get('detections'):
                for detection in result['detections']:
                    waste_type = detection.get('waste_type', '').upper()
                    confidence = detection.get('confidence', 0)
                    
                    # Count all detections above threshold
                    if (waste_type in class_counts and 
                        confidence >= confidence_threshold and
                        waste_type not in ['NO DETECTION'] and
                        'ERROR' not in waste_type):
                        total_detections += 1
                        
                        # Check if bbox exists and is valid
                        has_valid_bbox = False
                        if 'bbox' in detection and detection['bbox']:
                            bbox = detection['bbox']
                            if len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                                x1, y1, x2, y2 = bbox
                                if x2 > x1 and y2 > y1:
                                    has_valid_bbox = True
                                    valid_bbox_count += 1
                                    class_counts[waste_type] += 1
        
        for class_name, count in class_counts.items():
            stats_content += f"{class_name.lower()}: {count} detections\n"
        
        stats_content += f"""
Detection Summary:
- Total detections above threshold: {total_detections}
- Detections with valid bboxes: {valid_bbox_count}
- Bbox success rate: {(valid_bbox_count / total_detections * 100):.1f}% (if > 0 detections)

If label files are empty, check:
1. Confidence threshold (currently {confidence_threshold:.2f})
2. Bounding box coordinate extraction
3. Class name mapping"""
        
        zip_file.writestr("training_dataset/export_stats.txt", stats_content)
    
    zip_buffer.seek(0)
    return zip_buffer, successful_exports, failed_exports

# Function to create streamlined PDF report (executive summary, parameters, and visual analytics only)
def create_pdf_report(image_results, stats, confidence_threshold):
    """
    Create a streamlined PDF report with executive summary, processing parameters, and visual analytics only

    Args:
        image_results: Dictionary of image processing results
        stats: Summary statistics from generate_summary_stats
        confidence_threshold: Confidence threshold used

    Returns:
        io.BytesIO: PDF file buffer
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.lib.colors import HexColor
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
    except ImportError:
        # Fallback to simple text-based PDF if ReportLab not available
        return create_simple_pdf_report(image_results, stats, confidence_threshold)
    
    # Create PDF buffer
    pdf_buffer = io.BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.darkgreen
    )
    
    # Title page
    elements.append(Paragraph("🗂️ Waste Classification Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(Paragraph("📋 Executive Summary", heading_style))
    
    if stats:
        summary_data = [
            ['Metric', 'Value', 'Details'],
            ['Total Images Processed', f"{stats['total_images']}", 'Images uploaded for analysis'],
            ['Successful Detections', f"{stats['successful_images']}", f"Images with detected waste items"],
            ['Detection Rate', f"{stats['detection_rate']:.1f}%", 'Percentage of images with detections'],
            ['Total Objects Found', f"{stats['total_detections']}", 'Individual waste items detected'],
            ['Average Confidence', f"{stats['average_confidence']:.1%}", 'Model confidence in detections'],
            ['Recyclable Items', f"{stats['recyclable_items']}", 'Items that can be recycled'],
            ['Recyclability Rate', f"{stats['recyclability_rate']:.1f}%", 'Percentage of recyclable items']
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(summary_table)
    
    elements.append(Spacer(1, 20))
    
    # Processing Parameters
    elements.append(Paragraph("⚙️ Processing Parameters", heading_style))
    param_text = f"""
    <b>Confidence Threshold:</b> {confidence_threshold:.0%}<br/>
    <b>Processing Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>Model:</b> YOLOv8 Waste Classification<br/>
    <b>Classes:</b> Biodegradable, Cardboard, Glass, Metal, Paper, Plastic
    """
    elements.append(Paragraph(param_text, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Visual Analytics Section
    if stats and stats['waste_type_counts']:
        elements.append(PageBreak())
        elements.append(Paragraph("📊 Visual Analytics", heading_style))
        
        # Create waste type distribution chart
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart for waste type distribution
            waste_types = list(stats['waste_type_counts'].keys())
            waste_counts = list(stats['waste_type_counts'].values())
            
            colors_list = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
            ax1.pie(waste_counts, labels=waste_types, autopct='%1.1f%%', startangle=90, colors=colors_list)
            ax1.set_title('Waste Type Distribution')
            
            # Bar chart for recyclability
            recyclable_data = ['Recyclable', 'Non-Recyclable']
            recyclable_counts = [stats['recyclable_items'], stats['total_detections'] - stats['recyclable_items']]
            
            bars = ax2.bar(recyclable_data, recyclable_counts, color=['#2E8B57', '#CD5C5C'])
            ax2.set_title('Recyclability Analysis')
            ax2.set_ylabel('Number of Items')
            
            # Add value labels on bars
            for bar, count in zip(bars, recyclable_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save chart to buffer
            chart_buffer = io.BytesIO()
            plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
            chart_buffer.seek(0)
            plt.close()
            
            # Add chart to PDF
            chart_image = RLImage(chart_buffer)
            chart_image.drawHeight = 300
            chart_image.drawWidth = 500
            elements.append(chart_image)
            elements.append(Spacer(1, 20))
            
        except Exception as e:
            elements.append(Paragraph(f"Chart generation error: {str(e)}", styles['Normal']))
    
    # Skip detailed results and additional statistics - streamlined report per user request
    # (Detailed detection results and additional statistics sections removed)


    # Streamlined report - individual image details and additional statistics removed per user request

    # Footer
    elements.append(Spacer(1, 30))
    footer_text = f"""
    <i>Report generated by AI-Enhanced Waste Segregation Platform<br/>
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    Confidence threshold: {confidence_threshold:.0%}</i>
    """
    elements.append(Paragraph(footer_text, styles['Normal']))

    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)

    return pdf_buffer

def create_simple_pdf_report(image_results, stats, confidence_threshold):
    """Simple text-based PDF fallback"""
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Waste Classification Report - Executive Summary', 0, 1, 'C')
            self.ln(10)

    # Create PDF and add basic content
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Processing completed with {stats.get("total_images", 0)} images', 0, 1)
    pdf.cell(0, 10, f'Confidence threshold: {confidence_threshold:.0%}', 0, 1)

    # Return PDF buffer
    pdf_buffer = io.BytesIO()
    pdf_buffer.write(pdf.output(dest='S').encode('latin-1'))
    pdf_buffer.seek(0)
    return pdf_buffer

def extract_images_from_zip(zip_file):
    """Extract and validate images from ZIP file"""
    extracted_files = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.is_dir():
                extracted_files.append(file_info.filename)
    return extracted_files

def process_bulk_images(uploaded_files, model, confidence_threshold, store_images=True, session_id=None, db=None):
    """Process multiple images with memory optimization"""
    # Basic implementation
    return {}

def generate_summary_stats(df, image_results):
    """Generate summary statistics from detection results"""
    return {
        'total_images': 0,
        'successful_images': 0,
        'total_detections': 0,
        'avg_confidence': 0.0,
        'error_images': 0,
        'no_detection_images': 0
    }
    
    # Additional Statistics
    elements.append(PageBreak())
    elements.append(Paragraph("📈 Additional Statistics", heading_style))
    
    # Create detailed statistics
    detailed_stats = []
    if stats:
        detailed_stats.extend([
            ['Processing Statistics', '', ''],
            ['Images with no detections', f"{stats['no_detection_images']}", f"{(stats['no_detection_images']/stats['total_images']*100):.1f}%"],
            ['Images with errors', f"{stats['error_images']}", f"{(stats['error_images']/stats['total_images']*100):.1f}%"],
            ['Average detections per image', f"{(stats['total_detections']/stats['successful_images']):.1f}" if stats['successful_images'] > 0 else "0", 'Objects per successful image'],
            ['', '', ''],
            ['Quality Metrics', '', ''],
            ['High confidence detections (>80%)', '', ''],
            ['Medium confidence detections (50-80%)', '', ''],
            ['Low confidence detections (<50%)', '', '']
        ])
    
    if detailed_stats:
        stats_table = Table(detailed_stats)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(stats_table)
    
    # Footer
    elements.append(Spacer(1, 30))
    footer_text = f"""
    <i>Report generated by AI-Enhanced Waste Segregation Platform<br/>
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    Confidence threshold: {confidence_threshold:.0%}</i>
    """
    elements.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    
    return pdf_buffer

# Fallback simple PDF function (in case ReportLab is not available)
def create_simple_pdf_report(image_results, stats, confidence_threshold):
    """Simple text-based PDF fallback"""
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Waste Classification Report', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Add content
    pdf.cell(0, 10, f'Processing Summary:', 0, 1)
    if stats:
        pdf.cell(0, 10, f'Total Images: {stats["total_images"]}', 0, 1)
        pdf.cell(0, 10, f'Detection Rate: {stats["detection_rate"]:.1f}%', 0, 1)
        pdf.cell(0, 10, f'Total Detections: {stats["total_detections"]}', 0, 1)
        pdf.cell(0, 10, f'Recyclable Items: {stats["recyclable_items"]}', 0, 1)
    
    # Save to buffer
    pdf_buffer = io.BytesIO()
    pdf_string = pdf.output(dest='S').encode('latin-1')
    pdf_buffer.write(pdf_string)
    pdf_buffer.seek(0)
    
    return pdf_buffer

# Function to extract images from ZIP file
def extract_images_from_zip(zip_file):
    """Extract image files from uploaded ZIP file and return as file-like objects"""
    extracted_files = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get list of all files in ZIP
            file_list = zip_ref.namelist()
            
            # Filter for image files
            image_files = [f for f in file_list if f.lower().endswith(supported_formats) and not f.startswith('__MACOSX/')]
            
            if not image_files:
                st.error("No supported image files found in the ZIP archive.")
                return []
            
            # Extract each image file
            for image_file in image_files:
                try:
                    # Extract file data
                    file_data = zip_ref.read(image_file)
                    
                    # Create a file-like object
                    file_obj = io.BytesIO(file_data)
                    file_obj.name = os.path.basename(image_file)  # Get just the filename
                    file_obj.size = len(file_data)
                    
                    # Verify it's a valid image
                    try:
                        img = Image.open(file_obj)
                        img.verify()  # Verify it's a valid image
                        file_obj.seek(0)  # Reset file pointer
                        extracted_files.append(file_obj)
                    except Exception:
                        st.warning(f"Skipping invalid image file: {image_file}")
                        continue
                        
                except Exception as e:
                    st.warning(f"Error extracting {image_file}: {str(e)}")
                    continue
            
            return extracted_files
            
    except zipfile.BadZipFile:
        st.error("Invalid ZIP file. Please upload a valid ZIP archive.")
        return []
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return []

# Bulk processing function with memory optimization and database support
def process_bulk_images(uploaded_files, model, confidence_threshold, store_images=True, session_id=None, db=None):
    results_data = []
    image_results = {}  # Store detection data and optionally images
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            
            # Load and process image
            image = Image.open(uploaded_file)
            results = predict_waste(image, model, confidence_threshold)
            
            # Extract detection data
            if len(results.boxes) > 0:
                detections = []
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    recyclable = class_name.lower() in ['paper', 'plastic', 'glass', 'metal', 'cardboard']
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection_data = {
                        'filename': uploaded_file.name,
                        'waste_type': class_name,
                        'confidence': confidence,
                        'recyclable': recyclable,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],  # Add bounding box coordinates
                        'bbox_x1': float(x1),  # Also add individual coordinates for compatibility
                        'bbox_y1': float(y1),
                        'bbox_x2': float(x2),
                        'bbox_y2': float(y2),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    results_data.append(detection_data)
                    detections.append(detection_data)
                
                # Store image results (optionally with images)
                annotated_img = results.plot() if len(results.boxes) > 0 else np.array(image)
                
                image_results[uploaded_file.name] = {
                    'annotated_image': annotated_img if store_images else None,
                    'original_image': np.array(image) if store_images else None,
                    'detections': detections,
                    'detection_count': len(results.boxes),
                    'status': 'success',
                    'file_object': uploaded_file if not store_images else None
                }
                
                # Save to database if enabled
                if db and session_id:
                    try:
                        from PIL import Image as PILImage
                        
                        # Convert images to bytes for database storage
                        original_buffer = io.BytesIO()
                        image.save(original_buffer, format='PNG')
                        original_bytes = original_buffer.getvalue()
                        
                        annotated_buffer = io.BytesIO()
                        PILImage.fromarray(annotated_img).save(annotated_buffer, format='PNG')
                        annotated_bytes = annotated_buffer.getvalue()
                        
                        # Add bounding box info to detections for database
                        db_detections = []
                        for j, detection in enumerate(detections):
                            db_detection = detection.copy()
                            if len(results.boxes) > j:
                                box = results.boxes[j]
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                db_detection.update({
                                    'bbox_x1': float(x1),
                                    'bbox_y1': float(y1), 
                                    'bbox_x2': float(x2),
                                    'bbox_y2': float(y2)
                                })
                            db_detections.append(db_detection)
                        
                        # Save to database
                        db.save_image_result(
                            session_id, 
                            uploaded_file.name,
                            original_bytes,
                            annotated_bytes,
                            db_detections,
                            getattr(uploaded_file, 'size', 0)
                        )
                    except Exception as db_error:
                        st.warning(f"Failed to save {uploaded_file.name} to database: {str(db_error)}")
                        continue
            else:
                # No detection
                no_detection_data = {
                    'filename': uploaded_file.name,
                    'waste_type': 'No detection',
                    'confidence': 0.0,
                    'recyclable': False,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                results_data.append(no_detection_data)
                
                image_results[uploaded_file.name] = {
                    'annotated_image': np.array(image) if store_images else None,
                    'original_image': np.array(image) if store_images else None,
                    'detections': [no_detection_data],
                    'detection_count': 0,
                    'status': 'no_detection',
                    'file_object': uploaded_file if not store_images else None
                }
                
        except Exception as e:
            # Error processing image
            error_data = {
                'filename': uploaded_file.name,
                'waste_type': f'Error: {str(e)}',
                'confidence': 0.0,
                'recyclable': False,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            results_data.append(error_data)
            
            image_results[uploaded_file.name] = {
                'annotated_image': None,
                'original_image': None,
                'detections': [error_data],
                'detection_count': 0,
                'status': 'error',
                'file_object': uploaded_file if not store_images else None
            }
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results_data), image_results

# Generate summary statistics
def generate_summary_stats(df, image_results):
    if df.empty:
        return None
    
    # Image-level statistics
    total_images = len(image_results)
    successful_images = sum(1 for result in image_results.values() if result['status'] == 'success')
    error_images = sum(1 for result in image_results.values() if result['status'] == 'error')
    no_detection_images = sum(1 for result in image_results.values() if result['status'] == 'no_detection')
    
    # Detection-level statistics
    valid_df = df[(df['waste_type'] != 'No detection') & (~df['waste_type'].str.contains('Error'))]
    
    stats = {
        'total_images': total_images,
        'successful_images': successful_images,
        'error_images': error_images,
        'no_detection_images': no_detection_images,
        'total_detections': len(valid_df),
        'detection_rate': (successful_images / total_images * 100) if total_images > 0 else 0,
        'average_confidence': valid_df['confidence'].mean() if len(valid_df) > 0 else 0,
        'recyclable_items': valid_df['recyclable'].sum() if len(valid_df) > 0 else 0,
        'recyclability_rate': (valid_df['recyclable'].mean() * 100) if len(valid_df) > 0 else 0,
        'waste_type_counts': valid_df['waste_type'].value_counts().to_dict() if len(valid_df) > 0 else {}
    }
    
    return stats

# Create visualizations
def create_charts(df):
    if df.empty:
        return None, None, None

    # Filter out errors and no detections for charts
    valid_df = df[(df['waste_type'] != 'No detection') & (~df['waste_type'].str.contains('Error'))]

    if valid_df.empty:
        return None, None, None

    # Waste type distribution pie chart
    waste_counts = valid_df['waste_type'].value_counts()
    pie_chart = px.pie(
        values=waste_counts.values,
        names=waste_counts.index,
        title="Waste Type Distribution"
    )

    # Confidence distribution histogram
    confidence_chart = px.histogram(
        valid_df,
        x='confidence',
        nbins=20,
        title="Confidence Score Distribution",
        labels={'confidence': 'Confidence Score', 'count': 'Number of Detections'}
    )

    # Recyclability summary bar chart
    recyclability_data = valid_df['recyclable'].value_counts()
    recyclability_chart = px.bar(
        x=['Non-Recyclable', 'Recyclable'],
        y=[recyclability_data.get(False, 0), recyclability_data.get(True, 0)],
        title="Recyclable vs Non-Recyclable Items",
        labels={'x': 'Category', 'y': 'Count'},
        color=['Non-Recyclable', 'Recyclable'],
        color_discrete_map={'Recyclable': '#2E8B57', 'Non-Recyclable': '#CD5C5C'}
    )

    return pie_chart, confidence_chart, recyclability_chart

def professional_dashboard():
    """Professional waste detection dashboard functionality"""
    st.header("🔬 Professional Analysis Dashboard")
    st.markdown("**Advanced waste classification with batch processing, database storage, and detailed analytics**")
    
    model = load_model()
    db = get_detection_database()
    
    # Sidebar settings
    st.sidebar.header("🔧 Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only detections above this confidence level will be shown"
    )
    
    st.sidebar.write(f"Current threshold: {confidence_threshold:.0%}")
    
    # Input type selection
    input_type = st.radio(
        "Choose analysis type:",
        ["Single Image", "Bulk Processing", "History & Database", "Real-time Webcam"],
        horizontal=True
    )
    
    if input_type == "Single Image":
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file:",
            type=['jpg', 'jpeg', 'png', 'webp', 'heic'],
            help="📱 Mobile users: Click to take photo with camera or select from gallery"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            # Make prediction
            st.subheader("🔍 Classification Results")
            
            with st.spinner("Analyzing image..."):
                results = predict_waste(image, model, confidence_threshold)
            
            # Display results
            if len(results.boxes) > 0:
                st.success(f"✅ Detected {len(results.boxes)} waste item{'s' if len(results.boxes) > 1 else ''}")
                
                # Results in two columns
                results_col1, results_col2 = st.columns(2)
                
                with results_col1:
                    st.subheader("🗂️ Detected Items")
                    
                    # Group detections by waste type
                    detections = []
                    for box in results.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        recyclable = class_name.lower() in ['paper', 'plastic', 'glass', 'metal', 'cardboard']
                        detections.append({
                            'waste_type': class_name,
                            'confidence': confidence,
                            'recyclable': recyclable
                        })
                    
                    # Group by waste type
                    waste_groups = {}
                    for detection in detections:
                        waste_type = detection['waste_type']
                        if waste_type not in waste_groups:
                            waste_groups[waste_type] = []
                        waste_groups[waste_type].append(detection)
                    
                    # Display grouped results
                    for waste_type, grouped_detections in waste_groups.items():
                        recyclable_icon = "♻️" if grouped_detections[0]['recyclable'] else "🗑️"
                        recyclable_text = "Recyclable" if grouped_detections[0]['recyclable'] else "Non-recyclable"
                        count = len(grouped_detections)
                        
                        with st.container():
                            if count == 1:
                                conf = grouped_detections[0]['confidence']
                                st.write(f"**{waste_type}**")
                                st.write(f"📊 **Confidence:** {conf:.1%}")
                                st.progress(conf)
                            else:
                                confidences = [d['confidence'] for d in grouped_detections]
                                avg_conf = sum(confidences) / len(confidences)
                                max_conf = max(confidences)
                                min_conf = min(confidences)
                                
                                st.write(f"**{waste_type}** (×{count} items)")
                                st.write(f"📊 **Average Confidence:** {avg_conf:.1%}")
                                st.write(f"📊 **Range:** {min_conf:.1%} - {max_conf:.1%}")
                                st.progress(avg_conf)
                            
                            # Recyclability status
                            if grouped_detections[0]['recyclable']:
                                st.success(f"{recyclable_icon} {recyclable_text}")
                            else:
                                st.warning(f"{recyclable_icon} {recyclable_text}")
                            
                            st.write("---")
                
                with results_col2:
                    st.subheader("🔍 Detection Results")
                    annotated_img = results.plot()
                    
                    # Resize for display
                    from PIL import Image as PILImage
                    if isinstance(annotated_img, np.ndarray):
                        display_img = PILImage.fromarray(annotated_img)
                    else:
                        display_img = annotated_img
                    
                    max_width = 600
                    max_height = 400
                    display_img.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)
                    
                    st.image(display_img, caption="Detected Objects", width=min(display_img.width, 600))
                
                # Summary statistics
                recyclable_count = sum(1 for box in results.boxes 
                                     if model.names[int(box.cls[0])].lower() in ['paper', 'plastic', 'glass', 'metal', 'cardboard'])
                non_recyclable_count = len(results.boxes) - recyclable_count
                
                st.subheader("📊 Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Total Items", len(results.boxes))
                
                with summary_col2:
                    st.metric("♻️ Recyclable", recyclable_count)
                
                with summary_col3:
                    st.metric("🗑️ Non-recyclable", non_recyclable_count)
                
            else:
                st.warning(f"⚠️ No waste items detected with confidence ≥ {confidence_threshold:.0%}")
                st.info("💡 Try lowering the confidence threshold or ensuring good lighting")
    
    elif input_type == "Bulk Processing":
        st.subheader("📦 Bulk Image Processing")
        st.markdown("Upload multiple images for batch classification and comprehensive analysis")
        
        # Upload method selection
        upload_method = st.radio(
            "Choose upload method:",
            ["Multiple Files", "ZIP Archive"],
            horizontal=True,
            help="Upload multiple files individually or upload a ZIP file containing images"
        )
        
        uploaded_files = []
        
        if upload_method == "Multiple Files":
            uploaded_files_raw = st.file_uploader(
                "Choose multiple image files:",
                type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Select multiple images to process in batch"
            )
            if uploaded_files_raw:
                uploaded_files = uploaded_files_raw
                
        else:  # ZIP Archive
            # ZIP file upload
            zip_file = st.file_uploader(
                "Choose a ZIP file containing images:",
                type=['zip'],
                help="Upload a ZIP file containing image files. Supported image formats: JPG, JPEG, PNG, WEBP, BMP, TIFF"
            )
            
            if zip_file:
                with st.spinner("Extracting images from ZIP file..."):
                    uploaded_files = extract_images_from_zip(zip_file)
                
                if uploaded_files:
                    st.success(f"Successfully extracted {len(uploaded_files)} images from ZIP file")
                else:
                    st.error("No valid images found in the ZIP file")
        
        if uploaded_files:
            st.info(f"📁 Selected {len(uploaded_files)} images for processing")
            
            # Show selected files with better formatting
            with st.expander("📋 View selected files"):
                if len(uploaded_files) <= 50:  # Show all files if <= 50
                    for i, file in enumerate(uploaded_files):
                        file_size = f"{file.size / 1024:.1f} KB" if file.size < 1024*1024 else f"{file.size / (1024*1024):.1f} MB"
                        st.write(f"{i+1}. **{file.name}** ({file_size})")
                else:  # Show first 25 and last 25 if more than 50
                    for i, file in enumerate(uploaded_files[:25]):
                        file_size = f"{file.size / 1024:.1f} KB" if file.size < 1024*1024 else f"{file.size / (1024*1024):.1f} MB"
                        st.write(f"{i+1}. **{file.name}** ({file_size})")
                    
                    st.write(f"... *({len(uploaded_files) - 50} more files)* ...")
                    
                    for i, file in enumerate(uploaded_files[-25:], len(uploaded_files)-25):
                        file_size = f"{file.size / 1024:.1f} KB" if file.size < 1024*1024 else f"{file.size / (1024*1024):.1f} MB"
                        st.write(f"{i+1}. **{file.name}** ({file_size})")
            
            # Tips for ZIP files
            if upload_method == "ZIP Archive":
                with st.expander("💡 Tips for ZIP files"):
                    st.markdown("""
                    **Best practices for ZIP uploads:**
                    - ✅ Place all images in the root of the ZIP file or organize in folders
                    - ✅ Supported formats: JPG, JPEG, PNG, WEBP, BMP, TIFF
                    - ✅ Files in subfolders will be automatically found
                    - ⚠️ Very large ZIP files (>500MB) may take time to upload
                    - ⚠️ Images with corrupted data will be automatically skipped
                    
                    **Memory recommendation:**
                    - For 1000+ images, enable memory optimization mode below
                    """)
            
            # Memory optimization option
            memory_mode = st.checkbox(
                "Memory optimization mode (for 500+ images)",
                help="Reduces memory usage by processing images on-demand. Recommended for large batches.",
                value=len(uploaded_files) > 500
            )
            
            # Initialize session state for processed results
            if 'processed_results' not in st.session_state:
                st.session_state.processed_results = None
                st.session_state.processed_files_hash = None
            
            # Create a hash of uploaded files to detect changes
            files_hash = hash(tuple(f.name + str(f.size) for f in uploaded_files))
            
            # Check if we need to reprocess (new files or different files)
            need_processing = (
                st.session_state.processed_results is None or 
                st.session_state.processed_files_hash != files_hash
            )
            
            if need_processing:
                process_button_text = "Process All Images"
            else:
                process_button_text = "Reprocess All Images"
            
            # Session name for database storage
            save_to_db = st.checkbox("💾 Save results to database", value=True)
            
            session_name = ""
            session_description = ""
            if save_to_db:
                col1, col2 = st.columns(2)
                with col1:
                    session_name = st.text_input(
                        "Session Name:",
                        value=f"Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}",
                        help="Name for this analysis session"
                    )
                with col2:
                    session_description = st.text_area(
                        "Description (optional):",
                        height=70,
                        placeholder="Brief description of this analysis batch..."
                    )

            if st.button(process_button_text, type="primary") or not need_processing:
                if need_processing:
                    st.subheader("📊 Processing Results")
                    
                    # Create database session if saving
                    session_id = None
                    if save_to_db and session_name.strip():
                        session_id = db.create_session(
                            session_name.strip(), 
                            confidence_threshold, 
                            session_description.strip()
                        )
                        st.info(f"📝 Created session: {session_name} (ID: {session_id})")
                    
                    # Process all images
                    with st.spinner("Processing images..."):
                        store_images = not memory_mode
                        results_df, image_results = process_bulk_images(
                            uploaded_files, model, confidence_threshold, 
                            store_images, session_id, db if save_to_db else None
                        )
                    
                    # Store results in session state
                    st.session_state.processed_results = {
                        'results_df': results_df,
                        'image_results': image_results,
                        'memory_mode': memory_mode
                    }
                    st.session_state.processed_files_hash = files_hash
                else:
                    st.subheader("📊 Processing Results")
                
                # Get results from session state
                results_df = st.session_state.processed_results['results_df']
                image_results = st.session_state.processed_results['image_results']
                memory_mode = st.session_state.processed_results['memory_mode']
                
                # Generate and display statistics
                stats = generate_summary_stats(results_df, image_results)
                
                if stats:
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Images", stats['total_images'])
                    with col2:
                        st.metric("Successful Detections", f"{stats['successful_images']}/{stats['total_images']}")
                    with col3:
                        st.metric("Detection Rate", f"{stats['detection_rate']:.1f}%")
                    with col4:
                        st.metric("Total Objects Found", stats['total_detections'])
                    
                    # Second row of metrics
                    col5, col6, col7, col8 = st.columns(4)
                    
                    with col5:
                        st.metric("Avg Confidence", f"{stats['average_confidence']:.1%}")
                    with col6:
                        st.metric("Recyclable Items", stats['recyclable_items'])
                    with col7:
                        st.metric("Recyclability Rate", f"{stats['recyclability_rate']:.1f}%")
                    with col8:
                        if stats['error_images'] > 0 or stats['no_detection_images'] > 0:
                            st.metric("Issues", f"{stats['error_images'] + stats['no_detection_images']}")
                        else:
                            st.metric("Issues", "0")
                    
                    # Create and display charts
                    pie_chart, confidence_chart, recyclability_chart = create_charts(results_df)
                    
                    if pie_chart:
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            st.plotly_chart(pie_chart, use_container_width=True)
                            st.plotly_chart(recyclability_chart, use_container_width=True)
                        
                        with chart_col2:
                            st.plotly_chart(confidence_chart, use_container_width=True)
                    
                    # Image Results Section with Visual Display and Filtering
                    st.subheader("🖼️ Image-by-Image Results")
                    
                    # Create tabs for different viewing modes
                    tab1, tab2 = st.tabs(["Visual Results", "Data Table"])
                    
                    with tab1:
                        # Filtering and Search Controls
                        st.markdown("**Filter Results:**")
                        filter_col1, filter_col2, filter_col3 = st.columns(3)
                        
                        # Store previous filter state to detect changes
                        if 'prev_filters' not in st.session_state:
                            st.session_state.prev_filters = {"status": "All", "waste_type": "All", "filename": ""}
                        
                        with filter_col1:
                            # Status filter
                            status_filter = st.selectbox(
                                "Filter by status:",
                                ["All", "Success", "No Detection", "Error"],
                                key="status_filter"
                            )
                        
                        with filter_col2:
                            # Waste type filter
                            all_waste_types = set()
                            for result in image_results.values():
                                for detection in result['detections']:
                                    if detection['waste_type'] not in ['No detection'] and 'Error' not in detection['waste_type']:
                                        all_waste_types.add(detection['waste_type'])
                            
                            waste_type_filter = st.selectbox(
                                "Filter by waste type:",
                                ["All"] + sorted(list(all_waste_types)),
                                key="waste_type_filter"
                            )
                        
                        with filter_col3:
                            # Filename search
                            filename_search = st.text_input(
                                "Search filename:",
                                placeholder="Enter filename to search...",
                                key="filename_search"
                            )
                        
                        # Check if filters have changed and reset page if so
                        current_filters = {"status": status_filter, "waste_type": waste_type_filter, "filename": filename_search}
                        if current_filters != st.session_state.prev_filters:
                            st.session_state.current_page = 1  # Reset to first page when filters change
                            st.session_state.prev_filters = current_filters
                        
                        # Apply filters
                        filtered_results = {}
                        for filename, result in image_results.items():
                            # Status filter
                            if status_filter != "All":
                                if status_filter == "Success" and result['status'] != 'success':
                                    continue
                                elif status_filter == "No Detection" and result['status'] != 'no_detection':
                                    continue
                                elif status_filter == "Error" and result['status'] != 'error':
                                    continue
                            
                            # Waste type filter
                            if waste_type_filter != "All":
                                found_waste_type = False
                                for detection in result['detections']:
                                    if detection['waste_type'] == waste_type_filter:
                                        found_waste_type = True
                                        break
                                if not found_waste_type:
                                    continue
                            
                            # Filename search
                            if filename_search and filename_search.lower() not in filename.lower():
                                continue
                            
                            filtered_results[filename] = result
                        
                        # Pagination controls
                        total_images = len(filtered_results)
                        
                        # Images per page selector
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            images_per_page = st.selectbox(
                                "Images per page:",
                                [10, 20, 50, 100],
                                index=0,
                                key="images_per_page"
                            )
                        
                        # Calculate total pages
                        total_pages = (total_images + images_per_page - 1) // images_per_page if total_images > 0 else 0
                        
                        with col2:
                            # Initialize current page in session state
                            if 'current_page' not in st.session_state:
                                st.session_state.current_page = 1
                            
                            if total_pages > 0:
                                # Reset to page 1 if current page exceeds total pages (due to filtering)
                                if st.session_state.current_page > total_pages:
                                    st.session_state.current_page = 1
                                    
                                current_page = st.number_input(
                                    "Page:",
                                    min_value=1,
                                    max_value=total_pages,
                                    value=st.session_state.current_page,
                                    key="page_input"
                                )
                                
                                # Update session state when page input changes
                                if current_page != st.session_state.current_page:
                                    st.session_state.current_page = current_page
                            else:
                                current_page = 1
                        
                        with col3:
                            if total_images > 0:
                                st.write(f"Showing page {current_page} of {total_pages} ({total_images} filtered images)")
                            else:
                                st.write("No images match the current filters")
                        
                        # Calculate start and end indices
                        if total_images > 0:
                            start_idx = (current_page - 1) * images_per_page
                            end_idx = min(start_idx + images_per_page, total_images)
                            
                            # Get current page items
                            image_items = list(filtered_results.items())
                            current_page_items = image_items[start_idx:end_idx]
                        else:
                            current_page_items = []
                        
                        # Display results for current page only - inline images
                        for idx, (filename, result) in enumerate(current_page_items):
                            # Create a container for each image result
                            with st.container():
                                # Header with filename and detection count
                                st.subheader(f"📸 {filename} - {result['detection_count']} objects detected")
                                
                                col_img, col_details = st.columns([2, 1])
                                
                                with col_img:
                                    # Handle image display based on memory mode
                                    if result['annotated_image'] is not None:
                                        # Image was stored during processing - resize for display
                                        from PIL import Image as PILImage
                                        if isinstance(result['annotated_image'], np.ndarray):
                                            display_img = PILImage.fromarray(result['annotated_image'])
                                        else:
                                            display_img = result['annotated_image']
                                        
                                        # Resize for display while keeping aspect ratio
                                        max_width = 600
                                        max_height = 400
                                        display_img.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)
                                        
                                        st.image(display_img, caption=f"Analyzed: {filename}", width=min(display_img.width, 600))
                                    elif result['file_object'] is not None:
                                        # Memory optimization mode - load and show image immediately
                                        with st.spinner(f"Loading {filename}..."):
                                            annotated_img, original_img = process_image_on_demand(
                                                result['file_object'], model, confidence_threshold
                                            )
                                            if annotated_img is not None:
                                                # Resize for display
                                                from PIL import Image as PILImage
                                                if isinstance(annotated_img, np.ndarray):
                                                    display_img = PILImage.fromarray(annotated_img)
                                                else:
                                                    display_img = annotated_img
                                                
                                                max_width = 600
                                                max_height = 400
                                                display_img.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)
                                                
                                                st.image(display_img, caption=f"Analyzed: {filename}", width=min(display_img.width, 600))
                                            else:
                                                st.error("Could not load image")
                                    else:
                                        st.error("Could not display image due to processing error")
                                
                                with col_details:
                                    # Status badge with better visual feedback
                                    if result['status'] == 'success':
                                        st.success(f"✅ Status: Success")
                                    elif result['status'] == 'no_detection':
                                        st.warning(f"⚠️ Status: No Detection")
                                    else:
                                        st.error(f"❌ Status: Error")
                                    
                                    # Detection count with icon
                                    st.info(f"🎯 Objects detected: {result['detection_count']}")
                                    
                                    # Show detection details with grouping for similar items
                                    if result['detections'] and any(d['waste_type'] not in ['No detection'] and 'Error' not in d['waste_type'] for d in result['detections']):
                                        st.write("**🔍 Detection Details:**")
                                        
                                        # Group detections by waste type
                                        waste_groups = {}
                                        for detection in result['detections']:
                                            if detection['waste_type'] not in ['No detection'] and 'Error' not in detection['waste_type']:
                                                waste_type = detection['waste_type']
                                                if waste_type not in waste_groups:
                                                    waste_groups[waste_type] = []
                                                waste_groups[waste_type].append(detection)
                                        
                                        # Display grouped results
                                        for waste_type, detections in waste_groups.items():
                                            recyclable_status = "♻️ Recyclable" if detections[0]['recyclable'] else "🗑️ Non-recyclable"
                                            count = len(detections)
                                            
                                            if count == 1:
                                                # Single detection - show as before
                                                conf = detections[0]['confidence']
                                                st.write(f"• **{waste_type}** ({conf:.1%}) - {recyclable_status}")
                                            else:
                                                # Multiple detections - show summary
                                                confidences = [d['confidence'] for d in detections]
                                                avg_conf = sum(confidences) / len(confidences)
                                                max_conf = max(confidences)
                                                min_conf = min(confidences)
                                                
                                                st.write(f"• **{waste_type}** ×{count} - {recyclable_status}")
                                                st.write(f"  → Avg: {avg_conf:.1%} | Range: {min_conf:.1%} - {max_conf:.1%}")
                                        
                                    elif result['detections']:
                                        # Show no detection or error message
                                        for detection in result['detections']:
                                            if 'Error' in detection['waste_type']:
                                                st.error(f"❌ {detection['waste_type']}")
                                            else:
                                                st.info(f"ℹ️ {detection['waste_type']}")
                                
                                # Add separator between images (except for the last one)
                                if idx < len(current_page_items) - 1:
                                    st.divider()
                        
                        # Navigation buttons with callback functions
                        if total_pages > 1:
                            nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 1])
                            
                            with nav_col1:
                                if st.button("⏮️ First", disabled=(current_page == 1), key="nav_first"):
                                    st.session_state.current_page = 1
                                    st.rerun()
                            
                            with nav_col2:
                                if st.button("◀️ Previous", disabled=(current_page == 1), key="nav_prev"):
                                    st.session_state.current_page = current_page - 1
                                    st.rerun()
                            
                            with nav_col4:
                                if st.button("Next ▶️", disabled=(current_page == total_pages), key="nav_next"):
                                    st.session_state.current_page = current_page + 1
                                    st.rerun()
                            
                            with nav_col5:
                                if st.button("Last ⏭️", disabled=(current_page == total_pages), key="nav_last"):
                                    st.session_state.current_page = total_pages
                                    st.rerun()
                    
                    with tab2:
                        # Detailed results table
                        st.dataframe(results_df, use_container_width=True)
                    
                    # Download Section (after tabs to access filtered_results)
                    st.subheader("📥 Download Options")
                    
                    # Information about different download types
                    with st.expander("ℹ️ Download Types Explained"):
                        st.markdown("""
                        **📊 PDF Reports**: Summary of processing
                        
                        **📥 Annotated Images**: Visual images with bounding boxes drawn for viewing/presentation
                        
                        **📥 Filtered Images**: Original images matching your current filter criteria
                        
                        **🎯 Training Dataset**: YOLO format dataset for machine learning training with:
                        - Original images (no annotations drawn)
                        - YOLO format label files (.txt) with normalized coordinates
                        - data.yaml configuration file
                        - Complete directory structure for training
                        
                        *Use Training Dataset to create new ML models or fine-tune existing ones!*
                        """)
                    
                    # Quick download buttons
                    download_col1, download_col2, download_col3, download_col4 = st.columns(4)
                    
                    with download_col1:
                        # PDF report download
                        if st.button("📊 Generate PDF Report", help="Create comprehensive PDF report with analysis", key="generate_pdf"):
                            with st.spinner("Generating PDF report..."):
                                try:
                                    pdf_data = create_pdf_report(image_results, stats, confidence_threshold)
                                    if pdf_data:
                                        st.session_state.pdf_report = {
                                            'data': pdf_data,
                                            'filename': f"waste_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                        }
                                        st.success("✅ PDF report generated successfully!")
                                    else:
                                        st.error("❌ Failed to generate PDF report")
                                except Exception as e:
                                    st.error(f"❌ Error generating PDF: {str(e)}")
                        
                        # Show download button if PDF is generated
                        if 'pdf_report' in st.session_state:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=st.session_state.pdf_report['data'],
                                file_name=st.session_state.pdf_report['filename'],
                                mime="application/pdf",
                                help="Download comprehensive PDF report with visualizations"
                            )
                    
                    with download_col2:
                        # Download all annotated images
                        if st.button("📥 Prepare Annotated Images", help="Create ZIP file with all annotated images", key="prep_annotated"):
                            with st.spinner("Creating ZIP file with annotated images..."):
                                zip_data = create_annotated_images_zip(image_results, model, confidence_threshold)
                                
                                st.session_state.annotated_zip = {
                                    'data': zip_data.getvalue(),
                                    'filename': f"annotated_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                                }
                                st.success(f"✅ ZIP file prepared! ({len(image_results)} images)")
                        
                        # Show download button if ZIP is prepared
                        if 'annotated_zip' in st.session_state:
                            st.download_button(
                                label="⬇️ Download Annotated ZIP",
                                data=st.session_state.annotated_zip['data'],
                                file_name=st.session_state.annotated_zip['filename'],
                                mime="application/zip",
                                type="primary"
                            )
                    
                    with download_col3:
                        # Show filtered images option only if filtering is applied
                        try:
                            filtered_count = len(filtered_results)
                            total_count = len(image_results)
                            
                            if filtered_count < total_count:
                                if st.button("📥 Prepare Filtered Images", help=f"Create ZIP with {filtered_count} filtered original images", key="prep_filtered"):
                                    with st.spinner("Creating ZIP file with filtered images..."):
                                        zip_data = create_filtered_images_zip(filtered_results, image_results)
                                        
                                        st.session_state.filtered_zip = {
                                            'data': zip_data.getvalue(),
                                            'filename': f"filtered_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                                        }
                                        st.success(f"✅ Filtered ZIP prepared! ({filtered_count} images)")
                                
                                # Show download button if ZIP is prepared
                                if 'filtered_zip' in st.session_state:
                                    st.download_button(
                                        label="⬇️ Download Filtered ZIP",
                                        data=st.session_state.filtered_zip['data'],
                                        file_name=st.session_state.filtered_zip['filename'],
                                        mime="application/zip",
                                        type="primary"
                                    )
                            else:
                                st.info("💡 Apply filters above to enable filtered image download")
                        except NameError:
                            # filtered_results not available (shouldn't happen, but safety net)
                            st.info("💡 Apply filters above to enable filtered image download")
                    
                    with download_col4:
                        # Export Training Dataset
                        if st.button("🎯 Prepare Training Dataset", help="Create YOLO format dataset for ML training", key="prep_training"):
                            with st.spinner("Creating YOLO training dataset..."):
                                # Use all image results for training dataset (not just filtered)
                                training_zip, successful_exports, failed_exports = create_yolo_training_dataset(
                                    image_results, model, confidence_threshold
                                )
                                
                                st.session_state.training_zip = {
                                    'data': training_zip.getvalue(),
                                    'filename': f"yolo_training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    'successful_exports': successful_exports,
                                    'failed_exports': failed_exports
                                }
                                
                                success_rate = (successful_exports / len(image_results) * 100) if len(image_results) > 0 else 0
                                st.success(f"✅ Training dataset prepared!")
                                st.info(f"📊 {successful_exports}/{len(image_results)} images exported ({success_rate:.1f}% success rate)")
                                
                                if failed_exports > 0:
                                    st.warning(f"⚠️ {failed_exports} images failed to export")
                        
                        # Show download button if training dataset is prepared
                        if 'training_zip' in st.session_state:
                            st.download_button(
                                label="⬇️ Download Training Dataset",
                                data=st.session_state.training_zip['data'],
                                file_name=st.session_state.training_zip['filename'],
                                mime="application/zip",
                                type="primary",
                                help="Download YOLO format dataset with images, labels, and data.yaml"
                            )
                            
                            # Show export statistics
                            if st.session_state.training_zip.get('successful_exports'):
                                with st.expander("📊 Dataset Statistics"):
                                    st.write(f"**Images Exported:** {st.session_state.training_zip['successful_exports']}")
                                    if st.session_state.training_zip['failed_exports'] > 0:
                                        st.write(f"**Failed Exports:** {st.session_state.training_zip['failed_exports']}")
                                    st.write(f"**Confidence Threshold:** {confidence_threshold:.2f}")
                                    
                                    # Show dataset structure
                                    st.write("**Dataset Structure:**")
                                    st.code("""
training_dataset/
├── images/           # Original images (JPG/PNG)
├── labels/           # YOLO format labels (.txt)
├── data.yaml         # Dataset configuration
├── README.md         # Usage instructions
└── export_stats.txt  # Export statistics
""", language="text")
    
    elif input_type == "History & Database":
        st.subheader("📚 Analysis History & Database")
        st.markdown("View and manage your saved analysis sessions and results")
        
        # Database statistics
        with st.expander("📊 Database Statistics"):
            stats = db.get_database_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sessions", stats['total_sessions'])
            with col2:
                st.metric("Total Images", stats['total_images'])
            with col3:
                st.metric("Total Detections", stats['total_detections'])
            with col4:
                st.metric("Database Size", f"{stats['database_size_mb']:.1f} MB")
            
            if stats['waste_distribution']:
                st.write("**Waste Type Distribution:**")
                for waste_type, count in stats['waste_distribution'].items():
                    st.write(f"• {waste_type}: {count}")
        
        # Get all sessions
        sessions = db.get_sessions()
        
        if not sessions:
            st.info("📝 No analysis sessions found. Process some images first!")
        else:
            # Session management
            st.subheader("🗂️ Analysis Sessions")
            
            # Session selector
            session_options = [f"{s['session_name']} ({s['created_at']}) - {s['total_images']} images" for s in sessions]
            selected_session_idx = st.selectbox(
                "Select a session to view:",
                range(len(session_options)),
                format_func=lambda x: session_options[x] if x < len(session_options) else "No sessions"
            )
            
            if selected_session_idx is not None and selected_session_idx < len(sessions):
                selected_session = sessions[selected_session_idx]
                session_id = selected_session['session_id']
                
                # Display session details
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Session:** {selected_session['session_name']}")
                    st.info(f"**Created:** {selected_session['created_at']}")
                    st.info(f"**Confidence Threshold:** {selected_session['confidence_threshold']:.0%}")
                
                with col2:
                    st.info(f"**Total Images:** {selected_session['total_images']}")
                    st.info(f"**Total Detections:** {selected_session['total_detections']}")
                    if selected_session['description']:
                        st.info(f"**Description:** {selected_session['description']}")
                
                # Session actions
                if st.button("🗑️ Delete Session", key=f"delete_{session_id}"):
                    success = db.delete_session(session_id)
                    if success:
                        st.success("Session deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete session")
                
                # Display images from session with enhanced interface
                images = db.get_session_images(session_id)
                if images:
                    st.subheader(f"📸 Session Images ({len(images)} total)")
                    
                    # Image display options
                    display_col1, display_col2 = st.columns(2)
                    with display_col1:
                        show_thumbnails = st.checkbox("Show Image Thumbnails", value=False)
                    with display_col2:
                        images_per_page = st.selectbox("Images per page:", [5, 10, 20, 50], index=1)
                    
                    # Pagination for images
                    total_pages = (len(images) + images_per_page - 1) // images_per_page
                    if total_pages > 1:
                        page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
                        start_idx = (page - 1) * images_per_page
                        end_idx = min(start_idx + images_per_page, len(images))
                        page_images = images[start_idx:end_idx]
                        st.info(f"Showing images {start_idx + 1}-{end_idx} of {len(images)}")
                    else:
                        page_images = images[:images_per_page]
                    
                    # Display images
                    for i, image_info in enumerate(page_images):
                        with st.container():
                            # Image header
                            img_col1, img_col2, img_col3 = st.columns([3, 1, 1])
                            with img_col1:
                                st.write(f"**📷 {image_info['filename']}**")
                            with img_col2:
                                st.write(f"🎯 {image_info['detection_count']} detections")
                            with img_col3:
                                file_size_mb = image_info['file_size'] / (1024 * 1024) if image_info['file_size'] else 0
                                st.write(f"📁 {file_size_mb:.1f} MB")
                            
                            # Image details and thumbnail
                            if show_thumbnails:
                                try:
                                    # Get image data from database
                                    image_data = db.get_image_data(image_info['image_id'])
                                    if image_data and image_data['annotated_image']:
                                        # Display thumbnail
                                        thumbnail_col1, thumbnail_col2 = st.columns([1, 2])
                                        with thumbnail_col1:
                                            from PIL import Image as PILImage
                                            
                                            # Convert BLOB to image
                                            img_bytes = io.BytesIO(image_data['annotated_image'])
                                            pil_img = PILImage.open(img_bytes)
                                            
                                            # Create thumbnail
                                            pil_img.thumbnail((200, 150), PILImage.Resampling.LANCZOS)
                                            st.image(pil_img, caption=f"Annotated: {image_info['filename']}")
                                        
                                        with thumbnail_col2:
                                            st.write(f"**Upload Time:** {image_info['uploaded_at']}")
                                            st.write(f"**Status:** {image_info['status'].title()}")
                                            
                                            # Show detection details
                                            if image_data['detections']:
                                                st.write("**Detections:**")
                                                for det in image_data['detections'][:3]:  # Show first 3
                                                    recyclable_icon = "♻️" if det['recyclable'] else "🗑️"
                                                    st.write(f"  • {recyclable_icon} {det['waste_type']} ({det['confidence']:.1%})")
                                                if len(image_data['detections']) > 3:
                                                    st.write(f"  ... and {len(image_data['detections']) - 3} more")
                                
                                except Exception as e:
                                    st.warning(f"Could not load thumbnail for {image_info['filename']}")
                            else:
                                # Just show basic info without thumbnail
                                st.write(f"📅 **Uploaded:** {image_info['uploaded_at']} | 📊 **Status:** {image_info['status'].title()}")
                            
                            st.divider()
                    
                    # Bulk actions for session
                    st.subheader("🔧 Session Actions")
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("📊 Export Session Data", key=f"export_{session_id}"):
                            # Create CSV export for this session
                            
                            # Get all detection data for this session
                            session_data = []
                            for img in images:
                                image_data = db.get_image_data(img['image_id'])
                                if image_data and image_data['detections']:
                                    for det in image_data['detections']:
                                        session_data.append({
                                            'filename': img['filename'],
                                            'waste_type': det['waste_type'],
                                            'confidence': det['confidence'],
                                            'recyclable': det['recyclable'],
                                            'uploaded_at': img['uploaded_at']
                                        })
                            
                            if session_data:
                                df = pd.DataFrame(session_data)
                                csv_data = df.to_csv(index=False)
                                
                                st.download_button(
                                    label="⬇️ Download Session CSV",
                                    data=csv_data,
                                    file_name=f"session_{selected_session['session_name']}_{selected_session['created_at'][:10]}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No detection data to export")
                    
                    with action_col2:
                        if st.button("🖼️ Download All Images", key=f"download_imgs_{session_id}"):
                            st.info("💡 Preparing image download... This may take a moment for large sessions.")
                            
                            # Create ZIP of all session images
                            try:
                                zip_buffer = io.BytesIO()
                                
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for img in images:
                                        image_data = db.get_image_data(img['image_id'])
                                        if image_data:
                                            # Add original image
                                            if image_data.get('original_image'):
                                                zip_file.writestr(f"original_{img['filename']}", image_data['original_image'])
                                            
                                            # Add annotated image
                                            if image_data.get('annotated_image'):
                                                base_name, ext = os.path.splitext(img['filename'])
                                                zip_file.writestr(f"annotated_{base_name}{ext}", image_data['annotated_image'])
                                
                                zip_buffer.seek(0)
                                
                                st.download_button(
                                    label="⬇️ Download Session Images ZIP",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"session_images_{selected_session['session_name']}_{selected_session['created_at'][:10]}.zip",
                                    mime="application/zip"
                                )
                            except Exception as e:
                                st.error(f"Failed to create image ZIP: {str(e)}")
                    
                    with action_col3:
                        # Session statistics
                        if st.button("📈 Session Analytics", key=f"analytics_{session_id}"):
                            st.subheader("📊 Session Analytics")
                            
                            # Calculate session statistics
                            total_detections = sum(img['detection_count'] for img in images)
                            successful_images = len([img for img in images if img['status'] == 'success'])
                            
                            # Detection distribution
                            waste_type_counts = {}
                            recyclable_count = 0
                            total_confidence = 0
                            confidence_count = 0
                            
                            for img in images:
                                image_data = db.get_image_data(img['image_id'])
                                if image_data and image_data['detections']:
                                    for det in image_data['detections']:
                                        waste_type = det['waste_type']
                                        waste_type_counts[waste_type] = waste_type_counts.get(waste_type, 0) + 1
                                        
                                        if det['recyclable']:
                                            recyclable_count += 1
                                        
                                        total_confidence += det['confidence']
                                        confidence_count += 1
                            
                            # Display analytics
                            analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
                            
                            with analytics_col1:
                                st.metric("Success Rate", f"{(successful_images / len(images) * 100):.1f}%" if images else "0%")
                                st.metric("Avg Detections/Image", f"{(total_detections / len(images)):.1f}" if images else "0")
                            
                            with analytics_col2:
                                st.metric("Total Detections", total_detections)
                                st.metric("Recyclable Items", recyclable_count)
                            
                            with analytics_col3:
                                avg_confidence = (total_confidence / confidence_count) if confidence_count > 0 else 0
                                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                                st.metric("Recyclability Rate", f"{(recyclable_count / total_detections * 100):.1f}%" if total_detections > 0 else "0%")
                            
                            # Waste type distribution chart
                            if waste_type_counts:
                                st.subheader("🗂️ Waste Type Distribution")
                                
                                chart_data = pd.DataFrame(list(waste_type_counts.items()), columns=['Waste Type', 'Count'])
                                fig = px.pie(chart_data, values='Count', names='Waste Type', title="Detection Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📝 No images found in this session.")
    
    else:  # Real-time Webcam
        st.subheader("📹 Real-time Webcam Analysis")
        
        # Mode selection
        webcam_mode = st.radio(
            "Choose detection mode:",
            ["🔴 Real-time Detection", "📸 Capture & Analyze"],
            horizontal=True,
            help="Real-time: Continuous detection (may be slower). Capture: Take photo and analyze (recommended for better performance)"
        )
        
        if webcam_mode == "📸 Capture & Analyze":
            # === CAPTURE MODE ===
            st.info("📸 **Professional Capture Mode** - Position waste item and click capture for detailed analysis")
            
            # WebRTC configuration for capture
            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            # Create capture preview
            ctx = webrtc_streamer(
                key="professional-capture",
                video_transformer_factory=CapturePreviewTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            capture_col1, capture_col2 = st.columns(2)
            
            with capture_col1:
                # Capture button
                if st.button("📷 Capture Image", type="primary", help="Take a photo for professional analysis"):
                    if ctx.video_transformer:
                        ctx.video_transformer.capture_image()
                        st.success("📸 Image captured! Processing...")
            
            with capture_col2:
                # Settings for capture mode
                use_human_filter_capture = st.checkbox(
                    "Filter out humans",
                    value=True,
                    help="Prevent human faces/bodies from being detected as waste"
                )
            
            # Process captured frame
            if ctx.video_transformer:
                captured_frame = ctx.video_transformer.get_captured_frame()
                
                if captured_frame is not None:
                    st.subheader("📊 Professional Analysis Results")
                    
                    with st.spinner("Analyzing captured image..."):
                        # Load models
                        waste_model, person_model = load_models()
                        
                        # Detect humans if filtering enabled
                        human_boxes = []
                        if use_human_filter_capture:
                            human_boxes = detect_humans(captured_frame, person_model, 0.3)
                        
                        # Run waste detection
                        results = predict_waste(captured_frame, waste_model, confidence_threshold)
                        
                        # Filter detections
                        valid_detections = []
                        if len(results.boxes) > 0:
                            filtered_boxes = []
                            for box in results.boxes:
                                # Get box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                detection_box = [x1, y1, x2, y2]
                                
                                # Check if detection overlaps with human
                                if use_human_filter_capture and overlaps_with_human(detection_box, human_boxes, 0.3):
                                    continue  # Skip this detection
                                
                                filtered_boxes.append(box)
                                
                                class_id = int(box.cls[0])
                                class_name = waste_model.names[class_id]
                                confidence = float(box.conf[0])
                                recyclable = class_name.lower() in ['paper', 'plastic', 'glass', 'metal', 'cardboard']
                                
                                valid_detections.append({
                                    'waste_type': class_name,
                                    'confidence': confidence,
                                    'recyclable': recyclable,
                                    'bbox': [x1, y1, x2, y2]
                                })
                        
                        # Show results
                        if valid_detections:
                            st.success(f"✅ Detected {len(valid_detections)} waste items!")
                            
                            # Create annotated image manually
                            annotated_img = captured_frame.copy()
                            
                            for i, detection in enumerate(valid_detections):
                                x1, y1, x2, y2 = detection['bbox']
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw bounding box
                                color = (0, 255, 0) if detection['recyclable'] else (255, 165, 0)
                                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw label
                                label = f"{detection['waste_type']} {detection['confidence']:.2f}"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                                            (x1 + label_size[0], y1), color, -1)
                                cv2.putText(annotated_img, label, (x1, y1 - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                            # Display results in two columns
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.image(annotated_img, caption="Detection Results", use_container_width=True)
                            
                            with result_col2:
                                st.subheader("🔍 Detection Details")
                                
                                # Group detections by waste type
                                waste_groups = {}
                                for detection in valid_detections:
                                    waste_type = detection['waste_type']
                                    if waste_type not in waste_groups:
                                        waste_groups[waste_type] = []
                                    waste_groups[waste_type].append(detection)
                                
                                # Display grouped results
                                for waste_type, detections in waste_groups.items():
                                    recyclable_icon = "♻️" if detections[0]['recyclable'] else "🗑️"
                                    recyclable_text = "Recyclable" if detections[0]['recyclable'] else "Non-recyclable"
                                    count = len(detections)
                                    
                                    with st.container():
                                        if count == 1:
                                            conf = detections[0]['confidence']
                                            st.write(f"**{waste_type}**")
                                            st.write(f"📊 **Confidence:** {conf:.1%}")
                                            st.progress(conf)
                                        else:
                                            confidences = [d['confidence'] for d in detections]
                                            avg_conf = sum(confidences) / len(confidences)
                                            max_conf = max(confidences)
                                            min_conf = min(confidences)
                                            
                                            st.write(f"**{waste_type}** (×{count} items)")
                                            st.write(f"📊 **Average Confidence:** {avg_conf:.1%}")
                                            st.write(f"📊 **Range:** {min_conf:.1%} - {max_conf:.1%}")
                                            st.progress(avg_conf)
                                        
                                        # Recyclability status
                                        if detections[0]['recyclable']:
                                            st.success(f"{recyclable_icon} {recyclable_text}")
                                        else:
                                            st.warning(f"{recyclable_icon} {recyclable_text}")
                                        
                                        st.write("---")
                            
                            # Summary statistics
                            recyclable_count = sum(1 for d in valid_detections if d['recyclable'])
                            non_recyclable_count = len(valid_detections) - recyclable_count
                            
                            st.subheader("📊 Analysis Summary")
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            
                            with summary_col1:
                                st.metric("Total Items", len(valid_detections))
                            
                            with summary_col2:
                                st.metric("♻️ Recyclable", recyclable_count)
                            
                            with summary_col3:
                                st.metric("🗑️ Non-recyclable", non_recyclable_count)
                            
                        else:
                            st.warning("⚠️ No waste items detected")
                            st.info("Try adjusting the position of items or lowering the confidence threshold")
                            
                            # Show original image
                            st.image(captured_frame, caption="Captured Image", use_container_width=True)
                    
                    # Reset capture frame
                    ctx.video_transformer.capture_frame = None
        
        else:
            # === REAL-TIME MODE ===
            st.info("🔴 **Professional Real-time Mode** - Advanced continuous detection with tracking")
            
            # Advanced webcam settings
            webcam_col1, webcam_col2 = st.columns(2)
            
            with webcam_col1:
                enable_human_filter = st.checkbox(
                    "Enable Human Detection Filter",
                    value=True,
                    help="Filters out detections that overlap with human faces/bodies to reduce false positives"
                )
                
                enable_tracking = st.checkbox(
                    "Enable Object Tracking",
                    value=True,
                    help="Maintains consistent detection of objects across frames for stable annotations"
                )
            
            with webcam_col2:
                human_confidence = st.slider(
                    "Human Detection Sensitivity",
                    min_value=0.1,
                    max_value=0.8,
                    value=0.3,
                    step=0.05,
                    help="Higher values = less sensitive human detection (fewer false positives from humans)"
                )
                
                # Performance optimization
                frame_skip = st.selectbox(
                    "Performance Mode",
                    [1, 2, 3, 5],
                    index=0,
                    format_func=lambda x: f"Process every {x} frame{'s' if x > 1 else ''}" + (" (Faster)" if x > 1 else " (Best quality)"),
                    help="Skip frames to improve FPS - higher numbers = better performance but less smooth detection"
                )
            
            # WebRTC configuration
            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            # Create video transformer
            ctx = webrtc_streamer(
                key="professional-realtime",
                video_transformer_factory=WasteDetectionTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            # Update settings in real-time
            if ctx.video_transformer:
                ctx.video_transformer.set_confidence_threshold(confidence_threshold)
                ctx.video_transformer.set_human_filtering(enable_human_filter)
                ctx.video_transformer.set_tracking(enable_tracking)
                ctx.video_transformer.set_frame_skip(frame_skip)
                ctx.video_transformer.human_detection_confidence = human_confidence
            
            # Status indicators
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                if enable_human_filter:
                    st.success("✅ Human Filter: ON")
                else:
                    st.warning("⚠️ Human Filter: OFF")
            
            with status_col2:
                if enable_tracking:
                    st.success("✅ Object Tracking: ON")
                else:
                    st.warning("⚠️ Object Tracking: OFF")
            
            with status_col3:
                st.info(f"📊 Processing: Every {frame_skip} frame{'s' if frame_skip > 1 else ''}")
            
            st.info("Click 'START' to begin professional real-time waste detection using your webcam")
        
        # Common instructions for both modes
        with st.expander("📋 Professional Usage Instructions"):
            st.write(f"""
            **{webcam_mode} Instructions:**
            
            {'**📸 Professional Capture Mode:**' if webcam_mode == "📸 Capture & Analyze" else '**🔴 Real-time Professional Mode:**'}
            1. Click 'START' to activate your webcam
            2. Allow browser access to your camera when prompted
            {'3. Position waste item in view and click "📷 Capture Image"' if webcam_mode == "📸 Capture & Analyze" else '3. Hold waste items in front of the camera for real-time detection'}
            {'4. View detailed analysis results with recyclability assessment' if webcam_mode == "📸 Capture & Analyze" else '4. Adjust advanced settings for optimal performance'}
            5. Click 'STOP' when finished
            
            **💡 Professional Detection Features:**
            - Advanced confidence scoring and analysis
            - Detailed recyclability assessment
            - Comprehensive detection statistics
            {'- Professional-grade image capture and processing' if webcam_mode == "📸 Capture & Analyze" else '- Real-time object tracking with persistent IDs'}
            
            **🚫 Human Detection Filter:**
            - Prevents faces/bodies from being detected as plastic
            - Automatically filters out human overlapping detections
            - Adjustable sensitivity for different environments
            
            **⚙️ Advanced Settings:**
            - **Confidence Threshold:** Higher = only high-confidence detections
            - **Human Filter:** Toggle to prevent human false positives
            {'- **Capture Quality:** Single-frame high-quality analysis' if webcam_mode == "📸 Capture & Analyze" else '- **Performance Mode:** Frame skipping for optimal speed'}
            """)
        
        # Performance comparison
        with st.expander("⚡ Professional Mode Comparison"):
            st.write("""
            **📸 Professional Capture Mode (Recommended):**
            - ✅ **Maximum Accuracy** - Full processing power on single frame
            - ✅ **Detailed Analysis** - Comprehensive detection statistics
            - ✅ **Professional Quality** - High-resolution capture and analysis
            - ✅ **Recyclability Assessment** - Complete material classification
            - ✅ **Export Capability** - Save results for professional reports
            
            **🔴 Real-time Professional Mode:**
            - ⚡ **Immediate Feedback** - See detections as you position items
            - 🎯 **Advanced Tracking** - Maintains object IDs across frames
            - 🔧 **Performance Controls** - Frame skipping and optimizations
            - 📊 **Continuous Analysis** - Stream processing for extended sessions
            - ⚠️ **Resource Intensive** - May impact performance on slower devices
            
            **Professional Recommendation:** Use Capture Mode for detailed analysis and documentation!
            """)
        
        # Professional features note
        st.markdown("---")
        st.info("💼 **Professional Features**: Advanced analytics, detailed reporting, and export capabilities available in full analysis modes above.")

def educational_game():
    """Educational waste sorting game functionality"""
    st.header("🎮 Educational Waste Sorting Game")
    st.markdown("**Learn waste classification through interactive gaming with points, achievements, and leaderboards!**")
    
    model = load_model()
    game_db = get_game_database()
    
    # Sidebar for user login
    st.sidebar.header("🎯 Player Portal")
    
    # User authentication
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
        st.session_state.current_game_session = None
    
    username = st.sidebar.text_input(
        "Enter your username:",
        placeholder="Your gaming name...",
        help="Choose a unique username to track your progress"
    )
    
    if username and st.sidebar.button("🚀 Start Playing!"):
        # Create or get user
        user_id = game_db.create_user(username)
        if user_id:
            st.session_state.user_data = game_db.get_user(username)
            st.session_state.current_game_session = game_db.start_game_session(user_id)
            st.sidebar.success(f"Welcome back, {username}!")
        else:
            st.sidebar.error("Failed to create/login user")
    
    if st.session_state.user_data:
        # Display current user info
        st.sidebar.success(f"👋 Welcome, {st.session_state.user_data['username']}!")
        
        # Game settings
        confidence_threshold = st.sidebar.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Higher values = more confident detections only"
        )
        
        # Display user stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Total Points", st.session_state.user_data['total_points'])
        
        with col2:
            st.metric("🎯 Items Scanned", st.session_state.user_data['items_scanned'])
        
        with col3:
            accuracy = 0
            if st.session_state.user_data['items_scanned'] > 0:
                accuracy = (st.session_state.user_data['correct_classifications'] / st.session_state.user_data['items_scanned']) * 100
            st.metric("📊 Accuracy", f"{accuracy:.1f}%")
        
        with col4:
            st.metric("🔥 Best Streak", st.session_state.user_data['best_streak'])
        
        st.write("---")
        
        # Game mode
        st.subheader("📸 Scan Your Waste Item")
        st.info("Position your waste item in the camera view and click 'Scan Item' to identify it!")
        
        # WebRTC camera setup
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        ctx = webrtc_streamer(
            key="waste-game",
            video_transformer_factory=GameCaptureTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Game controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📷 Scan Item", type="primary", use_container_width=True):
                if ctx.video_transformer:
                    ctx.video_transformer.capture_image()
                    st.session_state.scan_requested = True
                    st.success("📸 Item captured! Processing...")
        
        with col2:
            if st.button("🔄 Reset Game", use_container_width=True):
                if st.session_state.user_data:
                    st.session_state.current_game_session = game_db.start_game_session(
                        st.session_state.user_data['user_id']
                    )
                st.session_state.game_state = None
                st.success("Game reset!")
        
        # Process captured frame (simplified version)
        if ctx.video_transformer and hasattr(st.session_state, 'scan_requested'):
            captured_frame = ctx.video_transformer.get_captured_frame()
            
            if captured_frame is not None:
                st.subheader("🔍 Detection Results")
                
                with st.spinner("Analyzing waste item..."):
                    results = predict_waste(captured_frame, model, confidence_threshold)
                
                if len(results.boxes) > 0:
                    # Get the highest confidence detection
                    best_box = max(results.boxes, key=lambda x: float(x.conf[0]))
                    class_id = int(best_box.cls[0])
                    detected_waste = model.names[class_id]
                    confidence = float(best_box.conf[0])
                    
                    # Display detection result
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.success(f"✅ Detected: **{detected_waste}**")
                        st.info(f"📊 Confidence: {confidence:.1%}")
                        
                        # Show captured image
                        display_img = Image.fromarray(captured_frame)
                        max_width, max_height = 400, 300
                        display_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                        st.image(display_img, caption="Scanned Item")
                    
                    with col2:
                        st.subheader("🗑️ Choose the Correct Bin:")
                        st.write("**Which bin should this item go into?**")
                        
                        # Bin selection buttons
                        bin_choice = None
                        
                        if st.button("♻️ Recyclable Bin", key="recyclable", use_container_width=True):
                            bin_choice = "recyclable"
                        
                        if st.button("🌱 Organic Bin", key="organic", use_container_width=True):
                            bin_choice = "organic"
                        
                        if st.button("🗑️ General Waste Bin", key="general", use_container_width=True):
                            bin_choice = "general"
                        
                        # Process bin selection
                        if bin_choice:
                            correct_bin = get_correct_bin(detected_waste)
                            
                            # Record the action in database
                            result = game_db.record_action(
                                st.session_state.current_game_session,
                                st.session_state.user_data['user_id'],
                                detected_waste,
                                confidence,
                                bin_choice,
                                correct_bin
                            )
                            
                            # Display result
                            st.write("---")
                            if result['is_correct']:
                                st.success("🎉 **Correct!** Well done!")
                                points_msg = f"**+{result['points_awarded']} points**"
                                
                                if result['confidence_bonus']:
                                    points_msg += " (including confidence bonus!)"
                                
                                if result['streak_bonus'] > 0:
                                    points_msg += f" + **{result['streak_bonus']} streak bonus!**"
                                
                                st.success(points_msg)
                                
                                if result['current_streak'] > 1:
                                    st.info(f"🔥 {result['current_streak']} correct in a row!")
                            else:
                                st.error("❌ **Incorrect!** Try again!")
                                st.info(f"💡 **{detected_waste}** should go in the **{correct_bin}** bin")
                                if result['points_awarded'] == 0:
                                    st.warning("No points lost, but no points gained either!")
                            
                            # Check for achievements
                            new_achievements = game_db.check_and_award_achievements(
                                st.session_state.user_data['user_id']
                            )
                            
                            if new_achievements:
                                st.balloons()
                                for achievement in new_achievements:
                                    st.success(f"🏆 **Achievement Unlocked!** {achievement['icon']} {achievement['name']} (+{achievement['points_reward']} points)")
                            
                            # Update user data
                            st.session_state.user_data = game_db.get_user_by_id(
                                st.session_state.user_data['user_id']
                            )
                            
                            # Reset for next scan
                            ctx.video_transformer.capture_frame = None
                            if 'scan_requested' in st.session_state:
                                del st.session_state.scan_requested
                
                else:
                    st.warning("⚠️ No waste items detected!")
                    st.info("💡 Try positioning the item better and ensuring good lighting")
                    
                    # Reset for next scan
                    ctx.video_transformer.capture_frame = None
                    if 'scan_requested' in st.session_state:
                        del st.session_state.scan_requested
        
        # Leaderboard and achievements tabs
        st.write("---")
        tab1, tab2 = st.tabs(["🏆 Leaderboard", "🏅 Achievements"])
        
        with tab1:
            st.subheader("🏆 Leaderboard")
            
            leaderboard = game_db.get_leaderboard(10)
            
            if leaderboard:
                for player in leaderboard:
                    if player['rank'] <= 3:
                        # Top 3 get special styling
                        medals = ["🥇", "🥈", "🥉"]
                        medal = medals[player['rank'] - 1]
                        st.success(f"{medal} **{player['username']}** - {player['total_points']} points ({player['accuracy_rate']:.1f}% accuracy)")
                    else:
                        st.info(f"#{player['rank']} **{player['username']}** - {player['total_points']} points ({player['accuracy_rate']:.1f}% accuracy)")
            else:
                st.info("No players yet! Be the first to play!")
        
        with tab2:
            achievements = game_db.get_user_achievements(st.session_state.user_data['user_id'])
            
            if achievements:
                st.subheader("🏅 Your Achievements")
                for achievement in achievements:
                    st.success(f"{achievement['icon']} **{achievement['name']}** - {achievement['description']}")
            else:
                st.info("No achievements yet! Start playing to earn some!")
        
        # Game instructions
        with st.expander("📖 How to Play"):
            st.markdown("""
            ### 🎮 Game Rules:
            
            1. **📷 Scan Items**: Click "Scan Item" to capture and identify waste
            2. **🗑️ Choose Bin**: Select the correct waste bin for the detected item
            3. **🏆 Earn Points**: 
               - ✅ Correct classification: **+10 points**
               - ❌ Wrong classification: **-5 points** (minimum 0)
               - 🎯 High confidence bonus: **+5 points** (when confidence > 90%)
               - 🔥 Streak bonus: **+20 points** every 5 correct in a row
            
            ### 🗑️ Bin Types:
            - **♻️ Recyclable**: Plastic, Metal, Glass, Paper, Cardboard
            - **🌱 Organic**: Biodegradable waste
            - **🗑️ General**: Everything else
            
            ### 🏅 Achievements:
            Unlock special achievements by reaching milestones and earn bonus points!
            
            **Good luck and have fun sorting waste! 🌍♻️**
            """)
    
    else:
        # Welcome screen for new users
        st.info("👆 Please enter a username in the sidebar to start playing!")
        
        st.subheader("🌟 Game Features:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("🎯 **Point System**\nEarn points for correct classifications!")
        
        with col2:
            st.info("🏆 **Leaderboard**\nCompete with other players!")
        
        with col3:
            st.warning("🏅 **Achievements**\nUnlock special badges!")

def main():
    # Header with platform branding
    st.title("♻️ AI-Enhanced Waste Segregation for Sustainable Cities")
    st.markdown("**Comprehensive waste classification through professional analysis and educational gaming**")
    
    # Mode selection in main area with enhanced styling
    st.markdown("### 🚀 Choose Your Experience:")
    
    mode_col1, mode_col2 = st.columns(2)
    
    with mode_col1:
        if st.button("🔬 **Professional Dashboard**", 
                    help="Advanced analysis with batch processing, database storage, and detailed analytics", 
                    use_container_width=True, type="primary"):
            st.session_state.platform_mode = "professional"
    
    with mode_col2:
        if st.button("🎮 **Educational Game**", 
                    help="Learn waste sorting through interactive gaming with points and achievements", 
                    use_container_width=True, type="secondary"):
            st.session_state.platform_mode = "educational"
    
    # Initialize mode if not set
    if 'platform_mode' not in st.session_state:
        st.session_state.platform_mode = None
    
    # Mode descriptions
    st.markdown("---")
    desc_col1, desc_col2 = st.columns(2)
    
    with desc_col1:
        st.markdown("""
        **🔬 Professional Dashboard Features:**
        - 📊 Batch image processing and analysis
        - 💾 Database storage and session management
        - 📈 Advanced analytics and visualizations
        - 🔄 Real-time webcam detection
        - 📥 Export results and annotated images
        - 📚 Historical analysis and reporting
        """)
    
    with desc_col2:
        st.markdown("""
        **🎮 Educational Game Features:**
        - 🎯 Interactive point-based learning
        - 🏆 Leaderboards and competition
        - 🏅 Achievement system with badges
        - 📊 Personal progress tracking
        - 🎓 Learn correct waste sorting
        - 🌱 Environmental awareness building
        """)
    
    # Load selected mode
    if st.session_state.platform_mode == "professional":
        st.markdown("---")
        professional_dashboard()
    elif st.session_state.platform_mode == "educational":
        st.markdown("---")
        educational_game()
    else:
        # Welcome screen
        st.markdown("---")
        st.info("👆 **Select a mode above to get started!**")
        
        # Platform statistics (if databases exist)
        try:
            detection_db = get_detection_database()
            game_db = get_game_database()
            
            st.subheader("📊 Platform Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Get stats from both databases
            detection_stats = detection_db.get_database_stats()
            
            with col1:
                st.metric("🔬 Analysis Sessions", detection_stats.get('total_sessions', 0))
            
            with col2:
                st.metric("📸 Images Analyzed", detection_stats.get('total_images', 0))
            
            with col3:
                st.metric("🎯 Waste Items Detected", detection_stats.get('total_detections', 0))
            
            with col4:
                # Get game player count
                leaderboard = game_db.get_leaderboard(1000)  # Get all players
                st.metric("🎮 Game Players", len(leaderboard))
            
        except Exception as e:
            st.info("💡 **Getting started:** Choose a mode above to begin waste classification!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <small>
        🌍 <strong>AI-Enhanced Waste Segregation for Sustainable Cities</strong> - Combining AI-powered analysis with educational gaming<br>
        💡 Switch between Professional and Educational modes anytime using the buttons above
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()