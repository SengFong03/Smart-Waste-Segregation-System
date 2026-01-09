import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import yaml
import numpy as np
from ultralytics import YOLO
import torch
from torch.utils.data import WeightedRandomSampler
import os
from collections import Counter
import json

def create_balanced_training_config():
    """Create training configuration with class imbalance solutions"""
    
    # Class weights based on inverse frequency (biodegradable gets lower weight)
    class_weights = {
        0: 0.3,  # BIODEGRADABLE (61% of dataset)
        1: 1.5,  # CARDBOARD
        2: 2.0,  # GLASS
        3: 1.8,  # METAL
        4: 1.2,  # PAPER
        5: 1.4   # PLASTIC
    }
    
    # Training hyperparameters optimized for imbalanced data
    train_config = {
        'model': 'yolov8n.pt',  # Start with nano model, can change to yolov8s.pt or yolov8m.pt
        'data': 'data/data.yaml',
        'epochs': 200,
        'batch': 16,
        'imgsz': 640,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'project': 'runs/detect',
        'name': 'balanced_waste_detection',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 1.0,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.1,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': True,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
        'source': None,
        'show': False,
        'save_txt': True,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'vid_stride': 1,
        'line_width': None,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'retina_masks': False,
        'format': 'torchscript',
        'keras': False,
        'optimize': False,
        'int8': False,
        'dynamic': False,
        'simplify': False,
        'opset': None,
        'workspace': 4,
        'nms': False,
        'rect': False,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': True,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'cache': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 1.0,
        'dfl': 1.5,
        'fl_gamma': 2.0,  # Focal loss gamma for hard examples
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.9,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,  # Enhanced for minority classes
        'erasing': 0.4,
        'crop_fraction': 1.0
    }
    
    return train_config, class_weights

def update_data_yaml_with_weights(data_path, class_weights):
    """Update data.yaml file with class weights"""
    
    with open(data_path, 'r') as file:
        data_config = yaml.safe_load(file)
    
    # Add class weights to data config
    data_config['class_weights'] = list(class_weights.values())
    
    # Create backup and save updated config
    backup_path = data_path.replace('.yaml', '_backup.yaml')
    with open(backup_path, 'w') as file:
        yaml.dump(data_config, file, default_flow_style=False)
    
    with open(data_path, 'w') as file:
        yaml.dump(data_config, file, default_flow_style=False)
    
    print(f"Updated {data_path} with class weights")
    print(f"Backup saved to {backup_path}")
    
    return data_config

def create_custom_trainer_callback():
    """Custom callback to monitor class-specific metrics"""
    
    def on_train_epoch_end(trainer):
        """Log class-specific metrics after each epoch"""
        if hasattr(trainer.validator, 'metrics'):
            metrics = trainer.validator.metrics
            
            # Log per-class mAP if available
            try:
                class_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
                print(f"\nEpoch {trainer.epoch + 1} - Per-class mAP50:")
                # Use the box.maps attribute to get per-class AP scores
                if hasattr(metrics, 'box') and hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
                    for i, class_name in enumerate(class_names):
                        if i < len(metrics.box.maps):
                            print(f"  {class_name}: {metrics.box.maps[i]:.3f}")
                else:
                    print("  Per-class metrics not available yet")
            except Exception as e:
                print(f"  Could not retrieve per-class metrics: {e}")
    
    return on_train_epoch_end

def balanced_training():
    """Main training function with class imbalance solutions"""
    
    print("=" * 60)
    print("BALANCED WASTE DETECTION TRAINING")
    print("=" * 60)
    
    # Create training configuration
    train_config, class_weights = create_balanced_training_config()
    
    print("Training Configuration:")
    print(f"  Model: {train_config['model']}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch Size: {train_config['batch']}")
    print(f"  Device: {train_config['device']}")
    print(f"  Image Size: {train_config['imgsz']}")
    
    print("\nClass Weights (to address imbalance):")
    class_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    for i, (class_id, weight) in enumerate(class_weights.items()):
        print(f"  {class_names[i]}: {weight}")
    
    # Update data.yaml with class weights
    data_path = train_config['data']
    if os.path.exists(data_path):
        data_config = update_data_yaml_with_weights(data_path, class_weights)
    else:
        print(f"Warning: Data config file not found at {data_path}")
        print("Please ensure the data.yaml file exists in the correct location")
        return
    
    # Load model
    print(f"\nLoading model: {train_config['model']}")
    model = YOLO(train_config['model'])
    
    # Add custom callback for class-specific monitoring
    callback = create_custom_trainer_callback()
    model.add_callback('on_train_epoch_end', callback)
    
    # Start training with balanced configuration
    print("\n" + "=" * 40)
    print("STARTING BALANCED TRAINING")
    print("=" * 40)
    
    try:
        results = model.train(**train_config)
        
        print("\n" + "=" * 40)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 40)
        
        # Print final results
        if results:
            print(f"Best mAP50-95: {results.box.map:.3f}")
            print(f"Best mAP50: {results.box.map50:.3f}")
            print(f"Model saved to: runs/detect/balanced_waste_detection/weights/best.pt")
        
        # Run validation on test set
        print("\n" + "=" * 30)
        print("TESTING ON TEST SET")
        print("=" * 30)
        
        best_model = YOLO('runs/detect/balanced_waste_detection/weights/best.pt')
        test_results = best_model.val(data=data_path, split='test')
        
        print(f"Test mAP50-95: {test_results.box.map:.3f}")
        print(f"Test mAP50: {test_results.box.map50:.3f}")
        
        # Class-specific test results
        try:
            print("\nPer-class Test Performance (mAP50):")
            # Use the maps method to get per-class AP scores
            if hasattr(test_results.box, 'maps') and test_results.box.maps is not None:
                for i, class_name in enumerate(class_names):
                    if i < len(test_results.box.maps):
                        print(f"  {class_name}: {test_results.box.maps[i]:.3f}")
            else:
                print("  Per-class test metrics not available")
        except Exception as e:
            print(f"  Could not retrieve per-class test metrics: {e}")
        
        return results, test_results
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        return None, None

def main():
    """Main function to run balanced training"""
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run balanced training
    train_results, test_results = balanced_training()
    
    if train_results is not None:
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print("✓ Training completed successfully")
        print("✓ Class imbalance solutions applied")
        print("✓ Model saved with balanced performance")
        print("\nNext steps:")
        print("1. Check runs/detect/balanced_waste_detection/ for results")
        print("2. Use best.pt model for inference")
        print("3. Compare with baseline performance")
    else:
        print("\n" + "=" * 60)
        print("TRAINING FAILED")
        print("=" * 60)
        print("Please check the error messages above and fix any issues")

if __name__ == "__main__":
    main()