from ultralytics import YOLO
import torch
import os
from datetime import datetime

def conservative_fine_tuning_fixed():
    """
    FIXED Conservative fine-tuning - Forces custom learning rate
    """
    
    print("="*60)
    print("CONSERVATIVE FINE-TUNING - FIXED VERSION")
    print("="*60)
    print("Issue: Previous script used lr=0.001 instead of 0.0001")
    print("Fix: Explicitly force optimizer and learning rate")
    print("Target: Use TRUE 0.0001 learning rate for gentle training")
    print("="*60)
    
    # Paths
    base_model = "/home/sengf/FYP2/models/fyp1/best.pt"
    data_config = "/home/sengf/FYP2/data/data.yaml"
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/home/sengf/FYP2/results/conservative_fixed_{timestamp}"
    
    print(f"Loading original FYP1 model: {base_model}")
    print(f"Results will be saved to: {results_dir}")
    
    # Load original FYP1 model
    model = YOLO(base_model)
    
    print("\n" + "="*40)
    print("FIXED TRAINING PARAMETERS")
    print("="*40)
    print("Optimizer: SGD (explicit, no auto)")
    print("Learning rate: 0.0001 (FORCED)")
    print("Momentum: 0.937 (standard)")
    print("Warmup epochs: 5")
    print("Patience: 20")
    print("Epochs: 40")
    print("="*40)
    
    try:
        # Fixed training with forced parameters
        results = model.train(
            data=data_config,
            project=results_dir,
            name="conservative_fixed",
            
            # Core parameters
            epochs=40,
            batch=16,
            imgsz=640,
            
            # FORCE specific optimizer (no auto)
            optimizer='SGD',       # Explicit optimizer
            lr0=0.0001,           # This will now be respected
            lrf=0.0001,           # Conservative final LR
            momentum=0.937,       # Standard momentum
            weight_decay=0.0005,  # Light regularization
            
            # Gentle training approach
            warmup_epochs=5,      # Gentle warmup
            warmup_bias_lr=0.001, # Conservative warmup
            warmup_momentum=0.8,
            
            # Patient training
            patience=20,          # More patience
            save_period=10,       # Save every 10 epochs
            
            # Conservative augmentation
            hsv_h=0.005,         # Very minimal color changes
            hsv_s=0.3,           
            hsv_v=0.2,
            degrees=3.0,         # Minimal rotation
            translate=0.05,      # Minimal translation
            scale=0.2,           # Minimal scaling
            shear=0.5,           # Minimal shearing
            flipud=0.0,          # No vertical flip
            fliplr=0.5,          # Only horizontal flip
            mosaic=0.5,          # Reduced mosaic
            mixup=0.0,           # No mixup
            
            # Stability
            cos_lr=False,        # Linear LR schedule for stability
            deterministic=True,  # Reproducible
            
            # Monitoring
            plots=True,
            verbose=True
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED - EVALUATING RESULTS")
        print("="*50)
        
        # Load best model
        best_model_path = f"{results_dir}/conservative_fixed/weights/best.pt"
        best_model = YOLO(best_model_path)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = best_model.val(data=data_config, split='test')
        
        # Results analysis
        print("\n" + "="*50)
        print("FIXED CONSERVATIVE RESULTS")
        print("="*50)
        print(f"Original baseline: 76.9% mAP50")
        print(f"Quick test (lr=0.001): 73.8% mAP50")
        print(f"Conservative FIXED (lr=0.0001): {test_results.box.map50:.3f} ({test_results.box.map50*100:.1f}%) mAP50")
        
        improvement = (test_results.box.map50 - 0.769) * 100
        
        if test_results.box.map50 > 0.80:
            print(f"🎯 EXCELLENT: {improvement:+.1f} points improvement!")
            print("Ready to push toward 85% target!")
        elif test_results.box.map50 > 0.769:
            print(f"✅ SUCCESS: {improvement:+.1f} points improvement!")
            print("Conservative approach works - can optimize further")
        elif test_results.box.map50 > 0.75:
            print(f"🔄 STABLE: {improvement:+.1f} points change")
            print("Need larger model or address class imbalance")
        else:
            print(f"❌ DECLINING: {improvement:+.1f} points")
            print("Fundamental issue - need different approach")
        
        # Class breakdown
        print(f"\nDetailed results:")
        print(f"Test mAP50: {test_results.box.map50:.3f}")
        print(f"Test mAP50-95: {test_results.box.map:.3f}")
        
        return test_results.box.map50, best_model_path
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None, None

if __name__ == "__main__":
    conservative_fine_tuning_fixed()
