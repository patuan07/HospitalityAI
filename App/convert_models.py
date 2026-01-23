"""
Convert PyTorch models to mobile format for React Native app
This script converts the three AI models to mobile-friendly format
"""

import torch
from pathlib import Path
import sys

def convert_stage1_binary():
    """Convert Stage1_Binary.pth to mobile format"""
    print("\n" + "="*60)
    print("Converting Stage 1: Binary Classifier (Made/Unmade)")
    print("="*60)
    
    try:
        # Load the model
        model_path = 'Stage1_Binary.pth'
        print(f"Loading model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check if it's a state dict or full model
        if isinstance(checkpoint, dict):
            print("Detected state_dict format, loading model architecture...")
            # You may need to define your model architecture here
            # For now, we'll try to use the model if it has a 'model' key
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                print("State dict found, but need model architecture to load it")
                print("Skipping Stage1 - please provide model architecture")
                return False
            else:
                print("Unknown checkpoint format")
                return False
        else:
            model = checkpoint
        
        model.eval()
        
        # Create example input (standard image size)
        example_input = torch.rand(1, 3, 224, 224)
        
        # Trace the model
        print("Tracing model...")
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize for mobile
        print("Optimizing for mobile...")
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized_model = optimize_for_mobile(traced_model)
        
        # Save the mobile model
        output_path = "Stage1_Binary_mobile.ptl"
        optimized_model._save_for_lite_interpreter(output_path)
        print(f"✅ Success! Saved to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting Stage1_Binary: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_stage2_detection():
    """Convert Stage2_Detection.pt (YOLO) to mobile format"""
    print("\n" + "="*60)
    print("Converting Stage 2: Defect Detection (Items/Wrinkles/Untucked)")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        import os
        
        model_path = 'Stage2_Detection.pt'
        print(f"Loading YOLO model from {model_path}...")
        
        # Load YOLO model - convert to string to avoid Path issues on Windows
        model = YOLO(str(model_path))
        
        # Export to TorchScript format
        print("Exporting to TorchScript format...")
        export_path = model.export(format='torchscript', optimize=True)
        
        print(f"✅ Success! Exported to {export_path}")
        
        # Rename to simpler name
        import shutil
        final_path = "Stage2_Detection_mobile.torchscript"
        export_path_str = str(export_path) if hasattr(export_path, '__fspath__') else export_path
        if os.path.exists(export_path_str):
            shutil.move(export_path_str, final_path)
            print(f"✅ Renamed to {final_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting Stage2_Detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_stage3_alignment():
    """Convert Stage3_BedPillow.pt (YOLO) to mobile format"""
    print("\n" + "="*60)
    print("Converting Stage 3: Bed/Pillow Detection (for Alignment)")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        
        model_path = 'Stage3_BedPillow.pt'
        print(f"Loading YOLO model from {model_path}...")
        
        # Load YOLO model
        model = YOLO(model_path)
        
        # Export to TorchScript format
        print("Exporting to TorchScript format...")
        export_path = model.export(format='torchscript', optimize=True)
        
        print(f"✅ Success! Exported to {export_path}")
        
        # Rename to simpler name
        import shutil
        final_path = "Stage3_BedPillow_mobile.torchscript"
        if Path(export_path).exists():
            shutil.move(export_path, final_path)
            print(f"✅ Renamed to {final_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting Stage3_BedPillow: {e}")
        print("Make sure 'ultralytics' is installed: pip install ultralytics")
        return False


def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO'
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\nPlease install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    print("\n" + "="*60)
    print("Bed Quality Checker - Model Converter")
    print("="*60)
    print("This will convert your PyTorch models to mobile format")
    print("Compatible with react-native-pytorch-core")
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install dependencies first and try again.")
        return
    
    # Check if model files exist
    models = {
        'Stage1_Binary.pth': 'Binary Classifier',
        'Stage2_Detection.pt': 'Defect Detection',
        'Stage3_BedPillow.pt': 'Bed/Pillow Detection'
    }
    
    print("\nChecking model files...")
    missing_models = []
    for model_file, desc in models.items():
        if Path(model_file).exists():
            print(f"✅ Found {desc}: {model_file}")
        else:
            print(f"❌ Missing {desc}: {model_file}")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\n⚠️  Missing model files: {', '.join(missing_models)}")
        print("Please ensure all model files are in the project root directory.")
        return
    
    # Convert models
    results = []
    
    # Stage 1: Binary Classifier
    results.append(('Stage1_Binary', convert_stage1_binary()))
    
    # Stage 2: Defect Detection
    results.append(('Stage2_Detection', convert_stage2_detection()))
    
    # Stage 3: Alignment Detection
    results.append(('Stage3_BedPillow', convert_stage3_alignment()))
    
    # Summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status} - {name}")
    
    if all(success for _, success in results):
        print("\n🎉 All models converted successfully!")
        print("\nNext steps:")
        print("1. Copy the mobile models to your app's assets folder")
        print("2. Update ModelService.ts to use the new model paths")
        print("3. Build and run the app")
    else:
        print("\n⚠️  Some conversions failed. Check the errors above.")


if __name__ == "__main__":
    main()
