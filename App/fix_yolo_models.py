"""
Fix YOLO models trained on Linux to work on Windows
This script loads and re-saves the models to remove PosixPath issues
"""

import torch
import pickle
from pathlib import WindowsPath, PureWindowsPath

class PathFixer:
    """Helper to fix PosixPath to WindowsPath during unpickling"""
    
    @staticmethod
    def find_class(module, name):
        if module == 'posixpath':
            module = 'ntpath'
        elif name == 'PosixPath':
            return WindowsPath
        elif name == 'PurePosixPath':
            return PureWindowsPath
        return getattr(__import__(module, fromlist=[name]), name)


def fix_yolo_model(input_path, output_path):
    """Load and re-save a YOLO model to fix path issues"""
    print(f"Fixing {input_path}...")
    
    try:
        # Custom unpickler to handle PosixPath
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                return PathFixer.find_class(module, name)
        
        # Load the model with custom unpickler
        with open(input_path, 'rb') as f:
            checkpoint = CustomUnpickler(f).load()
        
        # Re-save the model
        torch.save(checkpoint, output_path)
        print(f"✅ Fixed! Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("YOLO Model Path Fixer")
    print("="*60)
    print("This fixes models trained on Linux to work on Windows\n")
    
    models_to_fix = [
        ('Stage2_Detection.pt', 'Stage2_Detection_fixed.pt'),
    ]
    
    success_count = 0
    for input_model, output_model in models_to_fix:
        if fix_yolo_model(input_model, output_model):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Fixed {success_count}/{len(models_to_fix)} models")
    print(f"{'='*60}")
    
    if success_count == len(models_to_fix):
        print("\n✅ All models fixed!")
        print("Now run convert_models.py again")


if __name__ == "__main__":
    main()
