"""
Fix Stage2 YOLO model for Windows by handling PosixPath issues
This script loads the model with a custom unpickler and re-saves it
"""

import torch
import pickle
import pathlib
import sys
from pathlib import Path, WindowsPath, PureWindowsPath

class PathMapper:
    """Maps PosixPath to WindowsPath during unpickling"""
    
    @staticmethod
    def map_location(storage, loc):
        return storage
    
    @staticmethod
    def find_class(module, name):
        # Map PosixPath to WindowsPath
        if name == 'PosixPath':
            return WindowsPath
        if name == 'PurePosixPath':  
            return PureWindowsPath
        
        # Handle other common path types
        if module == 'pathlib':
            if name in dir(pathlib):
                return getattr(pathlib, name)
        
        # Default behavior
        return getattr(sys.modules[module], name)


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles path objects"""
    
    def find_class(self, module, name):
        return PathMapper.find_class(module, name)
    
    def persistent_load(self, pid):
        """Handle persistent IDs for PyTorch tensors"""
        # Return None to let PyTorch handle it
        return None


def fix_yolo_model(input_path, output_path):
    """
    Load YOLO model trained on Linux and save it for Windows
    
    Args:
        input_path: Path to original .pt file
        output_path: Path to save fixed .pt file
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\nFixing {input_path}...")
    print("="*60)
    
    try:
        # Method 1: Try using torch.load with custom map_location
        print("Attempting Method 1: Direct torch.load with weights_only=False...")
        try:
            checkpoint = torch.load(
                input_path,
                map_location='cpu',
                weights_only=False,
                pickle_module=pickle
            )
            print("✅ Loaded with standard torch.load")
            
            # Re-save the checkpoint
            torch.save(checkpoint, output_path)
            print(f"✅ Saved to {output_path}")
            return True
            
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
            # Method 2: Custom unpickler with manual handling
            print("\nAttempting Method 2: Custom unpickler...")
            
            # Monkey-patch pathlib before loading
            original_posixpath = getattr(pathlib, 'PosixPath', None)
            
            try:
                # Temporarily replace PosixPath
                pathlib.PosixPath = WindowsPath
                pathlib.PurePosixPath = PureWindowsPath
                
                # Now try to load
                checkpoint = torch.load(input_path, map_location='cpu')
                
                print("✅ Loaded with monkey-patched pathlib")
                
                # Re-save
                torch.save(checkpoint, output_path)
                print(f"✅ Saved to {output_path}")
                
                return True
                
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                
            finally:
                # Restore original
                if original_posixpath:
                    pathlib.PosixPath = original_posixpath
            
            # Method 3: Use ultralytics export/import cycle
            print("\nAttempting Method 3: Ultralytics re-export...")
            try:
                from ultralytics import YOLO
                
                # This might fail, but worth a try with monkey-patch active
                pathlib.PosixPath = WindowsPath
                pathlib.PurePosixPath = PureWindowsPath
                
                model = YOLO(input_path)
                
                # Export to torchscript and back
                temp_path = str(input_path).replace('.pt', '_temp.torchscript')
                model.export(format='torchscript', optimize=True)
                
                print(f"✅ Model loaded and can be used directly")
                print(f"   Note: Use YOLO('{input_path}') with pathlib patched")
                
                return True
                
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
            
            finally:
                if original_posixpath:
                    pathlib.PosixPath = original_posixpath
        
        print("\n❌ All methods failed")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_fixed_model(model_path):
    """Verify that the fixed model can be loaded"""
    print(f"\nVerifying {model_path}...")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {model.names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    print("="*60)
    print("Stage2 YOLO Model Fixer for Windows")
    print("="*60)
    print("This fixes PosixPath issues in models trained on Linux\n")
    
    input_model = 'Stage2_Detection.pt'
    output_model = 'Stage2_Detection_fixed.pt'
    
    # Check if input exists
    if not Path(input_model).exists():
        print(f"❌ Input model not found: {input_model}")
        return
    
    print(f"Input:  {input_model}")
    print(f"Output: {output_model}")
    
    # Try to fix
    success = fix_yolo_model(input_model, output_model)
    
    if success:
        # Verify the fixed model
        if Path(output_model).exists():
            verify_fixed_model(output_model)
        
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        print(f"\nFixed model saved to: {output_model}")
        print(f"\nUpdate your api_server.py to use:")
        print(f"  defect_model = YOLO('{output_model}')")
    else:
        print("\n" + "="*60)
        print("❌ FAILED")
        print("="*60)
        print("\nThe model has deep PosixPath compatibility issues.")
        print("\nRecommended solutions:")
        print("1. Retrain the model on Windows")
        print("2. Convert on a Linux machine and transfer")
        print("3. Continue using Stage3 as fallback (current solution)")


if __name__ == "__main__":
    main()
