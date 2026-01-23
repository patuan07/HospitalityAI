"""
Load and inspect Stage1_Binary.pth to determine the correct architecture
"""
import torch

checkpoint = torch.load('Stage1_Binary.pth', map_location='cpu')

print("Checkpoint structure:")
print(f"Type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"Keys: {list(checkpoint.keys())}")
    
    if 'model_state' in checkpoint:
        print("\nInspecting model_state:")
        state_dict = checkpoint['model_state']
        
        # Print first few layer names to understand architecture
        layer_names = list(state_dict.keys())[:10]
        print(f"\nFirst 10 layers:")
        for name in layer_names:
            print(f"  {name}: {state_dict[name].shape}")
        
        print(f"\nTotal layers: {len(state_dict)}")
        
        # Check for common architectures
        if any('resnet' in name.lower() or 'layer' in name for name in layer_names):
            print("\n✅ Likely a ResNet architecture")
        elif any('vgg' in name.lower() for name in layer_names):
            print("\n✅ Likely a VGG architecture")
        elif any('mobilenet' in name.lower() for name in layer_names):
            print("\n✅ Likely a MobileNet architecture")
        elif any('efficientnet' in name.lower() for name in layer_names):
            print("\n✅ Likely an EfficientNet architecture")
        else:
            print("\n⚠️  Unknown architecture")
            print("    Checking layer patterns...")
            
            # Check for sequential features
            if 'features.0.weight' in state_dict:
                print("    Found 'features' layers - possibly custom CNN")
            if 'fc.weight' in state_dict or 'classifier.weight' in state_dict:
                print("    Found classifier layer")
    
    if 'class_to_idx' in checkpoint:
        print(f"\nClass mapping: {checkpoint['class_to_idx']}")
