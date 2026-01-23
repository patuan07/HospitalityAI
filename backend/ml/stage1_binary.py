import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataclasses import dataclass
from typing import Dict

@dataclass
class Stage1Result:
    prob_made: float
    pred_made: bool
    debug: Dict

# Use the absolute path to ensure the file is found during the hackathon
MODEL_PATH = "/home/maxwell-guico/RobotRoarz/ai_competition_housekeeping_product_template/backend/ml/models/classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_labels():
    """Loads weights and extracts label mapping from the checkpoint."""
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            
            # 1. Map labels
            class_to_idx = checkpoint["class_to_idx"]
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            
            # 2. Initialize ResNet18
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
            
            # 3. Load weights from 'model_state'
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            return model, idx_to_class
        except Exception as e:
            print(f"❌ Error loading Stage 1: {e}")
    return None, {0: "Made", 1: "Unmade"}

# Global instances
MODEL, IDX_TO_CLASS = load_model_and_labels()

def predict_stage1(image_path: str, *, input_width: int = 224) -> Stage1Result:
    if not os.path.exists(image_path) or MODEL is None:
        return Stage1Result(prob_made=0.0, pred_made=False, debug={"error": "Model/Image missing"})

    # MIRROR YOUR SCRIPT'S TRANSFORMS
    # Note: No normalization is used here to match your provided script exactly.
    transform = transforms.Compose([
        transforms.Resize((input_width, input_width)),
        transforms.ToTensor()
    ])
    
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = MODEL(x)
        probs = torch.softmax(outputs, dim=1)
        
        # Determine which index is 'Made' based on your idx_to_class
        # This prevents 'flipped' results between different computers
        made_idx = next((i for i, label in IDX_TO_CLASS.items() if label.lower() == "made"), 0)
        prob_made = float(probs[0, made_idx].item())
        
        pred_idx = probs.argmax(dim=1).item()
        top_label = IDX_TO_CLASS[pred_idx]

    return Stage1Result(
        prob_made=prob_made,
        pred_made=(top_label.lower() == "made"),
        debug={
            "top_label": top_label,
            "confidence": float(probs[0, pred_idx].item()),
            "idx_to_class": IDX_TO_CLASS
        }
    )