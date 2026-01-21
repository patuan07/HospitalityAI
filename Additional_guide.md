# Additional Guide

This document provides detailed instructions on where and how to replace the baseline code with your trained models for the Hospitality AI Competition.

## Table of Contents

1. [Overview](#overview)
2. [Stage 1: Binary Classification](#stage-1-binary-classification)
3. [Stage 2: Multi-label Defect Classification](#stage-2-multi-label-defect-classification)
4. [Stage 3: Localization/Weak CAM](#stage-3-localizationweak-cam)
5. [Stage 4: Geometry Alignment](#stage-4-geometry-alignment)
6. [Stage 5: Robustness (Optional)](#stage-5-robustness-optional)
7. [Configuration & Dependencies](#configuration--dependencies)
8. [API Contract Requirements](#api-contract-requirements)
9. [Testing Your Integration](#testing-your-integration)

---

## Overview

The template provides baseline heuristics that work without heavy ML libraries. Students should replace these with trained models while maintaining the API contract structure.

### Key Principles:
- **Keep API contract stable** - Maintain response schema keys
- **Maintain function signatures** - Return proper dataclasses
- **Generate proper artifact URLs** - Create HTTP-accessible image URLs
- **Include debug information** - Provide helpful debug context

---

## Stage 1: Binary Classification

**File**: `backend/ml/stage1_binary.py`

### What to Replace

Replace the `predict_stage1()` function with your trained model inference.

### Current Implementation
The baseline uses edge-density heuristic:
```python
def predict_stage1(
    image_path: str,
    *,
    input_width: int = 640,
    edge_threshold: float = 0.085,
) -> Stage1Result:
```

### Replacement Example

```python
import torch
from torchvision import transforms
from PIL import Image

# Load your trained model (do this once, not per request)
model = torch.load("models/stage1_model.pth")
model.eval()

def predict_stage1(
    image_path: str,
    *,
    input_width: int = 224,  # Your model's input size
    model_path: str = "models/stage1_model.pth",
) -> Stage1Result:
    """Your trained Stage 1 model inference."""
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_width, input_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        prob = model(img_tensor).sigmoid().item()
    
    pred_made = prob >= 0.5
    
    return Stage1Result(
        prob_made=prob,
        pred_made=pred_made,
        debug={
            "model": "custom_trained",
            "note": "Replace with your model loading logic"
        },
    )
```

### Requirements
- Must return `Stage1Result` with:
  - `prob_made`: float (0-1)
  - `pred_made`: bool
  - `debug`: dict

---

## Stage 2: Multi-label Defect Classification

**File**: `backend/ml/stage2_classifier.py`

### What to Replace

Replace the `predict_stage2()` function with your trained multi-label classification model.

### Current Implementation
The baseline uses heuristics for each defect type:
- Wrinkles: edge density
- Stain: dark-blob ratio
- Pillow Misaligned: placeholder (Stage 4)
- Blanket Off Center: left-right intensity imbalance

### Replacement Example

```python
import torch
from torchvision import transforms
from PIL import Image

# Load your trained model
model = torch.load("models/stage2_model.pth")
model.eval()

# Keep labels stable for judge compatibility
DEFAULT_LABELS: List[str] = [
    "wrinkles",
    "stain",
    "pillow_misaligned",
    "blanket_off_center",
]

def predict_stage2(
    image_path: str,
    *,
    labels: List[str] | None = None,
    input_width: int = 224,
) -> Stage2Result:
    """Your trained multi-label defect classifier."""
    
    labels = labels or DEFAULT_LABELS
    img = Image.open(image_path).convert("RGB")
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((input_width, input_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze().tolist()
    
    # Map to label names
    probs_map = {
        "wrinkles": probs[0],
        "stain": probs[1],
        "pillow_misaligned": probs[2],
        "blanket_off_center": probs[3],
    }
    
    # Return only requested labels
    probs = {k: float(probs_map.get(k, 0.0)) for k in labels}
    
    return Stage2Result(
        probs=probs,
        debug={
            "model": "custom_multi_label_classifier",
            "note": "Replace with your trained model"
        },
    )
```

### Requirements
- Must return `Stage2Result` with:
  - `probs`: dict mapping label to probability (0-1)
  - `debug`: dict
- Keep label names stable: `"wrinkles"`, `"stain"`, `"pillow_misaligned"`, `"blanket_off_center"`

---

## Stage 3: Localization/Weak CAM

**File**: `backend/ml/stage3_weak_cam.py`

### What to Replace

Replace or enhance:
1. `compute_cam_heatmap_fallback()` - Replace with true Grad-CAM
2. `run_stage3_weak_cam()` - Customize heatmap generation

### Replacement Example with Grad-CAM

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load your trained model
model = torch.load("models/stage2_model.pth")
model.eval()

# Define target layer for Grad-CAM
target_layer = model.features[-1]  # Adjust based on your model

def compute_cam_heatmap_gradcam(
    img_bgr: np.ndarray,
    class_idx: int,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for a specific class."""
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_float = np.float32(img_rgb) / 255.0
    
    # Prepare input tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_rgb).unsqueeze(0)
    
    # Create Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[class_idx])
    grayscale_cam = grayscale_cam[0, :]  # Get single image
    
    # Resize to match image size
    heatmap = cv2.resize(grayscale_cam, (img_bgr.shape[1], img_bgr.shape[0]))
    
    return heatmap
```

### Requirements
- Must return heatmap as float array (0-1) with same HxW as image
- Must generate overlay images and return URLs
- Support both heatmap_path and overlay_path in response

---

## Stage 4: Geometry Alignment

**File**: `backend/ml/stage4_geometry.py`

### What to Replace

Replace the `stage4_alignment()` function with your geometry analysis model.

### Replacement Example

```python
def stage4_alignment(
    image_path: str,
    *,
    artifact_dir: str,
    base_url: str,
    input_width: int = 640,
    model_path: str = "models/stage4_model.pth",
) -> Stage4Result:
    """Your trained geometry alignment model."""
    
    # Load your model
    model = load_geometry_model(model_path)
    
    # Preprocess image
    img = cv2.imread(image_path)
    img_processed = preprocess_for_geometry(img)
    
    # Inference
    with torch.no_grad():
        alignment_score = model(img_processed).item()
    
    # Determine pass/fail
    alignment_pass = alignment_score >= 0.75
    
    # Generate artifacts (if your model outputs alignment visualization)
    artifacts = {}
    if hasattr(model, 'get_visualization'):
        aligned_img = model.get_visualization()
        aligned_path = os.path.join(artifact_dir, "stage4_aligned.png")
        cv2.imwrite(aligned_path, aligned_img)
        artifacts["stage4_aligned"] = _to_url(base_url, aligned_path, artifact_dir)
    
    return Stage4Result(
        alignment_score=alignment_score,
        alignment_pass=alignment_pass,
        debug={
            "model": "custom_geometry_model",
            "alignment_threshold": 0.75,
        },
        artifacts=artifacts,
    )
```

### Requirements
- Must return `Stage4Result` with:
  - `alignment_score`: float (0-1)
  - `alignment_pass`: bool
  - `debug`: dict
  - `artifacts`: dict mapping name to URL

---

## Stage 5: Robustness (Optional)

**File**: `backend/ml/stage5_robustness.py`

### What to Replace

Replace the `evaluate_robustness()` function with your custom robustness evaluation.

### Replacement Example

```python
def evaluate_robustness(
    image_path: str,
    *,
    artifact_dir: str,
    base_url: str,
    model_path: str = "models/robustness_model.pth",
) -> Stage5Result:
    """Your custom robustness evaluation."""
    
    # Load your robustness model
    robustness_model = load_robustness_model(model_path)
    
    # Generate augmented versions
    aug_images = generate_augmentations(image_path)
    
    # Run your robustness model on each augmentation
    scores = []
    for aug_image in aug_images:
        score = robustness_model(aug_image)
        scores.append(score)
    
    # Calculate final robustness score
    robustness_score = np.mean(scores)
    
    return Stage5Result(
        robustness_score=robustness_score,
        details={
            "model": "custom_robustness_evaluator",
            "per_augmentation": {
                f"aug_{i}": {"score": float(score)} 
                for i, score in enumerate(scores)
            },
        },
    )
```

### Requirements
- Must return `Stage5Result` with:
  - `robustness_score`: float (0-1)
  - `details`: dict with evaluation information

---

## Configuration & Dependencies

### Add Dependencies

Edit `backend/requirements.txt`:

```txt
# Add your ML dependencies
torch>=1.9.0
torchvision>=0.10.0
pytorch-grad-cam>=1.0.0
tensorflow>=2.6.0
transformers>=4.20.0
# ... other dependencies
```

### Configuration Settings

Edit `backend/app/settings.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... existing settings
    
    # Add your model paths
    MODEL_DIR: str = "models"
    STAGE1_MODEL_PATH: str = "models/stage1_model.pth"
    STAGE2_MODEL_PATH: str = "models/stage2_model.pth"
    STAGE3_MODEL_PATH: str = "models/stage3_model.pth"
    STAGE4_MODEL_PATH: str = "models/stage4_model.pth"
    STAGE5_MODEL_PATH: str = "models/stage5_model.pth"
    
    # Add other configuration
    CONFIDENCE_THRESHOLD: float = 0.5
    ALIGNMENT_THRESHOLD: float = 0.75

settings = Settings()
```

### Load Models Efficiently

Modify `backend/ml/pipeline.py` to load models once:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def load_stage1_model():
    """Load Stage 1 model once and cache it."""
    import torch
    model = torch.load(settings.STAGE1_MODEL_PATH)
    model.eval()
    return model

def run_pipeline(*, image_path: str, ...):
    # Use cached model
    stage1_model = load_stage1_model()
    # ... rest of pipeline
```

---

## API Contract Requirements

### Critical: Do NOT Change These Response Keys

From `backend/app/schemas.py`, these keys must remain stable:

```python
class AnalyzeResponse(BaseModel):
    image_id: str
    stages_run: List[str]
    stage1: Dict[str, Any]        # DO NOT CHANGE
    defects: Dict[str, float]     # DO NOT CHANGE
    localizations: List[DefectLocalization]
    alignment_score: float        # DO NOT CHANGE
    alignment_pass: bool          # DO NOT CHANGE
    alignment_debug: Dict[str, Any]
    stage5: Optional[Dict[str, Any]]
    artifacts: Dict[str, str]
```

### Stable Keys to Maintain

| Key | Type | Description |
|-----|------|-------------|
| `image_id` | str | Unique image identifier |
| `stages_run` | List[str] | List of stages executed |
| `stage1.prob_made` | float | Probability bed is made |
| `stage1.pred_made` | bool | Binary prediction |
| `defects.*` | float | Defect probabilities per label |
| `localizations` | List[DefectLocalization] | Heatmaps and overlays |
| `alignment_score` | float | Geometry alignment score |
| `alignment_pass` | bool | Pass/fail for alignment |

---

## Testing Your Integration

### 1. Run Backend Health Check

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit: `http://localhost:8000/health`

### 2. Test Single Stage

```python
# Test in Python
from ml.stage1_binary import predict_stage1
from ml.stage2_classifier import predict_stage2

result1 = predict_stage1("test_images/bed_01.jpg")
print(f"Stage 1: {result1}")

result2 = predict_stage2("test_images/bed_01.jpg")
print(f"Stage 2: {result2}")
```

### 3. Full Integration Test

```bash
# Start Streamlit
cd webapp_streamlit
source .venv/bin/activate
streamlit run app.py
```

Visit: `http://localhost:8501`

Upload a test image and verify all stages run correctly.

### 4. API Test

```python
import requests

files = {"image": open("test_images/bed_01.jpg", "rb")}
response = requests.post(
    "http://localhost:8000/analyze",
    files=files,
    params={"run_stage5": "false"}
)

print(response.json())
```

---

## Common Issues & Solutions

### 1. Model Loading Errors

**Problem**: Model doesn't load or crashes on inference
**Solution**: 
- Ensure models are in correct path
- Check input preprocessing (normalization, resizing)
- Verify tensor shapes match model expectations

### 2. Artifact URL Issues

**Problem**: Images not displaying in UI
**Solution**:
- Verify artifact_dir exists and is writable
- Ensure base_url is correct
- Check file permissions

### 3. API Response Errors

**Problem**: Response validation fails
**Solution**:
- Check all required keys are present
- Verify types match (float, bool, dict)
- Ensure lists and dicts have correct structure

### 4. Performance Issues

**Problem**: Slow inference
**Solution**:
- Load models once and cache
- Use GPU if available
- Optimize image preprocessing
- Consider batch processing

---

## Quick Checklist

Before submitting:

- [ ] Stage 1 returns `prob_made` and `pred_made`
- [ ] Stage 2 returns all required labels
- [ ] Stage 3 generates heatmaps/overlays with URLs
- [ ] Stage 4 returns alignment score and pass/fail
- [ ] Stage 5 works when enabled
- [ ] All artifact URLs are accessible
- [ ] Debug info is included in responses
- [ ] API response keys match contract
- [ ] Dependencies are in requirements.txt
- [ ] Backend starts without errors
- [ ] Streamlit UI displays results correctly

---

## Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **PyTorch Grad-CAM**: https://github.com/jacobgil/pytorch-grad-cam
- **Competition Rules**: See README.md

---

**Good luck with your competition submission! 🎯**
