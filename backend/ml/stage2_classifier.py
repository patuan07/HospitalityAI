from __future__ import annotations
import os
import torch
from ultralytics import YOLO
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List
import cv2
import numpy as np

# BGR Color Mapping for different classes 
CLASS_COLORS = {
    "Items": (255, 0, 0),      # Blue
    "Untucked": (255, 0, 255),  # Yellow
    "Wrinkles": (0, 0, 255)     # Red
}
GRID_COLOR = (200, 200, 200)   # Light Gray for grid lines

@dataclass
class Stage2Result:
    probs: Dict[str, float]
    debug: Dict

DEFAULT_LABELS = ["Items", "Untucked", "Wrinkles"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/maxwell-guico/RobotRoarz/ai_competition_housekeeping_product_template/backend/ml/models/stage2_model.pt"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            return model
        except Exception as e:
            print(f"Error: {e}")
    return None

model = load_model()

def predict_stage2(image_path: str, labels: List[str] | None = None) -> Stage2Result:
    if model is None:
        return Stage2Result(probs={l: 0.0 for l in DEFAULT_LABELS}, debug={"error": "Model not loaded"})

    # 1. Run YOLO Inference
    results = model.predict(source=image_path, device=DEVICE, conf=0.1, verbose=False) 
    result = results[0] 
    
    # 2. Load ORIGINAL image for high-resolution drawing
    img = cv2.imread(image_path) 
    height, width, _ = img.shape
    
    # Calculate dynamic scaling factors based on resolution
    thickness = max(1, int(width / 2000))
    font_scale = width / 600  # Adjust this to make text bigger or smaller
    
    # 3. Draw Spatial Grid Overlay with Anti-Aliasing
    overlay = img.copy()
    grid_thickness = max(1, int(thickness / 2))
    
    # Vertical lines
    cv2.line(overlay, (int(width * 0.4), 0), (int(width * 0.4), height), GRID_COLOR, grid_thickness, cv2.LINE_AA)
    cv2.line(overlay, (int(width * 0.6), 0), (int(width * 0.6), height), GRID_COLOR, grid_thickness, cv2.LINE_AA)
    # Horizontal lines
    cv2.line(overlay, (0, int(height * 0.4)), (width, int(height * 0.4)), GRID_COLOR, grid_thickness, cv2.LINE_AA)
    cv2.line(overlay, (0, int(height * 0.6)), (width, int(height * 0.6)), GRID_COLOR, grid_thickness, cv2.LINE_AA)
    
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    target_labels = labels or DEFAULT_LABELS
    probs_map = {label: 0.0 for label in target_labels}
    detections_list = []

    if result.boxes is not None:
        for box in result.boxes:
            coords = box.xyxy[0].tolist() 
            x1, y1, x2, y2 = map(int, coords)
            conf = float(box.conf[0]) 
            label_name = result.names.get(int(box.cls[0]), "Unknown") 

            # Quadrant Logic 
            x_center, y_center = float(box.xywhn[0][0]), float(box.xywhn[0][1])
            if 0.4 <= x_center <= 0.6 and 0.4 <= y_center <= 0.6:
                loc = "Center"
            else:
                v_pos = "Top" if y_center < 0.5 else "Bottom"
                h_pos = "Left" if x_center < 0.5 else "Right"
                loc = f"{v_pos}-{h_pos}"
            
            color = CLASS_COLORS.get(label_name, (0, 255, 0))
            
            # 4. Draw Scaled Bounding Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # 5. Draw Readable Label with Background
            ui_text = f"{label_name} ({loc}) {conf:.2f}"
            (w, h), baseline = cv2.getTextSize(ui_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Ensure label stays within image bounds
            label_y = y1 if y1 - h - 15 > 0 else y1 + h + 15
            
            # [cite_start]Draw solid background for text [cite: 12]
            cv2.rectangle(img, (x1, label_y - h - 15), (x1 + w, label_y + baseline), color, -1)
            
            # [cite_start]Draw smooth white text [cite: 13]
            cv2.putText(img, ui_text, (x1, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            detections_list.append({
                "label": label_name, "confidence": conf,
                "location": loc, "box_2d": coords
            }) 
            if label_name in probs_map:
                probs_map[label_name] = max(probs_map[label_name], conf)

    # 6. Save with high-quality JPEG compression
    os.makedirs("static", exist_ok=True) 
    output_path = os.path.join("static", f"out_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) 

    return Stage2Result(probs=probs_map, debug={
        "detections": detections_list, "annotated_image_path": output_path
    })