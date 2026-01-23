"""
Flask API Server for Bed Quality Analysis
Runs the AI models as a REST API for the mobile app to call
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import torch
from ultralytics import YOLO
from alignment_scorer import AlignmentScorer
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app requests

# Load models
print("Loading AI models...")

# Stage 1: Binary Classifier
try:
    checkpoint = torch.load('Stage1_Binary.pth', map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        # This is a state dict - load with ResNet architecture
        print("Loading binary classifier from state_dict...")
        
        import torchvision.models as models
        # It's a ResNet18 based on inspection (122 layers, layer1.x pattern)
        binary_model = models.resnet18(weights=None)
        
        # Modify final layer for 2 classes (Made/Unmade)
        num_features = binary_model.fc.in_features
        binary_model.fc = torch.nn.Linear(num_features, 2)
        
        # Load the trained weights
        binary_model.load_state_dict(checkpoint['model_state'])
        binary_model.eval()
        
        class_mapping = checkpoint.get('class_to_idx', {'Made': 0, 'Unmade': 1})
        print(f"✅ Binary classifier loaded (ResNet18)")
        print(f"   Classes: {class_mapping}")
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        binary_model = checkpoint['model']
        binary_model.eval()
        print("✅ Binary classifier loaded")
    else:
        binary_model = checkpoint
        binary_model.eval()
        print("✅ Binary classifier loaded")
except Exception as e:
    print(f"⚠️  Binary classifier failed to load: {e}")
    print("   Will use demo mode for classification")
    import traceback
    traceback.print_exc()
    binary_model = None

# Stage 2: Defect Detection
try:
    # First, try the TorchScript version (fixed for Windows)
    if Path('Stage2_Detection.torchscript').exists():
        defect_model = YOLO('Stage2_Detection.torchscript')
        print("✅ Defect detection model (Stage2 TorchScript) loaded")
    else:
        # Try to load original with pathlib monkey-patch
        import pathlib
        from pathlib import WindowsPath, PureWindowsPath
        
        # Temporarily patch PosixPath
        original_posix = getattr(pathlib, 'PosixPath', None)
        try:
            pathlib.PosixPath = WindowsPath
            pathlib.PurePosixPath = PureWindowsPath
            
            defect_model = YOLO('Stage2_Detection.pt')
            print("✅ Defect detection model (Stage2 with pathlib patch) loaded")
        finally:
            # Restore original
            if original_posix:
                pathlib.PosixPath = original_posix
except Exception as e:
    print(f"⚠️  Stage2 model failed to load: {e}")
    try:
        # Fallback to Stage3 for defect detection
        print("   Trying Stage3 model as fallback...")
        defect_model = YOLO('Stage3_BedPillow.pt')
        print("✅ Using Stage3 model for defect detection (fallback)")
    except Exception as e2:
        print(f"⚠️  Stage3 fallback also failed: {e2}")
        defect_model = None

try:
    # Stage 3: Bed/Pillow Detection for alignment
    alignment_model = YOLO('Stage3_BedPillow.pt')
    alignment_scorer = AlignmentScorer()
    print("✅ Alignment model loaded")
except Exception as e:
    print(f"⚠️  Alignment model failed to load: {e}")
    alignment_model = None
    alignment_scorer = None


def decode_image(base64_string):
    """Decode base64 image to numpy array"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'binary_classifier': binary_model is not None,
            'defect_detection': defect_model is not None,
            'alignment_detection': alignment_model is not None
        }
    })


@app.route('/classify', methods=['POST'])
def classify_bed():
    """
    Classify if bed is made or unmade
    Input: base64 encoded image
    Output: {prediction: 0/1, confidence: float}
    """
    try:
        data = request.json
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Demo mode if model not loaded
        if binary_model is None:
            import random
            # Return a simulated result
            prediction = random.choice([0, 1])
            confidence = random.uniform(0.7, 0.95)
            return jsonify({
                'prediction': prediction,
                'confidence': float(confidence),
                'message': 'Made' if prediction == 0 else 'Unmade',
                'demo_mode': True,
                'warning': 'Using demo mode - binary classifier not loaded'
            })
        
        # Decode image
        img = decode_image(image_b64)
        
        # Preprocess
        img_resized = cv2.resize(img, (224, 224))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Inference
        with torch.no_grad():
            output = binary_model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'message': 'Made' if prediction == 0 else 'Unmade'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_defects', methods=['POST'])
def detect_defects():
    """
    Detect defects on bed (items, wrinkles, untucked)
    Input: base64 encoded image
    Output: {defects: [{class, confidence, bbox}]}
    """
    try:
        # Demo mode if model not loaded
        if defect_model is None:
            return jsonify({
                'defects': [],
                'count': 0,
                'demo_mode': True,
                'warning': 'Defect detection model not loaded'
            })
        
        data = request.json
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        img = decode_image(image_b64)
        
        # Run detection
        results = defect_model(img, conf=0.5, verbose=False)
        
        defects = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                class_name = result.names[class_id]
                
                defects.append({
                    'class': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [float(b) for b in box]
                })
        
        defect_names = {0: 'Items on bed', 1: 'Not tucked', 2: 'Wrinkles'}
        
        return jsonify({
            'defects': defects,
            'count': len(defects)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/check_alignment', methods=['POST'])
def check_alignment():
    """
    Check bed alignment using pillow detection
    Input: base64 encoded image
    Output: {is_aligned, score, confidence, details}
    """
    if alignment_model is None or alignment_scorer is None:
        return jsonify({'error': 'Alignment model not loaded'}), 500
    
    try:
        data = request.json
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        img = decode_image(image_b64)
        
        # Run detection
        results = alignment_model(img, conf=0.5, verbose=False)
        
        # Parse detections
        bed_boxes = []
        pillow_boxes = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                class_id = int(boxes.cls[i])
                class_name = result.names[class_id].lower()
                
                if 'bed' in class_name:
                    bed_boxes.append([float(b) for b in box])
                elif 'pillow' in class_name:
                    pillow_boxes.append([float(b) for b in box])
        
        # Check alignment
        if len(bed_boxes) == 0:
            return jsonify({
                'is_aligned': False,
                'score': 0.0,
                'confidence': 0.0,
                'message': 'No bed detected in image',
                'details': {
                    'beds_found': 0,
                    'pillows_found': len(pillow_boxes)
                }
            })
        
        # Use first bed detected
        bed_box = bed_boxes[0]
        
        # Calculate alignment score
        if len(pillow_boxes) >= 2:
            centerline = alignment_scorer.calculate_regression_centerline(bed_box, pillow_boxes)
            alignment_data = alignment_scorer.calculate_alignment_scores(
                bed_box, pillow_boxes, centerline, img
            )
            
            score = alignment_data.get('alignment_score', 0.0)
            is_aligned = score >= 70  # 70% threshold
            
            return jsonify({
                'is_aligned': is_aligned,
                'score': float(score),
                'confidence': float(alignment_data.get('confidence', 0.8)),
                'message': f"Alignment score: {score:.1f}%",
                'details': {
                    'beds_found': len(bed_boxes),
                    'pillows_found': len(pillow_boxes),
                    'centerline_angle': float(centerline.get('angle', 0))
                }
            })
        else:
            return jsonify({
                'is_aligned': False,
                'score': 0.0,
                'confidence': 0.5,
                'message': f'Not enough pillows detected ({len(pillow_boxes)} found, need 2+)',
                'details': {
                    'beds_found': len(bed_boxes),
                    'pillows_found': len(pillow_boxes)
                }
            })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_full', methods=['POST'])
def analyze_full():
    """
    Complete bed quality analysis
    Input: base64 encoded image
    Output: complete analysis with all three stages
    """
    try:
        data = request.json
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400
        
        result = {}
        
        # Stage 1: Binary classification
        if binary_model is not None:
            img = decode_image(image_b64)
            img_resized = cv2.resize(img, (224, 224))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            with torch.no_grad():
                output = binary_model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            result['classification'] = {
                'prediction': prediction,
                'confidence': float(confidence),
                'status': 'Made' if prediction == 0 else 'Unmade'
            }
        
        # Stage 2: Defect detection (if bed is made)
        if result.get('classification', {}).get('prediction') == 0 and defect_model:
            img = decode_image(image_b64)
            detection_results = defect_model(img, conf=0.5, verbose=False)
            defects = []
            
            if detection_results and len(detection_results) > 0:
                result_obj = detection_results[0]
                boxes = result_obj.boxes
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    
                    defects.append({
                        'class': class_id,
                        'confidence': confidence,
                        'bbox': [float(b) for b in box]
                    })
            
            result['defects'] = defects
        
        # Stage 3: Alignment check (if bed is made and no major defects)
        if result.get('classification', {}).get('prediction') == 0 and alignment_model:
            alignment_result = check_alignment()
            result['alignment'] = alignment_result.get_json()
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Bed Quality Checker API Server")
    print("="*60)
    print("Server starting on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health - Check server status")
    print("  POST /classify - Binary classification (made/unmade)")
    print("  POST /detect_defects - Defect detection")
    print("  POST /check_alignment - Alignment check")
    print("  POST /analyze_full - Complete analysis")
    print("="*60)
    print("\nMake sure your mobile device is on the same WiFi network!")
    print("Your computer's IP will be needed in the mobile app.\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
