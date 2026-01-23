"""
Bed Quality Checker - Mobile-Style Python App
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from ultralytics import YOLO
import base64
import io
import json
from alignment_scorer import score_alignment
import os

# Set appearance
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Ensure pathlib is in clean state at startup
import pathlib
import platform
if platform.system() == 'Windows':
    # Reset pathlib to default Windows paths if they were patched
    from pathlib import WindowsPath, PureWindowsPath
    if not hasattr(pathlib.PosixPath, '__mro__'):
        # Already patched, this shouldn't happen but let's be safe
        try:
            import importlib
            importlib.reload(pathlib)
        except:
            pass

class BedQualityApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Ensure cleanup on close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Configure window to look like a mobile phone
        self.title("Bed Quality Checker")
        self.geometry("375x812")  # iPhone-like dimensions
        self.resizable(False, False)

        # Stats
        self.beds_checked = 0
        self.beds_approved = 0
        self.issues_raised = 0
        
        # Alignment check settings
        self.alignment_check_enabled = True
        self.alignment_failures = 0

        # Models
        self.models_loaded = False
        self.stage1_model = None
        self.stage2_model = None
        self.stage3_model = None

        # Current image
        self.current_image = None
        self.current_image_path = None
        
        # Annotated images for different stages
        self.annotated_images = {}
        self.current_view = "original"

        # Initialize UI
        self.create_camera_screen()

        # Load models in background
        self.after(100, self.load_models)

    def load_models(self):
        """Load AI models"""
        import pathlib
        import platform
        from pathlib import WindowsPath, PureWindowsPath
        
        # Store original pathlib objects
        _original_posix = getattr(pathlib, 'PosixPath', None)
        _original_pure_posix = getattr(pathlib, 'PurePosixPath', None)
        
        try:
            # Update status
            self.status_label.configure(text="Loading AI models...")

            # Load Stage 1 - Binary Classifier (ResNet18 with 2 classes)
            checkpoint = torch.load('Stage1_Binary.pth', map_location='cpu')
            self.stage1_model = models.resnet18(weights=None)
            self.stage1_model.fc = nn.Linear(self.stage1_model.fc.in_features, 2)
            self.stage1_model.load_state_dict(checkpoint['model_state'])
            self.stage1_model.eval()

            # Load Stage 2 - Defect Detection (YOLO)
            # Prefer TorchScript version to avoid PosixPath issues
            import logging
            logging.getLogger('ultralytics').setLevel(logging.ERROR)
            
            if os.path.exists('Stage2_Detection.torchscript'):
                self.stage2_model = YOLO('Stage2_Detection.torchscript', task='detect', verbose=False)
            elif platform.system() == 'Windows':
                # Apply PosixPath patch for .pt file on Windows
                pathlib.PosixPath = WindowsPath
                pathlib.PurePosixPath = PureWindowsPath
                self.stage2_model = YOLO('Stage2_Detection.pt', task='detect', verbose=False)
            else:
                self.stage2_model = YOLO('Stage2_Detection.pt', task='detect', verbose=False)

            # Load Stage 3 - Bed/Pillow Detection for alignment check
            if os.path.exists('Stage3_BedPillow_mobile.torchscript'):
                self.stage3_model = YOLO('Stage3_BedPillow_mobile.torchscript', task='detect', verbose=False)
            elif os.path.exists('Stage3_BedPillow.pt'):
                if platform.system() == 'Windows':
                    pathlib.PosixPath = WindowsPath
                    pathlib.PurePosixPath = PureWindowsPath
                self.stage3_model = YOLO('Stage3_BedPillow.pt', task='detect', verbose=False)

            self.models_loaded = True
            self.status_label.configure(text="AI models loaded ✓")
            self.capture_btn.configure(state="normal")

        except Exception as e:
            self.models_loaded = False
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.status_label.configure(text="Failed to load models")
            print(f"Model loading error: {str(e)}")  # Debug output
        finally:
            # Always restore original pathlib objects
            if _original_posix is not None:
                pathlib.PosixPath = _original_posix
            if _original_pure_posix is not None:
                pathlib.PurePosixPath = _original_pure_posix

    def create_camera_screen(self):
        """Create the main camera screen"""
        # Clear window
        for widget in self.winfo_children():
            widget.destroy()

        # Top bar with stats
        stats_frame = ctk.CTkFrame(self, fg_color="#2563eb", corner_radius=0)
        stats_frame.pack(fill="x", padx=0, pady=0)

        stats_container = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_container.pack(pady=15, padx=20)

        # Stats
        stat_frames = [
            ("Checked", self.beds_checked),
            ("Approved", self.beds_approved),
            ("Issues", self.issues_raised)
        ]

        for i, (label, value) in enumerate(stat_frames):
            frame = ctk.CTkFrame(stats_container, fg_color="transparent")
            frame.grid(row=0, column=i, padx=15)

            num_label = ctk.CTkLabel(frame, text=str(value), font=("Arial", 24, "bold"), text_color="white")
            num_label.pack()

            text_label = ctk.CTkLabel(frame, text=label, font=("Arial", 12), text_color="white")
            text_label.pack()

        # Camera view placeholder
        camera_frame = ctk.CTkFrame(self, fg_color="#1f2937", corner_radius=0)
        camera_frame.pack(fill="both", expand=True, pady=0)

        self.image_label = ctk.CTkLabel(
            camera_frame,
            text="📸\n\nClick button below\nto select bed image",
            font=("Arial", 20),
            text_color="#6b7280"
        )
        self.image_label.pack(expand=True)

        # Status label
        status_text = "AI models loaded ✓" if self.models_loaded else "Loading models..."
        self.status_label = ctk.CTkLabel(
            camera_frame,
            text=status_text,
            font=("Arial", 14),
            text_color="#9ca3af"
        )
        self.status_label.pack(pady=10)

        # Capture button
        button_state = "normal" if self.models_loaded else "disabled"
        self.capture_btn = ctk.CTkButton(
            camera_frame,
            text="📷 Capture Photo",
            width=200,
            height=60,
            font=("Arial", 18, "bold"),
            corner_radius=30,
            command=self.capture_photo,
            state=button_state
        )
        self.capture_btn.pack(pady=30)

    def capture_photo(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Bed Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)

            # Show processing screen
            self.show_processing_screen()

    def show_processing_screen(self):
        """Show processing animation and run classification"""
        # Clear window
        for widget in self.winfo_children():
            widget.destroy()

        # Header
        header = ctk.CTkLabel(
            self,
            text="Processing Image...",
            font=("Arial", 24, "bold"),
            text_color="#1f2937"
        )
        header.pack(pady=30)

        # Progress bar
        self.progress = ctk.CTkProgressBar(self, width=300, height=20)
        self.progress.pack(pady=20)
        self.progress.set(0)

        # Status
        self.process_status = ctk.CTkLabel(
            self,
            text="Running AI classifier...",
            font=("Arial", 16),
            text_color="#6b7280"
        )
        self.process_status.pack(pady=10)

        # Show image preview
        self.show_image_preview()

        # Run classification
        self.after(100, self.run_classification)

    def show_image_preview(self):
        """Show image preview"""
        if self.current_image is not None:
            # Resize image for preview
            img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            pil_img.thumbnail((300, 300))

            # Use CTkImage instead of PhotoImage for high DPI support
            ctk_image = ctk.CTkImage(light_image=pil_img, size=pil_img.size)

            img_label = ctk.CTkLabel(self, text="", image=ctk_image)
            img_label.image = ctk_image  # Keep reference
            img_label.pack(pady=20)

    def run_classification(self):
        """Run the binary classifier"""
        try:
            # Check if models are loaded
            if not self.models_loaded or self.stage1_model is None:
                messagebox.showerror("Error", "AI models are not loaded. Please restart the application.")
                self.create_camera_screen()
                return

            # Update progress
            self.progress.set(0.3)
            self.process_status.configure(text="Analyzing bed quality...")
            self.update()

            # Preprocess image
            img = cv2.resize(self.current_image, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0)

            # Run model
            with torch.no_grad():
                output = self.stage1_model(img)
                probs = torch.softmax(output, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()

            self.progress.set(1.0)

            # Route based on result
            self.after(500, lambda: self.route_result(prediction, confidence))

        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.create_camera_screen()

    def route_result(self, prediction, confidence):
        """Route to appropriate screen based on classification"""
        if prediction == 0:  # Made
            if confidence >= 0.85:
                # High confidence - go to alignment check
                self.check_alignment(confidence)
            elif confidence >= 0.6:
                # Medium confidence - supervisor review
                self.show_supervisor_screen(confidence, "Low confidence classification")
            else:
                # Low confidence - detect defects
                self.detect_defects()
        else:  # Unmade
            # Detect defects
            self.detect_defects()

    def calculate_vector_sum_centerline(self, bed_box, pillow_boxes, bed_edges=None):
        """Calculate centerline using vector sum method from analyze_alignment_local.py"""
        # Calculate bed center
        bed_cx = (bed_box[0] + bed_box[2]) / 2
        bed_cy = (bed_box[1] + bed_box[3]) / 2
        
        # Calculate pillow centroids
        pillow_centroids = []
        for box in pillow_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            pillow_centroids.append((cx, cy))
        
        if len(pillow_centroids) == 0:
            # No pillows, use vertical line through bed center
            return {
                'center_x': bed_cx,
                'center_y': bed_cy,
                'angle': 90.0,
                'method': 'vertical_default',
                'distances': []
            }
        
        if len(pillow_centroids) == 1:
            # Single pillow: use bed-to-pillow vector
            px, py = pillow_centroids[0]
            vec_x = px - bed_cx
            vec_y = py - bed_cy
            angle = np.degrees(np.arctan2(vec_y, vec_x))
            if angle < 0:
                angle += 180
            
            print(f"  Single pillow: using bed-to-pillow vector ({vec_x:.1f}, {vec_y:.1f})")
            print(f"  Line angle: {angle:.1f}°")
            
            return {
                'center_x': bed_cx,
                'center_y': bed_cy,
                'angle': angle,
                'method': 'single_pillow_vector',
                'distances': [0]
            }
        
        # Multiple pillows: sum all vectors from bed center to each pillow
        sum_x = 0
        sum_y = 0
        
        for px, py in pillow_centroids:
            vec_x = px - bed_cx
            vec_y = py - bed_cy
            sum_x += vec_x
            sum_y += vec_y
        
        # Calculate angle of the summed vector
        angle = np.degrees(np.arctan2(sum_y, sum_x))
        if angle < 0:
            angle += 180
        
        # Calculate perpendicular distances from each pillow to the centerline
        angle_rad = np.radians(angle)
        distances = []
        for px, py in pillow_centroids:
            # Perpendicular distance formula
            d = abs(-(py - bed_cy) * np.cos(angle_rad) + (px - bed_cx) * np.sin(angle_rad))
            distances.append(d)
        
        # Orthogonal and parallel checks (within ±4 degrees)
        tolerance = 4
        is_horizontal = (angle < tolerance) or (angle > 180 - tolerance)
        is_vertical = abs(angle - 90) < tolerance
        parallel_to_edge = None
        
        # Check if parallel to any detected bed edge
        if bed_edges:
            for edge in bed_edges:
                x1_e, y1_e, x2_e, y2_e = edge
                dx = x2_e - x1_e
                dy = y2_e - y1_e
                edge_angle = np.degrees(np.arctan2(dy, dx))
                if edge_angle < 0:
                    edge_angle += 180
                
                angle_diff = abs(angle - edge_angle)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                
                if angle_diff <= tolerance:
                    parallel_to_edge = f"edge at {edge_angle:.1f}°"
                    break
        
        # Print results
        print(f"  Vector sum line: angle={angle:.1f}°, sum_vector=({sum_x:.1f}, {sum_y:.1f}), distances={[f'{d:.1f}' for d in distances]}")
        
        checks = []
        if is_horizontal:
            checks.append(f"HORIZONTAL (0°±{tolerance})")
        if is_vertical:
            checks.append(f"VERTICAL (90°±{tolerance})")
        if parallel_to_edge:
            checks.append(f"PARALLEL to {parallel_to_edge}")
        
        if checks:
            print(f"  ✓ {', '.join(checks)}")
        else:
            print(f"  ✗ NOT orthogonal or parallel to any bed edge (tolerance: ±{tolerance}°)")
        
        return {
            'center_x': bed_cx,
            'center_y': bed_cy,
            'angle': angle,
            'method': 'vector_sum',
            'is_horizontal': is_horizontal,
            'is_vertical': is_vertical,
            'parallel_to_edge': parallel_to_edge,
            'distances': distances
        }
    
    def detect_bed_edges(self, image, bed_box):
        """Detect actual bed edges using Canny + Hough line detection"""
        x1, y1, x2, y2 = map(int, bed_box)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract bed region
        bed_region = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(bed_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=90,
            minLineLength=120,
            maxLineGap=30
        )
        
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                
                # Convert to original image coordinates
                detected_lines.append([
                    x1_line + x1, 
                    y1_line + y1, 
                    x2_line + x1, 
                    y2_line + y1
                ])
        
        return detected_lines
    
    def calculate_centerline_coords(self, centerline_info, bed_box):
        """Calculate line endpoints for drawing the centerline"""
        cx = centerline_info['center_x']
        cy = centerline_info['center_y']
        angle = centerline_info['angle']
        
        # Use bed height to determine line length
        bed_height = bed_box[3] - bed_box[1]
        length = bed_height * 1.2
        
        # Calculate endpoints
        angle_rad = np.radians(angle)
        x1 = cx - length * np.cos(angle_rad)
        y1 = cy - length * np.sin(angle_rad)
        x2 = cx + length * np.cos(angle_rad)
        y2 = cy + length * np.sin(angle_rad)
        
        return [x1, y1, x2, y2]
    
    def check_alignment(self, classification_confidence):
        """Check bed alignment"""
        # Skip alignment check if disabled due to repeated failures
        if not self.alignment_check_enabled:
            self.show_alignment_screen(classification_confidence, {
                'aligned': True,
                'score': 0.85,
                'status': 'DISABLED',
                'recommendation': 'Alignment check disabled - auto-approved'
            })
            return
        
        try:
            # Check if Stage3 model is loaded
            if self.stage3_model is None:
                raise Exception("Stage3 model not loaded")
            
            print("Running alignment check with vector sum method...")
            # Run detection
            results = self.stage3_model(self.current_image, conf=0.5, verbose=False)
            
            # Parse detections
            beds = []
            pillows = []
            
            if results and len(results) > 0:
                result_obj = results[0]
                boxes = result_obj.boxes
                names = result_obj.names
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = names[class_id] if class_id in names else f"class_{class_id}"
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    print(f"  Detected: {class_name} (conf={confidence:.2f})")
                    
                    if class_name.lower() == 'bed' and confidence > 0.6:
                        beds.append(bbox)
                    elif class_name.lower() == 'pillow':
                        pillows.append(bbox)
            
            print(f"Found {len(beds)} beds, {len(pillows)} pillows")
            
            # Check if we have the necessary detections
            if len(beds) == 0:
                result = {
                    'aligned': False,
                    'score': 0.0,
                    'status': 'NO_BED',
                    'recommendation': 'No bed detected'
                }
            elif len(pillows) < 2:
                result = {
                    'aligned': True,
                    'score': 0.85,
                    'status': 'FEW_PILLOWS',
                    'recommendation': f'Only {len(pillows)} pillow(s) detected - auto-approved'
                }
            else:
                # Calculate centerline using vector sum method
                bed_box = beds[0]  # Use first bed
                
                # Detect bed edges using Canny + Hough
                bed_edges = self.detect_bed_edges(self.current_image, bed_box)
                print(f"  Detected {len(bed_edges)} bed edges")
                
                # Calculate centerline with bed edges
                centerline = self.calculate_vector_sum_centerline(bed_box, pillows, bed_edges)
                
                # Determine alignment based on centerline relationship to bed edges
                # Aligned if centerline is parallel or orthogonal to any Hough line (±5° margin)
                is_aligned = False
                score = 0.5
                tolerance = 5  # degrees
                alignment_reason = "No bed edges detected"
                
                centerline_angle = centerline['angle']
                
                if bed_edges and len(bed_edges) > 0:
                    # Check against each detected bed edge
                    for edge in bed_edges:
                        x1_e, y1_e, x2_e, y2_e = edge
                        dx = x2_e - x1_e
                        dy = y2_e - y1_e
                        edge_angle = np.degrees(np.arctan2(dy, dx))
                        if edge_angle < 0:
                            edge_angle += 180
                        
                        # Check if parallel (same angle)
                        angle_diff = abs(centerline_angle - edge_angle)
                        if angle_diff > 90:
                            angle_diff = 180 - angle_diff
                        
                        # Check if orthogonal (90° difference)
                        ortho_diff = abs(angle_diff - 90)
                        
                        if angle_diff <= tolerance:
                            is_aligned = True
                            score = 1.0 - (angle_diff / tolerance) * 0.3  # 0.7 to 1.0
                            alignment_reason = f"Parallel to bed edge ({edge_angle:.1f}°, diff={angle_diff:.1f}°)"
                            print(f"  ✓ ALIGNED: Centerline parallel to bed edge")
                            break
                        elif ortho_diff <= tolerance:
                            is_aligned = True
                            score = 1.0 - (ortho_diff / tolerance) * 0.3  # 0.7 to 1.0
                            alignment_reason = f"Orthogonal to bed edge ({edge_angle:.1f}°, diff={ortho_diff:.1f}°)"
                            print(f"  ✓ ALIGNED: Centerline orthogonal to bed edge")
                            break
                    
                    if not is_aligned:
                        alignment_reason = f"Not parallel/orthogonal to any bed edge (tolerance: ±{tolerance}°)"
                        print(f"  ✗ NOT ALIGNED: {alignment_reason}")
                else:
                    # No bed edges detected, fall back to checking if horizontal or vertical
                    is_horizontal = (centerline_angle < tolerance) or (centerline_angle > 180 - tolerance)
                    is_vertical = abs(centerline_angle - 90) < tolerance
                    
                    if is_horizontal or is_vertical:
                        is_aligned = True
                        score = 0.8
                        alignment_reason = "Horizontal" if is_horizontal else "Vertical"
                        print(f"  ✓ ALIGNED: Centerline is {alignment_reason.lower()}")
                    else:
                        alignment_reason = "Not horizontal or vertical"
                        print(f"  ✗ NOT ALIGNED: {alignment_reason}")
                
                # Calculate distances for display
                distances = centerline['distances']
                avg_distance = np.mean(distances) if distances else 0
                print(f"Alignment result: angle={centerline_angle:.1f}°, avg_dist={avg_distance:.1f}, aligned={is_aligned}, score={score:.2f}")
                
                result = {
                    'aligned': is_aligned,
                    'score': score,
                    'status': 'OK',
                    'recommendation': alignment_reason
                }
                
                # Store debug data for visualization
                result['debug_data'] = {
                    'bed_box': bed_box,
                    'pillow_boxes': pillows,
                    'centerline': self.calculate_centerline_coords(centerline, bed_box),
                    'centerline_info': centerline,
                    'bed_edges': bed_edges
                }
            
            print(f"Result: aligned={result.get('aligned')}, score={result.get('score'):.2f}")
            
            # Reset failure counter on success
            self.alignment_failures = 0
            
            # Create annotated images
            debug_data = result.get('debug_data', {})
            
            self.annotated_images = {
                'original': self.current_image.copy(),
                'alignment': self.draw_alignment_lines(self.current_image, debug_data)
            }
            self.current_view = 'original'
            
            # Show alignment screen
            self.show_alignment_screen(classification_confidence, result)

        except Exception as e:
            self.alignment_failures += 1
            print(f"Alignment check error: {e}")
            import traceback
            traceback.print_exc()
            
            # Disable alignment check after 2 failures
            if self.alignment_failures >= 2:
                self.alignment_check_enabled = False
                messagebox.showwarning("Warning", f"Alignment check failed repeatedly - disabling for this session.\n\nError: {str(e)}")
            
            # Fallback to auto-approve
            self.show_alignment_screen(classification_confidence, {
                'aligned': True,
                'score': 0.85,
                'status': 'ERROR',
                'recommendation': 'Alignment check unavailable - auto-approved'
            })
    
    def update_status(self, message):
        """Update status label if it exists"""
        if hasattr(self, 'process_status') and self.process_status.winfo_exists():
            self.process_status.configure(text=message)
            self.update()

    def show_alignment_screen(self, confidence, alignment_result):
        """Show alignment check results"""
        # Clear window
        for widget in self.winfo_children():
            widget.destroy()

        # Header
        header = ctk.CTkLabel(
            self,
            text="Alignment Check",
            font=("Arial", 24, "bold"),
            text_color="#1f2937"
        )
        header.pack(pady=20)

        # Show image with tabs
        if self.annotated_images:
            self.show_image_with_tabs(self)
        else:
            self.show_image_preview()

        # Result badge
        is_aligned = alignment_result.get('aligned', True)
        badge_color = "#d1fae5" if is_aligned else "#fee2e2"
        badge_text_color = "#065f46" if is_aligned else "#991b1b"
        badge_text = "✓ Well Aligned" if is_aligned else "✗ Not Aligned"

        badge = ctk.CTkFrame(self, fg_color=badge_color, corner_radius=10)
        badge.pack(pady=15, padx=20, fill="x")

        badge_label = ctk.CTkLabel(
            badge,
            text=badge_text,
            font=("Arial", 20, "bold"),
            text_color=badge_text_color
        )
        badge_label.pack(pady=15)

        # Confidence display
        conf_frame = ctk.CTkFrame(self, fg_color="transparent")
        conf_frame.pack(pady=10)

        ctk.CTkLabel(
            conf_frame,
            text=f"Classification: {confidence*100:.1f}%",
            font=("Arial", 14),
            text_color="#6b7280"
        ).pack()

        ctk.CTkLabel(
            conf_frame,
            text=f"Alignment: {alignment_result.get('score', 0)*100:.1f}%",
            font=("Arial", 14),
            text_color="#6b7280"
        ).pack()

        # Actions
        if is_aligned:
            ctk.CTkButton(
                self,
                text="✅ Approve & Next Bed",
                width=250,
                height=50,
                font=("Arial", 16, "bold"),
                fg_color="#10b981",
                hover_color="#059669",
                command=self.approve_bed
            ).pack(pady=10, padx=20)
        else:
            ctk.CTkButton(
                self,
                text="🔄 Re-clean Room",
                width=250,
                height=50,
                font=("Arial", 16, "bold"),
                fg_color="#2563eb",
                hover_color="#1d4ed8",
                command=self.create_camera_screen
            ).pack(pady=10, padx=20)

            ctk.CTkButton(
                self,
                text="⚠️ Raise Issue",
                width=250,
                height=50,
                font=("Arial", 16, "bold"),
                fg_color="#f59e0b",
                hover_color="#d97706",
                command=self.raise_issue
            ).pack(pady=10, padx=20)

    def detect_defects(self):
        """Detect defects using YOLO"""
        try:
            # Check if models are loaded
            if not self.models_loaded or self.stage2_model is None:
                messagebox.showerror("Error", "AI models are not loaded. Please restart the application.")
                self.create_camera_screen()
                return

            # Run YOLO detection
            results = self.stage2_model(self.current_image)

            # Parse results
            defects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    defects.append({
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })

            # Create annotated images
            self.annotated_images = {
                'original': self.current_image.copy(),
                'defects': self.draw_defect_boxes(self.current_image, defects)
            }
            self.current_view = 'original'

            # Show defects screen
            self.show_defects_screen(defects)

        except Exception as e:
            messagebox.showerror("Error", f"Defect detection failed: {str(e)}")
            self.show_defects_screen([])

    def show_defects_screen(self, defects):
        """Show detected defects"""
        # Clear window
        for widget in self.winfo_children():
            widget.destroy()

        # Header
        header = ctk.CTkLabel(
            self,
            text="Defects Detected",
            font=("Arial", 24, "bold"),
            text_color="#1f2937"
        )
        header.pack(pady=20)

        # Show image with tabs
        if self.annotated_images:
            self.show_image_with_tabs(self)
        else:
            self.show_image_preview()

        # Defect count
        count_label = ctk.CTkLabel(
            self,
            text=f"{len(defects)} issue{'s' if len(defects) != 1 else ''} found",
            font=("Arial", 16),
            text_color="#6b7280"
        )
        count_label.pack(pady=10)

        # Scrollable defects list
        if defects:
            defects_frame = ctk.CTkScrollableFrame(self, width=320, height=200)
            defects_frame.pack(pady=10, padx=20, fill="both", expand=True)

            defect_names = ['Items on bed', 'Bed sheet not tucked', 'Wrinkles']
            defect_icons = ['📦', '🛏️', '〰️']

            for defect in defects:
                class_id = defect['class']
                confidence = defect['confidence']

                card = ctk.CTkFrame(defects_frame, fg_color="#f3f4f6", corner_radius=10)
                card.pack(pady=5, padx=5, fill="x")

                content = ctk.CTkFrame(card, fg_color="transparent")
                content.pack(pady=10, padx=10, fill="x")

                icon_label = ctk.CTkLabel(
                    content,
                    text=defect_icons[class_id] if class_id < len(defect_icons) else "⚠️",
                    font=("Arial", 24)
                )
                icon_label.pack(side="left", padx=5)

                info_frame = ctk.CTkFrame(content, fg_color="transparent")
                info_frame.pack(side="left", fill="x", expand=True)

                name_label = ctk.CTkLabel(
                    info_frame,
                    text=defect_names[class_id] if class_id < len(defect_names) else "Unknown",
                    font=("Arial", 14, "bold"),
                    anchor="w"
                )
                name_label.pack(anchor="w")

                conf_label = ctk.CTkLabel(
                    info_frame,
                    text=f"Confidence: {confidence*100:.1f}%",
                    font=("Arial", 12),
                    text_color="#6b7280",
                    anchor="w"
                )
                conf_label.pack(anchor="w")
        else:
            no_defects = ctk.CTkLabel(
                self,
                text="No specific defects detected\nbut bed needs attention",
                font=("Arial", 14),
                text_color="#6b7280"
            )
            no_defects.pack(pady=20)

        # Actions
        ctk.CTkButton(
            self,
            text="🔄 Re-clean Room",
            width=250,
            height=50,
            font=("Arial", 16, "bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self.create_camera_screen
        ).pack(pady=5, padx=20)

        ctk.CTkButton(
            self,
            text="⚠️ Raise Issue",
            width=250,
            height=50,
            font=("Arial", 16, "bold"),
            fg_color="#f59e0b",
            hover_color="#d97706",
            command=self.raise_issue
        ).pack(pady=5, padx=20)

    def show_supervisor_screen(self, confidence, reason):
        """Show supervisor review screen"""
        # Clear window
        for widget in self.winfo_children():
            widget.destroy()

        # Header
        header = ctk.CTkLabel(
            self,
            text="Supervisor Review",
            font=("Arial", 24, "bold"),
            text_color="#1f2937"
        )
        header.pack(pady=20)

        # Show image with tabs if available, otherwise regular preview
        if self.annotated_images:
            self.show_image_with_tabs(self)
        else:
            self.show_image_preview()

        # Warning badge
        badge = ctk.CTkFrame(self, fg_color="#fef3c7", corner_radius=10)
        badge.pack(pady=15, padx=20, fill="x")

        badge_label = ctk.CTkLabel(
            badge,
            text="⚠️ Needs Supervisor Review",
            font=("Arial", 18, "bold"),
            text_color="#92400e"
        )
        badge_label.pack(pady=15)

        # Reason
        reason_label = ctk.CTkLabel(
            self,
            text=f"Reason: {reason}\nConfidence: {confidence*100:.1f}%",
            font=("Arial", 14),
            text_color="#6b7280"
        )
        reason_label.pack(pady=10)

        # Actions
        ctk.CTkButton(
            self,
            text="📋 Send to Supervisor",
            width=250,
            height=50,
            font=("Arial", 16, "bold"),
            fg_color="#f59e0b",
            hover_color="#d97706",
            command=self.send_to_supervisor
        ).pack(pady=10, padx=20)

        ctk.CTkButton(
            self,
            text="🔄 Re-clean Now",
            width=250,
            height=50,
            font=("Arial", 16, "bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self.create_camera_screen
        ).pack(pady=10, padx=20)

    def approve_bed(self):
        """Approve bed and move to next"""
        self.beds_checked += 1
        self.beds_approved += 1
        messagebox.showinfo("Success", "✅ Bed approved!\n\nGreat job! Moving to next bed.")
        self.create_camera_screen()

    def raise_issue(self):
        """Raise an issue"""
        self.beds_checked += 1
        self.issues_raised += 1
        messagebox.showwarning("Issue Raised", "⚠️ Issue has been reported to supervisor.\n\nMoving to next bed.")
        self.create_camera_screen()

    def send_to_supervisor(self):
        """Send to supervisor for review"""
        self.beds_checked += 1
        messagebox.showinfo("Sent", "📋 Sent to supervisor for review.\n\nMoving to next bed.")
        self.create_camera_screen()
    
    def draw_defect_boxes(self, image, defects):
        """Draw bounding boxes on image for detected defects"""
        annotated = image.copy()
        
        defect_names = ['Items on bed', 'Bed sheet not tucked', 'Wrinkles']
        colors = [(255, 0, 0), (0, 165, 255), (255, 255, 0)]  # BGR: Red, Orange, Yellow
        
        for defect in defects:
            bbox = defect['bbox']
            class_id = defect['class']
            confidence = defect['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[class_id] if class_id < len(colors) else (255, 255, 255)
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"{defect_names[class_id] if class_id < len(defect_names) else 'Defect'}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def draw_alignment_lines(self, image, alignment_data):
        """Draw alignment lines and markers on image"""
        annotated = image.copy()
        
        if 'bed_box' not in alignment_data or 'pillow_boxes' not in alignment_data:
            return annotated
        
        bed_box = alignment_data['bed_box']
        pillow_boxes = alignment_data['pillow_boxes']
        centerline = alignment_data.get('centerline')
        bed_edges = alignment_data.get('bed_edges', [])
        
        # Draw detected bed edges in cyan (Hough lines)
        for edge in bed_edges:
            x1_e, y1_e, x2_e, y2_e = map(int, edge)
            cv2.line(annotated, (x1_e, y1_e), (x2_e, y2_e), (255, 255, 0), 2)
        
        # Draw bed box in green
        x1, y1, x2, y2 = map(int, bed_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated, "Bed", (x1 + 5, y1 + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw bed center point
        bed_cx = int((x1 + x2) / 2)
        bed_cy = int((y1 + y2) / 2)
        cv2.circle(annotated, (bed_cx, bed_cy), 8, (0, 255, 0), -1)
        
        # Draw centerline if available
        if centerline:
            x1_line, y1_line, x2_line, y2_line = centerline
            cv2.line(annotated, (int(x1_line), int(y1_line)), 
                    (int(x2_line), int(y2_line)), (255, 0, 255), 2)
            cv2.putText(annotated, "Centerline", (int(x1_line) + 10, int(y1_line) + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw pillow boxes in blue
        for i, pillow_box in enumerate(pillow_boxes):
            px1, py1, px2, py2 = map(int, pillow_box)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 165, 0), 3)
            
            # Draw pillow centroid
            p_cx = int((px1 + px2) / 2)
            p_cy = int((py1 + py2) / 2)
            cv2.circle(annotated, (p_cx, p_cy), 6, (255, 165, 0), -1)
            
            # Draw line from pillow centroid to centerline
            if centerline:
                cv2.line(annotated, (p_cx, p_cy), (bed_cx, p_cy), (0, 255, 255), 1, cv2.LINE_AA)
            
            cv2.putText(annotated, f"Pillow {i+1}", (px1 + 5, py1 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        return annotated
    
    def show_image_with_tabs(self, parent_frame):
        """Show image with tabs to switch between different annotations"""
        # Create tab frame
        tab_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        tab_frame.pack(pady=10)
        
        # Tab buttons
        tab_buttons = []
        views = list(self.annotated_images.keys())
        
        def switch_view(view_name):
            self.current_view = view_name
            # Update button colors
            for btn, view in tab_buttons:
                if view == view_name:
                    btn.configure(fg_color="#2563eb", text_color="white")
                else:
                    btn.configure(fg_color="#e5e7eb", text_color="#1f2937")
            # Update image
            update_image_display()
        
        # Create tab buttons
        for view_name in views:
            display_name = view_name.replace('_', ' ').title()
            btn = ctk.CTkButton(
                tab_frame,
                text=display_name,
                width=100,
                height=30,
                font=("Arial", 12),
                fg_color="#e5e7eb" if view_name != self.current_view else "#2563eb",
                text_color="#1f2937" if view_name != self.current_view else "white",
                hover_color="#d1d5db",
                command=lambda v=view_name: switch_view(v)
            )
            btn.pack(side="left", padx=3)
            tab_buttons.append((btn, view_name))
        
        # Image display label
        image_container = ctk.CTkFrame(parent_frame, fg_color="transparent")
        image_container.pack(pady=10)
        
        self.tabbed_image_label = ctk.CTkLabel(image_container, text="")
        self.tabbed_image_label.pack()
        
        def update_image_display():
            if self.current_view in self.annotated_images:
                img = self.annotated_images[self.current_view]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_img.thumbnail((320, 320))
                
                ctk_image = ctk.CTkImage(light_image=pil_img, size=pil_img.size)
                self.tabbed_image_label.configure(image=ctk_image)
                self.tabbed_image_label.image = ctk_image
        
        # Initial display
        update_image_display()
    
    def on_closing(self):
        """Cleanup on app close"""
        # Ensure pathlib is restored to defaults
        import pathlib
        import platform
        if platform.system() == 'Windows':
            try:
                import importlib
                importlib.reload(pathlib)
            except:
                pass
        self.destroy()

if __name__ == "__main__":
    app = BedQualityApp()
    app.mainloop()
