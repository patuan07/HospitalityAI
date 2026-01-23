"""
Bed/Pillow/Blanket Detection using local YOLOv8 model with Linear Regression Centerline
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO


class ObjectDetector:
    """Detector using local YOLOv8 model"""

    def __init__(self, model_path='Stage3_BedPillow.pt', conf_threshold=0.5):
        """
        Initialize detector

        Args:
            model_path: Path to YOLOv8 model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_objects(self, image_path):
        """
        Detect beds, pillows, and blankets using local YOLOv8 model

        Args:
            image_path: Path to input image

        Returns:
            List of detections [(box, class_name, confidence), ...]
        """
        # Run YOLO inference
        results = self.model(str(image_path), conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        # Parse YOLO results
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates in xyxy format
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                
                # Get class name and confidence
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                class_name = result.names[class_id]
                
                detections.append(([x1, y1, x2, y2], class_name, confidence))
        
        return detections

    def calculate_centerline(self, bed_box, pillow_boxes, bed_edges=None):
        """
        Calculate line through bed center where pillow distances are equal
        
        Args:
            bed_box: [x1, y1, x2, y2] of bed
            pillow_boxes: List of [x1, y1, x2, y2] for each pillow
            bed_edges: List of detected bed edge lines (optional)
            
        Returns:
            Dictionary with centerline info: center_x, center_y, angle
        """
        # Calculate bed center
        bed_cx = (bed_box[0] + bed_box[2]) / 2
        bed_cy = (bed_box[1] + bed_box[3]) / 2
        
        return self.calculate_centerline_with_center(bed_cx, bed_cy, pillow_boxes, bed_edges)
    
    def calculate_centerline_with_center(self, bed_cx, bed_cy, pillow_boxes, bed_edges=None):
        """
        Calculate line through given center point where pillow distances are equal
        
        Args:
            bed_cx: X coordinate of bed center (or average of multiple beds)
            bed_cy: Y coordinate of bed center (or average of multiple beds)
            pillow_boxes: List of [x1, y1, x2, y2] for each pillow
            bed_edges: List of detected bed edge lines (optional)
            
        Returns:
            Dictionary with centerline info: center_x, center_y, angle
        """
        
        # Calculate pillow centroids
        pillow_centroids = []
        for box in pillow_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            pillow_centroids.append([cx, cy])
        
        if len(pillow_centroids) == 0:
            # No pillows, use vertical line through bed center
            return {
                'center_x': bed_cx,
                'center_y': bed_cy,
                'angle': 90.0,  # Vertical
                'method': 'vertical_default'
            }
        
        if len(pillow_centroids) == 1:
            # Single pillow: use bed-to-pillow vector
            px, py = pillow_centroids[0]
            vec_x = px - bed_cx
            vec_y = py - bed_cy
            
            # Calculate angle of the vector
            angle = np.degrees(np.arctan2(vec_y, vec_x))
            
            # Normalize to 0-180 range
            if angle < 0:
                angle += 180
            
            print(f"  Single pillow: using bed-to-pillow vector ({vec_x:.1f}, {vec_y:.1f})")
            print(f"  Line angle: {angle:.1f}°")
            
            return {
                'center_x': bed_cx,
                'center_y': bed_cy,
                'angle': angle,
                'method': 'single_pillow_vector'
            }
        
        # Multiple pillows: sum all vectors from bed center to each pillow
        sum_x = 0
        sum_y = 0
        
        for px, py in pillow_centroids:
            # Vector from bed center to pillow
            vec_x = px - bed_cx
            vec_y = py - bed_cy
            sum_x += vec_x
            sum_y += vec_y
        
        # Calculate angle of the summed vector
        angle = np.degrees(np.arctan2(sum_y, sum_x))
        
        # Normalize to 0-180 range
        if angle < 0:
            angle += 180
        
        # Calculate distances from each pillow to verify
        angle_rad = np.radians(angle)
        distances = []
        for px, py in pillow_centroids:
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
            'parallel_to_edge': parallel_to_edge
        }

    def detect_bed_edges(self, image, bed_box):
        """
        Detect actual bed edges using Canny + Hough line detection
        
        Args:
            image: Input image
            bed_box: [x1, y1, x2, y2] of bed bounding box
            
        Returns:
            List of detected lines [(x1, y1, x2, y2), ...] and their angles
        """
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
        line_angles = []
        if lines is not None:
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                
                # Calculate angle of this line
                dx = x2_line - x1_line
                dy = y2_line - y1_line
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < 0:
                    angle += 180
                
                line_angles.append(angle)
                
                # Convert to original image coordinates
                detected_lines.append([
                    x1_line + x1, 
                    y1_line + y1, 
                    x2_line + x1, 
                    y2_line + y1
                ])
        
        # Print angle distribution
        if line_angles:
            horizontal = sum(1 for a in line_angles if a < 30 or a > 150)
            vertical = sum(1 for a in line_angles if 60 < a < 120)
            diagonal = len(line_angles) - horizontal - vertical
            print(f"  Line angles: {horizontal} horizontal, {vertical} vertical, {diagonal} diagonal")
        
        return detected_lines

    def visualize_detections(self, image, detections, output_path):
        """
        Visualize detections with centerline
        
        Args:
            image: Input image
            detections: List of (box, class_name, confidence)
            output_path: Path to save visualization
        """
        vis_image = image.copy()
        
        # Separate beds and pillows
        beds = [(box, conf) for box, cls, conf in detections if cls == 'bed']
        pillows = [(box, conf) for box, cls, conf in detections if cls == 'pillow']
        
        # Draw all detections
        for box, class_name, conf in detections:
            x1, y1, x2, y2 = map(int, box)
            
            # Color based on class
            if class_name == 'bed':
                color = (255, 0, 0)  # Blue
            elif class_name == 'pillow':
                color = (0, 255, 0)  # Green
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(vis_image, (cx, cy), 8, color, -1)
            
            # Label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(vis_image, label, (cx + 10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate and draw centerline if we have bed and pillows
        if beds and pillows:
            # If multiple beds, calculate average center point
            if len(beds) > 1:
                print(f"  Multiple beds detected ({len(beds)}), using average center")
                total_cx = 0
                total_cy = 0
                for bed_box, _ in beds:
                    cx = (bed_box[0] + bed_box[2]) / 2
                    cy = (bed_box[1] + bed_box[3]) / 2
                    total_cx += cx
                    total_cy += cy
                avg_cx = total_cx / len(beds)
                avg_cy = total_cy / len(beds)
                
                # Create a combined bounding box for edge detection (use highest confidence bed)
                bed_box = max(beds, key=lambda x: x[1])[0]
                
                # Override bed center with average
                bed_box_with_avg_center = bed_box
            else:
                # Single bed, use its center
                bed_box = beds[0][0]
                avg_cx = (bed_box[0] + bed_box[2]) / 2
                avg_cy = (bed_box[1] + bed_box[3]) / 2
                bed_box_with_avg_center = bed_box
            
            pillow_boxes = [box for box, _ in pillows]
            
            # Detect actual bed edges using Canny + Hough (from highest confidence bed)
            bed_edges = self.detect_bed_edges(image, bed_box_with_avg_center)
            
            # Draw detected bed edges in cyan
            for edge in bed_edges:
                x1_e, y1_e, x2_e, y2_e = map(int, edge)
                cv2.line(vis_image, (x1_e, y1_e), (x2_e, y2_e), (255, 255, 0), 2)
            
            print(f"  Detected {len(bed_edges)} bed edges")
            
            # Calculate centerline using average center
            centerline = self.calculate_centerline_with_center(
                avg_cx, avg_cy, pillow_boxes, bed_edges
            )
            
            # Draw centerline
            cx = centerline['center_x']
            cy = centerline['center_y']
            angle_rad = np.radians(centerline['angle'])
            
            # Calculate line endpoints (long line across image)
            h, w = vis_image.shape[:2]
            length = max(w, h) * 1.5
            
            dx = np.cos(angle_rad) * length
            dy = np.sin(angle_rad) * length
            
            x1 = int(cx - dx)
            y1 = int(cy - dy)
            x2 = int(cx + dx)
            y2 = int(cy + dy)
            
            # Draw red centerline
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw bed center point
            cv2.circle(vis_image, (int(cx), int(cy)), 10, (0, 0, 255), -1)
            
            # Add text with angle
            text = f"Centerline: {centerline['angle']:.1f} deg ({centerline['method']})"
            cv2.putText(vis_image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save visualization
        cv2.imwrite(str(output_path), vis_image)
        print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Detect beds, pillows, and blankets')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLOv8 model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output', help='Output directory')

    args = parser.parse_args()

    detector = ObjectDetector(args.model, args.conf)
    
    source_path = Path(args.source)
    if source_path.is_file():
        print(f"Analyzing: {source_path}")
        detections = detector.detect_objects(source_path)
        
        print(f"\nDetected {len(detections)} objects:")
        for i, (box, class_name, conf) in enumerate(detections):
            print(f"{i+1}. {class_name} (confidence: {conf:.2f})")
            print(f"   Box: {box}")
        
        # Create visualization
        image = cv2.imread(str(source_path))
        if image is not None:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / f"{source_path.stem}_result.jpg"
            detector.visualize_detections(image, detections, output_path)
    
    elif source_path.is_dir():
        # Process directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = [f for f in source_path.iterdir()
                      if f.suffix.lower() in image_extensions]
        
        print(f"\nFound {len(image_files)} images in {source_path}")
        
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(image_files)}] Processing: {img_file.name}")
            print(f"{'='*60}")
            
            try:
                detections = detector.detect_objects(img_file)
                
                print(f"Detected {len(detections)} objects:")
                for j, (box, class_name, conf) in enumerate(detections):
                    print(f"{j+1}. {class_name} (confidence: {conf:.2f})")
                
                # Create visualization
                image = cv2.imread(str(img_file))
                if image is not None:
                    output_path = output_dir / f"{img_file.stem}_result.jpg"
                    detector.visualize_detections(image, detections, output_path)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Completed! Processed {len(image_files)} images")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
    
    else:
        print(f"ERROR: {args.source} is not a valid file or directory")


if __name__ == "__main__":
    main()
