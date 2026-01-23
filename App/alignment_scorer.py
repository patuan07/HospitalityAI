"""
Centroid + Edge Voting Alignment Scorer for Bed and Pillow Detection
Compares pillow centroids to bed's symmetry line + analyzes edge orientations within boxes
"""
import numpy as np
import cv2
import math
from typing import List, Tuple, Dict


class AlignmentScorer:
    """Calculates alignment scores by comparing pillow centroids to bed centerline"""

    def __init__(self, symmetry_tolerance_percent=5, edge_threshold_low=50, edge_threshold_high=150):
        """
        Initialize alignment scorer

        Args:
            symmetry_tolerance_percent: Percentage tolerance for symmetry (default 5%)
                                       e.g., 5% means pillows within 5% of bed width from center are "centered"
            edge_threshold_low: Lower threshold for Canny edge detection
            edge_threshold_high: Upper threshold for Canny edge detection
        """
        self.symmetry_tolerance_percent = symmetry_tolerance_percent
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high

    def calculate_centroid(self, box):
        """
        Calculate centroid (center point) of a bounding box

        Args:
            box: Bounding box [x1, y1, x2, y2]

        Returns:
            Tuple (centroid_x, centroid_y)
        """
        x1, y1, x2, y2 = box
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        return centroid_x, centroid_y

    def calculate_regression_centerline(self, bed_box, pillow_boxes):
        """
        Calculate centerline as the line of symmetry between pillow centroids,
        passing through the bed center point (like a mirror line)

        Args:
            bed_box: Bed bounding box [x1, y1, x2, y2]
            pillow_boxes: List of pillow bounding boxes

        Returns:
            Dict with centerline parameters: center_x, center_y, angle, width
        """
        # Calculate bed center
        bed_center_x = (bed_box[0] + bed_box[2]) / 2
        bed_center_y = (bed_box[1] + bed_box[3]) / 2
        bed_width = bed_box[2] - bed_box[0]
        
        if len(pillow_boxes) < 2:
            # Not enough pillows for regression, use vertical line through bed center
            return {
                'center_x': bed_center_x,
                'center_y': bed_center_y,
                'angle': 90,  # Vertical
                'width': bed_width,
                'box': bed_box
            }
        
        # Get pillow centroids
        pillow_centroids = []
        for pillow_box in pillow_boxes:
            cx, cy = self.calculate_centroid(pillow_box)
            pillow_centroids.append((cx, cy))
        
        if len(pillow_boxes) == 2:
            # For 2 pillows: centerline connects the bed center to the midpoint between pillows
            # This represents the symmetry axis
            p1_x, p1_y = pillow_centroids[0]
            p2_x, p2_y = pillow_centroids[1]
            
            # Midpoint between the two pillows
            mid_x = (p1_x + p2_x) / 2
            mid_y = (p1_y + p2_y) / 2
            
            # Vector from bed center to pillow midpoint
            dx = mid_x - bed_center_x
            dy = mid_y - bed_center_y
            
            # Calculate angle of this line
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Normalize to 0-180 range
            if angle < 0:
                angle += 180
            
            return {
                'center_x': bed_center_x,  # Line passes through bed center
                'center_y': bed_center_y,
                'angle': angle,
                'width': bed_width,
                'box': bed_box,
                'method': 'bed_center_to_pillow_midpoint'
            }
        
        # For more than 2 pillows: use PCA approach
        # Center the points around bed center
        centered_points = [(x - bed_center_x, y - bed_center_y) for x, y in pillow_centroids]
        X = np.array(centered_points)
        
        # Use PCA to find the direction perpendicular to pillow arrangement
        # The line with minimum variance (min eigenvalue) is the mirror/symmetry line
        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Get the direction with minimum eigenvalue (perpendicular to pillow spread)
        # This is the mirror line that evenly divides pillows on both sides
        min_idx = np.argmin(eigenvalues)
        direction_vector = eigenvectors[:, min_idx]
        
        # Calculate angle from direction vector
        angle = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
        
        # Normalize to 0-180 range
        if angle < 0:
            angle += 180
        
        return {
            'center_x': bed_center_x,
            'center_y': bed_center_y,
            'angle': angle,
            'width': bed_width,
            'box': bed_box,
            'method': 'pca'
        }

    def calculate_bed_centerline_with_orientation(self, image, bed_box):
        """
        Calculate the centerline of the bed based on detected edge orientation
        Uses Canny edge detection within the bed bounding box to find dominant parallel edges,
        then averages them to get the centerline

        Args:
            image: Input image
            bed_box: Bed bounding box [x1, y1, x2, y2]

        Returns:
            Dict with centerline parameters: center_x, center_y, angle, width
        """
        x1, y1, x2, y2 = map(int, bed_box)
        
        # Ensure coordinates are valid
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract bed ROI
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            # Fallback to vertical centerline
            return {
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'angle': 90,  # Vertical
                'width': x2 - x1,
                'box': bed_box
            }
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Edge detection
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=90,
            minLineLength=80,  # Increased from 50 to filter short lines
            maxLineGap=30  # Increased from 20 to connect longer segments
        )
        
        # Default values
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        dominant_angle = 90  # Default to vertical
        bed_width = x2 - x1
        
        if lines is not None and len(lines) > 0:
            # Convert lines to angle and position format
            line_data = []
            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                # Calculate angle in degrees (from horizontal)
                angle = np.degrees(np.arctan2(y2_l - y1_l, x2_l - x1_l))
                # Normalize to 0-180 range
                if angle < 0:
                    angle += 180
                
                # Calculate perpendicular distance from origin (for parallel line grouping)
                # Use the perpendicular distance to group parallel lines
                length = np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
                if length > 0:
                    # Get center point of line
                    cx = (x1_l + x2_l) / 2
                    cy = (y1_l + y2_l) / 2
                    
                    line_data.append({
                        'angle': angle,
                        'center': (cx, cy),
                        'length': length,
                        'line': line[0]
                    })
            
            if len(line_data) >= 2:
                # Find dominant angle by clustering
                angles = [ld['angle'] for ld in line_data]
                hist, bin_edges = np.histogram(angles, bins=18, range=(0, 180))
                dominant_bin = np.argmax(hist)
                dominant_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
                
                # Filter lines close to dominant angle (within 10 degrees)
                angle_tolerance = 10
                parallel_lines = [ld for ld in line_data 
                                if abs(ld['angle'] - dominant_angle) < angle_tolerance or 
                                   abs(ld['angle'] - dominant_angle - 180) < angle_tolerance]
                
                if len(parallel_lines) >= 2:
                    # Sort by position perpendicular to the line direction
                    # Calculate perpendicular coordinate for each line
                    perp_angle_rad = np.radians(dominant_angle + 90)
                    perp_positions = []
                    
                    for ld in parallel_lines:
                        cx, cy = ld['center']
                        # Project center onto perpendicular axis
                        perp_pos = cx * np.cos(perp_angle_rad) + cy * np.sin(perp_angle_rad)
                        perp_positions.append((perp_pos, ld))
                    
                    # Sort by perpendicular position
                    perp_positions.sort(key=lambda x: x[0])
                    
                    # Take the two most extreme lines (boundaries)
                    if len(perp_positions) >= 2:
                        left_boundary = perp_positions[0][1]
                        right_boundary = perp_positions[-1][1]
                        
                        # Average the two boundary positions to get centerline
                        left_cx, left_cy = left_boundary['center']
                        right_cx, right_cy = right_boundary['center']
                        
                        center_x = (left_cx + right_cx) / 2 + x1  # Add offset from ROI
                        center_y = (left_cy + right_cy) / 2 + y1
                        
                        # Calculate actual bed width from boundaries
                        boundary_distance = np.sqrt((right_cx - left_cx)**2 + (right_cy - left_cy)**2)
                        if boundary_distance > 0:
                            bed_width = boundary_distance
        
        # Convert lines to global coordinates for visualization
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                # Add ROI offset to convert to global coordinates
                detected_lines.append([x1_l + x1, y1_l + y1, x2_l + x1, y2_l + y1])
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'angle': dominant_angle,
            'width': bed_width,
            'box': bed_box,
            'detected_lines': detected_lines  # Store all detected Hough lines
        }

    def calculate_bed_centerline(self, bed_box):
        """
        Calculate the simple vertical centerline of the bed (for backward compatibility)

        Args:
            bed_box: Bed bounding box [x1, y1, x2, y2]

        Returns:
            x-coordinate of the bed's centerline
        """
        x1, _, x2, _ = bed_box
        return (x1 + x2) / 2

    def get_bed_width(self, bed_box):
        """Get width of the bed"""
        x1, _, x2, _ = bed_box
        return x2 - x1

    def analyze_edge_orientation(self, image, box):
        """
        Analyze edge orientations within a bounding box using Canny + Hough voting

        Args:
            image: Input image (should be blurred outside boxes)
            box: Bounding box [x1, y1, x2, y2]

        Returns:
            Dict with dominant angle, edge quality score, and edge alignment score
        """
        x1, y1, x2, y2 = map(int, box)

        # Ensure coordinates are valid
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Extract ROI
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return {
                'dominant_angle': None,
                'edge_quality': 0.0,
                'alignment_score': 0.0,
                'edge_count': 0
            }

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Edge detection
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)

        # Count edge pixels
        edge_count = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_quality = edge_count / total_pixels if total_pixels > 0 else 0

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=90,
            minLineLength=min(roi.shape[0], roi.shape[1]) // 4,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return {
                'dominant_angle': None,
                'edge_quality': float(edge_quality),
                'alignment_score': 0.0,
                'edge_count': int(edge_count)
            }

        # Calculate angles for all lines
        angles = []
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = math.degrees(math.atan2(y2_l - y1_l, x2_l - x1_l))
            # Normalize to [0, 180)
            angle = angle % 180
            angles.append(angle)

        # Vote for dominant angle (cluster around 0, 45, 90, 135 degrees)
        angle_bins = {0: [], 45: [], 90: [], 135: []}
        tolerance = 15  # degrees

        for angle in angles:
            for reference in [0, 45, 90, 135]:
                if abs(angle - reference) < tolerance or abs(angle - reference - 180) < tolerance:
                    angle_bins[reference].append(angle)
                    break

        # Find dominant orientation
        dominant_bin = max(angle_bins.keys(), key=lambda k: len(angle_bins[k]))
        dominant_angle = np.mean(angle_bins[dominant_bin]) if angle_bins[dominant_bin] else dominant_bin

        # Calculate alignment score: how well aligned are edges (higher is better)
        # Good alignment = most edges in dominant direction
        dominant_count = len(angle_bins[dominant_bin])
        total_lines = len(angles)
        alignment_score = dominant_count / total_lines if total_lines > 0 else 0

        return {
            'dominant_angle': float(dominant_angle),
            'edge_quality': float(edge_quality),
            'alignment_score': float(alignment_score),
            'edge_count': int(edge_count),
            'total_lines': int(total_lines)
        }

    def analyze_pillow_to_bed_alignment(self, bed_box, pillow_boxes, image=None):
        """
        Analyze alignment of pillows relative to bed centerline with edge voting

        Args:
            bed_box: Bed bounding box [x1, y1, x2, y2]
            pillow_boxes: List of pillow bounding boxes [[x1, y1, x2, y2], ...]
            image: Optional image for edge analysis

        Returns:
            Dict with alignment analysis and recommendations
        """
        if not pillow_boxes:
            return {
                'overall_score': 100,
                'status': 'NO_PILLOWS',
                'recommendation': 'No pillows detected',
                'details': []
            }

        # Calculate centerline using linear regression on pillow centroids
        # This line passes through bed center and acts as a mirror between pillows
        centerline_info = self.calculate_regression_centerline(bed_box, pillow_boxes)
        
        bed_centerline = centerline_info['center_x']
        bed_width = centerline_info['width']
        bed_x1 = bed_box[0]
        bed_x2 = bed_box[2]
        tolerance_distance = bed_width * (self.symmetry_tolerance_percent / 100)

        pillow_analyses = []
        total_offset = 0

        for i, pillow_box in enumerate(pillow_boxes):
            centroid_x, centroid_y = self.calculate_centroid(pillow_box)

            # Determine which side of centerline the pillow is on
            offset_from_centerline = centroid_x - bed_centerline
            
            # Calculate ideal position: midpoint between bed edge and centerline
            if offset_from_centerline < 0:
                # Left side: ideal is halfway between left edge and centerline
                ideal_x = (bed_x1 + bed_centerline) / 2
                position = 'LEFT_OF_CENTER'
            else:
                # Right side: ideal is halfway between centerline and right edge
                ideal_x = (bed_centerline + bed_x2) / 2
                position = 'RIGHT_OF_CENTER'
            
            # Calculate offset from ideal position
            offset = centroid_x - ideal_x
            offset_percent = (abs(offset) / (bed_width / 2)) * 100  # Percentage of half-bed width

            # Determine status based on offset from ideal position
            if abs(offset) <= tolerance_distance:
                status = 'GOOD'
            else:
                status = 'NEEDS_ADJUSTMENT'

            # Calculate individual pillow score (100% if at ideal position, decreases with offset)
            pillow_score = max(0, 100 - offset_percent * 2)  # 2x penalty for offset

            # Analyze edge orientation if image provided
            edge_analysis = None
            if image is not None:
                edge_analysis = self.analyze_edge_orientation(image, pillow_box)
                # Boost score if edges are well-aligned
                if edge_analysis['alignment_score'] > 0.7:
                    pillow_score = min(100, pillow_score * 1.1)  # 10% bonus for good edge alignment

            pillow_data = {
                'pillow_id': int(i),
                'centroid': [float(centroid_x), float(centroid_y)],  # Store as list of floats for JSON
                'ideal_position': float(ideal_x),
                'offset_from_ideal': float(offset),
                'offset_from_center': float(offset_from_centerline),  # Keep for reference
                'offset_percent': float(offset_percent),
                'position': position,
                'status': status,
                'score': float(pillow_score)
            }

            if edge_analysis:
                pillow_data['edge_analysis'] = edge_analysis

            pillow_analyses.append(pillow_data)

            total_offset += abs(offset)

        # Calculate overall score
        avg_score = np.mean([p['score'] for p in pillow_analyses])

        # Check for symmetry if multiple pillows
        symmetry_score = 100
        symmetry_status = 'N/A'

        if len(pillow_boxes) == 2:
            # For 2 pillows, check if they're symmetrically placed
            left_pillow = min(pillow_analyses, key=lambda p: p['centroid'][0])
            right_pillow = max(pillow_analyses, key=lambda p: p['centroid'][0])

            # Both should be at their ideal positions (equidistant from center on each side)
            left_offset_from_ideal = abs(left_pillow['offset_from_ideal'])
            right_offset_from_ideal = abs(right_pillow['offset_from_ideal'])
            
            # Check if both are close to their ideal positions
            avg_offset_from_ideal = (left_offset_from_ideal + right_offset_from_ideal) / 2
            offset_percent = (avg_offset_from_ideal / (bed_width / 2)) * 100

            if offset_percent <= self.symmetry_tolerance_percent:
                symmetry_status = 'SYMMETRIC'
                symmetry_score = 100
            else:
                symmetry_status = 'ASYMMETRIC'
                symmetry_score = max(0, 100 - offset_percent * 3)

        elif len(pillow_boxes) > 2:
            # For multiple pillows, check spacing consistency
            centroids = sorted([p['centroid'][0] for p in pillow_analyses])
            if len(centroids) > 1:
                spacings = [centroids[i+1] - centroids[i] for i in range(len(centroids)-1)]
                spacing_std = np.std(spacings)
                spacing_std_percent = (spacing_std / bed_width) * 100
                symmetry_score = max(0, 100 - spacing_std_percent * 2)
                symmetry_status = 'EVENLY_SPACED' if spacing_std_percent < 10 else 'UNEVEN_SPACING'

        # Overall score combines individual positions and symmetry
        overall_score = (avg_score * 0.7 + symmetry_score * 0.3)

        # Generate recommendation
        recommendation = self.generate_recommendation(pillow_analyses, symmetry_status, overall_score)

        return {
            'overall_score': float(round(overall_score, 1)),
            'bed_centerline': float(bed_centerline),
            'centerline_info': centerline_info,  # Include full centerline info with angle
            'bed_box': bed_box,
            'bed_width': float(bed_width),
            'tolerance_distance': float(tolerance_distance),
            'num_pillows': int(len(pillow_boxes)),
            'pillow_details': pillow_analyses,
            'symmetry_status': symmetry_status,
            'symmetry_score': float(round(symmetry_score, 1)),
            'recommendation': recommendation
        }

    def generate_recommendation(self, pillow_analyses, symmetry_status, overall_score):
        """
        Generate human-readable recommendation for housekeeper

        Args:
            pillow_analyses: List of pillow analysis dicts
            symmetry_status: Symmetry status string
            overall_score: Overall alignment score

        Returns:
            String with actionable recommendation
        """
        if overall_score >= 90:
            return "✓ Pillows are properly aligned. No action needed."

        recommendations = []

        # Check individual pillow positions
        for analysis in pillow_analyses:
            if analysis['status'] == 'NEEDS_ADJUSTMENT':
                pillow_num = analysis['pillow_id'] + 1
                offset = analysis['offset_from_ideal']
                
                # Determine direction to move
                if offset > 0:
                    # Pillow is too far from center, move inward
                    if analysis['position'] == 'LEFT_OF_CENTER':
                        direction = "RIGHT (toward center)"
                    else:
                        direction = "LEFT (toward center)"
                else:
                    # Pillow is too close to center, move outward
                    if analysis['position'] == 'LEFT_OF_CENTER':
                        direction = "LEFT (away from center)"
                    else:
                        direction = "RIGHT (away from center)"
                
                recommendations.append(f"Move pillow #{pillow_num} {direction} by {abs(offset):.0f}px ({analysis['offset_percent']:.1f}%) to ideal symmetrical position")

            # Check edge alignment quality
            if 'edge_analysis' in analysis:
                edge_info = analysis['edge_analysis']
                if edge_info['alignment_score'] < 0.5:
                    pillow_num = analysis['pillow_id'] + 1
                    recommendations.append(f"Straighten pillow #{pillow_num} - edges are not well-aligned ({edge_info['alignment_score']*100:.0f}% alignment)")

        # Check symmetry
        if symmetry_status == 'ASYMMETRIC':
            recommendations.append("Adjust pillows for symmetrical distribution - ensure equal spacing on both sides of bed centerline")
        elif symmetry_status == 'UNEVEN_SPACING':
            recommendations.append("Redistribute pillows evenly - maintain equal distances from centerline on each side")

        if not recommendations:
            recommendations.append("Make minor adjustments to improve symmetrical distribution")

        return " | ".join(recommendations)

    def process_detections(self, detections, image=None):
        """
        Process all detections and analyze pillow-to-bed alignment with edge voting

        Args:
            detections: List of (box, class_name, confidence) tuples
            image: Optional image for edge analysis (should be blurred outside boxes)

        Returns:
            Dict with alignment results or None if no bed detected
        """
        # Separate beds and pillows (case-insensitive)
        beds = [(box, conf) for box, class_name, conf in detections if class_name.lower() == 'bed' and conf > 0.6]
        pillows = [(box, conf) for box, class_name, conf in detections if class_name.lower() == 'pillow']

        if not beds:
            return {
                'status': 'NO_BED',
                'overall_score': 0,
                'recommendation': 'No bed detected with confidence > 0.6'
            }

        # Use the bed with highest confidence
        bed_box, bed_conf = max(beds, key=lambda x: x[1])

        # Filter pillows by confidence > 0.5
        high_conf_pillows = [box for box, conf in pillows if conf > 0.5]

        # Analyze alignment with edge voting
        result = self.analyze_pillow_to_bed_alignment(bed_box, high_conf_pillows, image)

        # Add bed edge analysis if image provided
        if image is not None:
            bed_edge_analysis = self.analyze_edge_orientation(image, bed_box)
            result['bed_edge_analysis'] = bed_edge_analysis

        result['bed_confidence'] = float(bed_conf)
        result['bed_box'] = [float(x) for x in bed_box]
        result['total_pillows_detected'] = int(len(pillows))
        result['high_confidence_pillows'] = int(len(high_conf_pillows))

        return result


# Global model cache to avoid reloading
_cached_alignment_model = None
_cached_model_path = None

def score_alignment(image_path_or_detections, image=None):
    """
    Convenience function to score alignment.
    
    Args:
        image_path_or_detections: Either a file path (str) to an image, or a list of 
                                  (box, class_name, confidence) detection tuples
        image: Optional image for edge analysis (only used if detections are passed directly)
        
    Returns:
        Dict with alignment results including 'aligned', 'score', and other metrics
    """
    from ultralytics import YOLO
    import os
    import pathlib
    import platform
    from pathlib import WindowsPath, PureWindowsPath
    
    global _cached_alignment_model, _cached_model_path
    
    scorer = AlignmentScorer()
    
    # Check if input is a file path
    if isinstance(image_path_or_detections, str):
        image_path = image_path_or_detections
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                'aligned': True,
                'score': 0.0,
                'status': 'IMAGE_ERROR',
                'recommendation': 'Could not load image'
            }
        
        # Try to load Stage3 model for bed/pillow detection
        model_path = None
        for candidate in ['Stage3_BedPillow_mobile.torchscript', 'Stage3_BedPillow.pt']:
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        if model_path is None:
            return {
                'aligned': True,
                'score': 0.85,
                'status': 'NO_MODEL',
                'recommendation': 'Alignment model not available'
            }
        
        # Load or use cached model
        _original_posix = getattr(pathlib, 'PosixPath', None)
        _original_pure_posix = getattr(pathlib, 'PurePosixPath', None)
        
        try:
            # Load model if not cached or path changed
            if _cached_alignment_model is None or _cached_model_path != model_path:
                # Suppress YOLO logging
                import logging
                logging.getLogger('ultralytics').setLevel(logging.ERROR)
                
                # Apply PosixPath patch for Windows if loading .pt file
                if platform.system() == 'Windows' and model_path.endswith('.pt'):
                    pathlib.PosixPath = WindowsPath
                    pathlib.PurePosixPath = PureWindowsPath
                
                _cached_alignment_model = YOLO(model_path, task='detect', verbose=False)
                _cached_model_path = model_path
            
            # Run inference
            results = _cached_alignment_model(image_path, conf=0.5, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    class_name = result.names[class_id]
                    detections.append(([x1, y1, x2, y2], class_name, confidence))
            
            alignment_result = scorer.process_detections(detections, img)
            
            # Map result to expected format
            if alignment_result.get('status') == 'NO_BED':
                return {
                    'aligned': True,
                    'score': 0.5,
                    'status': 'NO_BED',
                    'recommendation': 'No bed detected'
                }
            
            overall_score = alignment_result.get('overall_score', 0) / 100.0  # Convert to 0-1 range
            is_aligned = overall_score >= 0.7
            
            # Prepare debug data for visualization
            centerline_info = alignment_result.get('centerline_info', {})
            bed_box = alignment_result.get('bed_box', [])
            
            # Calculate centerline endpoints for drawing
            centerline = None
            if centerline_info and bed_box:
                cx = centerline_info.get('center_x', 0)
                cy = centerline_info.get('center_y', 0)
                angle = centerline_info.get('angle', 90)
                bed_height = bed_box[3] - bed_box[1]
                
                # Calculate line endpoints extending across bed height
                angle_rad = np.radians(angle)
                length = bed_height
                x1 = cx - length * np.cos(angle_rad)
                y1 = cy - length * np.sin(angle_rad)
                x2 = cx + length * np.cos(angle_rad)
                y2 = cy + length * np.sin(angle_rad)
                centerline = [x1, y1, x2, y2]
            
            # Get pillow boxes
            pillow_details = alignment_result.get('pillow_details', [])
            pillow_boxes = [p['box'] for p in pillow_details] if pillow_details else []
            
            return {
                'aligned': is_aligned,
                'score': overall_score,
                'status': alignment_result.get('status', 'OK'),
                'recommendation': alignment_result.get('recommendation', ''),
                'details': alignment_result,
                'debug_data': {
                    'bed_box': bed_box,
                    'pillow_boxes': pillow_boxes,
                    'centerline': centerline,
                    'centerline_info': centerline_info
                }
            }
            
        except Exception as e:
            return {
                'aligned': True,
                'score': 0.5,
                'status': 'ERROR',
                'recommendation': f'Detection error: {str(e)}'
            }
        finally:
            # Always restore original pathlib objects
            if _original_posix is not None:
                pathlib.PosixPath = _original_posix
            if _original_pure_posix is not None:
                pathlib.PurePosixPath = _original_pure_posix
    else:
        # Input is detections list
        detections = image_path_or_detections
        alignment_result = scorer.process_detections(detections, image)
        
        overall_score = alignment_result.get('overall_score', 0) / 100.0
        is_aligned = overall_score >= 0.7
        
        # Prepare debug data for visualization
        centerline_info = alignment_result.get('centerline_info', {})
        bed_box = alignment_result.get('bed_box', [])
        
        # Calculate centerline endpoints for drawing
        centerline = None
        if centerline_info and bed_box:
            cx = centerline_info.get('center_x', 0)
            cy = centerline_info.get('center_y', 0)
            angle = centerline_info.get('angle', 90)
            bed_height = bed_box[3] - bed_box[1]
            
            # Calculate line endpoints extending across bed height
            angle_rad = np.radians(angle)
            length = bed_height
            x1 = cx - length * np.cos(angle_rad)
            y1 = cy - length * np.sin(angle_rad)
            x2 = cx + length * np.cos(angle_rad)
            y2 = cy + length * np.sin(angle_rad)
            centerline = [x1, y1, x2, y2]
        
        # Get pillow boxes
        pillow_details = alignment_result.get('pillow_details', [])
        pillow_boxes = [p['box'] for p in pillow_details] if pillow_details else []
        
        return {
            'aligned': is_aligned,
            'score': overall_score,
            'status': alignment_result.get('status', 'OK'),
            'recommendation': alignment_result.get('recommendation', ''),
            'details': alignment_result,
            'debug_data': {
                'bed_box': bed_box,
                'pillow_boxes': pillow_boxes,
                'centerline': centerline,
                'centerline_info': centerline_info
            }
        }

