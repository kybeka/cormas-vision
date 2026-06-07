#!/usr/bin/env python3
"""
Simple Web-based Pseudo-Label Inspector

A minimal, clean implementation focused on polygon corner dragging functionality.
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import yaml
import colorsys

class SimplePseudoLabelInspector:
    def __init__(self, pseudo_dir: str = "pseudo_iterations"):
        self.pseudo_dir = Path(pseudo_dir)
        self.current_iteration = 0
        self.current_image_index = 0
        self.image_files = []
        self.detections = []
        self.image_statuses = {}  # Track approve/deny/pending status
        self.class_names = self.load_class_names()
        self.class_colors = self.generate_colors(len(self.class_names))
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_routes()
    
    def load_class_names(self) -> List[str]:
        """Load class names from data.yaml"""
        data_yaml = Path("frames.v3i.yolov11/data.yaml")
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            return data.get('names', [])
        else:
            # Default class names
            return ['blue-pawn', 'blue-token', 'board', 'green-token', 'hand', 'inner-board', 
                   'orange-token', 'red-pawn', 'red-token', 'white-pawn', 'yellow-pawn', 'yellow-token']
    
    def generate_colors(self, num_classes: int) -> List[str]:
        """Generate distinct colors for each class"""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex_color)
        return colors
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return self.get_html_template()
        
        @self.app.route('/api/init/<int:iteration>')
        def init_inspection(iteration):
            """Initialize inspection for given iteration"""
            self.current_iteration = iteration
            iter_dir = self.pseudo_dir / f"iter_{iteration:02d}"
            pseudo_images_dir = iter_dir / "pseudo_images"
            
            if not pseudo_images_dir.exists():
                return jsonify({'error': f'No pseudo-labels found for iteration {iteration}'}), 404
            
            self.image_files = sorted(list(pseudo_images_dir.glob("*.jpg")))
            if not self.image_files:
                return jsonify({'error': f'No images found in {pseudo_images_dir}'}), 404
            
            self.current_image_index = 0
            
            # Initialize image statuses if not exists
            for img_file in self.image_files:
                if img_file.name not in self.image_statuses:
                    self.image_statuses[img_file.name] = 'pending'
            
            return jsonify({
                'success': True,
                'total_images': len(self.image_files),
                'iteration': iteration,
                'class_names': self.class_names,
                'class_colors': self.class_colors
            })
        
        @self.app.route('/api/image/<int:index>')
        def get_image(index):
            """Get image and its detections"""
            if index < 0 or index >= len(self.image_files):
                return jsonify({'error': 'Invalid image index'}), 400
            
            self.current_image_index = index
            image_path = self.image_files[index]
            
            # Load detections
            label_path = image_path.parent.parent / "pseudo_labels" / f"{image_path.stem}.txt"
            detections = self.load_detections(label_path)
            
            # Convert image to base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Get image status
            status = self.image_statuses.get(image_path.name, 'pending')
            
            return jsonify({
                'success': True,
                'image_data': f'data:image/jpeg;base64,{image_data}',
                'detections': detections,
                'image_index': index,
                'filename': image_path.name,
                'status': status
            })
        
        @self.app.route('/api/save', methods=['POST'])
        def save_detections():
            """Save modified detections"""
            data = request.get_json()
            image_index = data.get('image_index')
            detections = data.get('detections', [])
            status = data.get('status')
            
            if image_index is None or image_index < 0 or image_index >= len(self.image_files):
                return jsonify({'error': 'Invalid image index'}), 400
            
            image_path = self.image_files[image_index]
            label_path = image_path.parent.parent / "pseudo_labels" / f"{image_path.stem}.txt"
            
            # Save detections
            self.save_detections_to_file(label_path, detections)
            
            # Update status if provided
            if status:
                self.image_statuses[image_path.name] = status
            
            return jsonify({'success': True})
        
        @self.app.route('/api/statistics')
        def get_statistics():
            """Get inspection statistics"""
            stats = {
                'total_images': len(self.image_files),
                'approved': sum(1 for s in self.image_statuses.values() if s == 'approved'),
                'denied': sum(1 for s in self.image_statuses.values() if s == 'denied'),
                'pending': sum(1 for s in self.image_statuses.values() if s == 'pending')
            }
            stats['completed'] = stats['approved'] + stats['denied']
            stats['progress_percent'] = round((stats['completed'] / stats['total_images']) * 100, 1) if stats['total_images'] > 0 else 0
            
            return jsonify(stats)
        
        @self.app.route('/api/finish_inspection', methods=['POST'])
        def finish_inspection():
            """Mark inspection as finished"""
            # Here you could trigger the next iteration of the pipeline
            # For now, just return success
            return jsonify({'success': True, 'message': 'Inspection completed successfully'})
    
    def load_detections(self, label_path: Path) -> List[Dict]:
        """Load detections from YOLO format file"""
        detections = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # At least class_id + 4 coordinates
                        class_id = int(parts[0])
                        if len(parts) == 5:  # Bounding box format: class_id + x_center + y_center + width + height
                            x_center, y_center, width, height = [float(x) for x in parts[1:5]]
                            detections.append({
                                'class_id': class_id,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height,
                                'type': 'bbox'
                            })
                        elif len(parts) >= 9:  # OBB format: class_id + 8 coordinates
                            coords = [float(x) for x in parts[1:9]]
                            # Convert to polygon points (4 corners)
                            points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                            detections.append({
                                'class_id': class_id,
                                'points': points,
                                'type': 'polygon'
                            })
        return detections
    
    def save_detections_to_file(self, label_path: Path, detections: List[Dict]):
        """Save detections to YOLO format file"""
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w') as f:
            for detection in detections:
                if detection['type'] == 'polygon' and len(detection['points']) == 4:
                    class_id = detection['class_id']
                    coords = []
                    for point in detection['points']:
                        coords.extend([point[0], point[1]])
                    line = f"{class_id} {' '.join(map(str, coords))}\n"
                    f.write(line)
                elif detection['type'] == 'bbox':
                    class_id = detection['class_id']
                    x_center = detection['x_center']
                    y_center = detection['y_center']
                    width = detection['width']
                    height = detection['height']
                    line = f"{class_id} {x_center} {y_center} {width} {height}\n"
                    f.write(line)
    
    def get_html_template(self):
        """Return the HTML UI (loaded from inspector_ui.html)."""
        ui = Path(__file__).resolve().parent / "inspector_ui.html"
        return ui.read_text().replace("__ITERATION__", str(self.current_iteration))

    def run(self, iteration: int = 0, debug: bool = False, port: int = 5000):
        """Run the inspector"""
        self.current_iteration = iteration
        print(f"\n=== Simple Pseudo-Label Inspector ===")
        print(f"Starting inspection for iteration {iteration}")
        print(f"Server will start on http://127.0.0.1:{port}")
        print(f"\nInstructions:")
        print(f"  • Click and drag polygon corners to adjust shapes")
        print(f"  • Hold Shift and click to select multiple objects")
        print(f"  • Change class for all selected objects at once")
        print(f"  • Use Previous/Next buttons or arrow keys to navigate")
        print(f"  • Press S or click Save to save changes")
        print(f"  • Press Ctrl+X (or Cmd+X on Mac) to remove all degenerate polygons")
        print(f"\nClose the browser window when finished.")
        print(f"\nPress Enter when you've finished the inspection...")
        
        self.app.run(host='127.0.0.1', port=port, debug=debug)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple Pseudo-Label Inspector')
    parser.add_argument('-i', '--iteration', type=int, default=0, help='Iteration to inspect')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    inspector = SimplePseudoLabelInspector()
    inspector.run(iteration=args.iteration, debug=args.debug, port=args.port)

if __name__ == "__main__":
    main()