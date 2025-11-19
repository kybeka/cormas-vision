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
        """Return the HTML template"""
        template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Pseudo-Label Inspector</title>
    <style>
        body { margin: 0; padding: 0; font-family: Arial, sans-serif; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 15px 20px; }
        .header h1 { margin: 0; font-size: 24px; }
        .file-info { margin: 5px 0; font-size: 14px; color: #ecf0f1; }
        
        .main-container { display: flex; height: calc(100vh - 80px); }
        .canvas-section { flex: 1; padding: 20px; background: white; margin: 10px; border-radius: 8px; display: flex; flex-direction: column; }
        .controls-section { width: 250px; padding: 20px; background: white; margin: 10px; border-radius: 8px; overflow-y: auto; }
        
        .canvas-container { border: 2px solid #ddd; border-radius: 4px; flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        #canvas { 
            cursor: crosshair; 
            display: block; 
            touch-action: none; 
            overscroll-behavior: contain; 
        }
        
        .control-group { margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #eee; }
        .control-group:last-child { border-bottom: none; }
        .control-group h3 { margin: 0 0 10px 0; font-size: 16px; color: #2c3e50; }
        
        .btn { padding: 8px 16px; margin: 2px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn:hover { opacity: 0.8; }
        .btn-primary { background: #3498db; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-warning { background: #f39c12; color: white; }
        .btn-secondary { background: #95a5a6; color: white; }
        .btn-active { background: #2c3e50; color: white; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .layer-controls {
            margin-top: 10px;
            display: flex;
            gap: 5px;
        }
        
        .layer-controls button {
            flex: 1;
            font-size: 12px;
            padding: 5px 8px;
        }
        
        .dark-mode-toggle {
            position: absolute;
            top: 10px;
            right: 10px;
            background: #333;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        
        /* Dark mode styles */
        body.dark-mode {
            background-color: #1a1a1a !important;
            color: #e0e0e0 !important;
        }
        
        body.dark-mode {
            background-color: #121212 !important;
            color: #e0e0e0 !important;
        }
        
        body.dark-mode .container {
            background-color: #1a1a1a !important;
        }
        
        body.dark-mode .header {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            border-bottom: 1px solid #444 !important;
        }
        
        body.dark-mode .header h1 {
            color: #e0e0e0 !important;
        }
        
        body.dark-mode .main-content {
            background-color: #1a1a1a !important;
        }
        
        body.dark-mode .controls {
            background-color: #2d2d2d !important;
            border-left: 1px solid #444 !important;
        }
        
        body.dark-mode .section {
            background-color: #333 !important;
            border: 1px solid #555 !important;
            color: #e0e0e0 !important;
        }
        
        body.dark-mode .section h3 {
            color: #fff !important;
            border-bottom: 1px solid #555 !important;
        }
        
        body.dark-mode select, body.dark-mode input {
            background-color: #444 !important;
            color: #e0e0e0 !important;
            border: 1px solid #666 !important;
        }
        
        body.dark-mode .btn {
            background-color: #555 !important;
            color: #e0e0e0 !important;
            border: 1px solid #666 !important;
        }
        
        body.dark-mode .btn:hover {
            background-color: #666 !important;
        }
        
        body.dark-mode .btn-active {
            background-color: #007bff !important;
            color: white !important;
        }
        
        body.dark-mode .class-item {
            color: #e0e0e0 !important;
        }
        
        body.dark-mode .canvas-section {
            background: #2d2d2d !important;
        }
        
        body.dark-mode .canvas-container {
            border: 2px solid #555 !important;
        }
        
        body.dark-mode canvas {
            border: 1px solid #555 !important;
            background-color: #2a2a2a !important;
        }
        
        body.dark-mode .stats {
            background-color: #333 !important;
            color: #e0e0e0 !important;
            border: 1px solid #555 !important;
        }
        
        body.dark-mode .selection-info {
            background-color: #333 !important;
            color: #e0e0e0 !important;
            border: 1px solid #555 !important;
        }
        
        body.dark-mode .finish-section {
            background-color: #3a3a2a !important;
            color: #e0e0e0 !important;
            border: 1px solid #666 !important;
        }
        
        body.dark-mode .dark-mode-toggle {
            background: #555 !important;
            color: #e0e0e0 !important;
        }
        
        body.dark-mode label {
            color: #e0e0e0 !important;
        }
        
        body.dark-mode textarea {
            background-color: #444 !important;
            color: #e0e0e0 !important;
            border: 1px solid #666 !important;
        }
        
        body.dark-mode .layer-controls button {
            background-color: #555 !important;
            color: #e0e0e0 !important;
            border: 1px solid #666 !important;
        }
        
        body.dark-mode .layer-controls button:hover {
            background-color: #666 !important;
        }
        
        body.dark-mode .controls-section {
            background: #2d2d2d !important;
        }
        
        body.dark-mode .control-group {
            border-bottom: 1px solid #555 !important;
        }
        
        body.dark-mode .control-group h3 {
            color: #e0e0e0 !important;
        }
        
        body.dark-mode .layer-controls button:disabled {
            background-color: #333 !important;
            color: #666 !important;
        }
        
        .tool-buttons { display: flex; flex-direction: column; gap: 5px; }
        .nav-buttons { display: flex; gap: 5px; }
        .status-buttons { display: flex; gap: 5px; }
        
        select { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 100%; }
        
        .stats { background: #ecf0f1; padding: 10px; border-radius: 4px; font-size: 12px; }
        .stats-item { display: flex; justify-content: space-between; margin: 2px 0; }
        
        .class-legend { font-size: 12px; }
        .class-item { display: flex; align-items: center; margin: 5px 0; }
        .color-box { width: 16px; height: 16px; border-radius: 2px; margin-right: 8px; }
        
        .selection-info { background: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 12px; }
        
        .finish-section { background: #fff3cd; padding: 15px; border-radius: 4px; border: 1px solid #ffeaa7; }
    </style>
</head>
<body>
    <button class="dark-mode-toggle" onclick="toggleDarkMode()">🌙 Dark Mode</button>
    <div class="header">
        <h1>Enhanced Pseudo-Label Inspector</h1>
        <div class="file-info" id="fileInfo">No image loaded</div>
    </div>
    
    <div class="main-container">
        <div class="canvas-section">
            <div class="canvas-container">
                <canvas id="canvas" width="1200" height="800"></canvas>
            </div>
        </div>
        
        <div class="controls-section">
            <div class="control-group">
                <h3>Tools</h3>
                <div class="tool-buttons">
                    <button class="btn btn-active" id="selectTool" onclick="setTool('select')">Select Tool</button>
                    <button class="btn btn-secondary" id="dragTool" onclick="setTool('drag')">Drag Tool</button>
                    <button class="btn btn-secondary" id="polygonTool" onclick="setTool('polygon')">Polygon Tool</button>
                    <button class="btn btn-secondary" id="bboxTool" onclick="setTool('bbox')">Bounding Box Tool</button>
                </div>
            </div>
            
            <div class="control-group">
                <h3>Navigation</h3>
                <div class="nav-buttons">
                    <button class="btn btn-primary" onclick="previousImage()">← Previous</button>
                    <button class="btn btn-primary" onclick="nextImage()">Next →</button>
                </div>
                <button class="btn btn-success" onclick="saveDetections()" style="width: 100%; margin-top: 10px;">💾 Save Changes</button>
            </div>
            
            <div class="control-group">
                <h3>Image Status</h3>
                <div class="status-buttons">
                    <button class="btn btn-success" onclick="setImageStatus('approved')">✓ Approve</button>
                    <button class="btn btn-danger" onclick="setImageStatus('denied')">✗ Deny</button>
                    <button class="btn btn-warning" onclick="setImageStatus('pending')">⏳ Pending</button>
                </div>
                <div id="currentStatus" style="margin-top: 10px; font-weight: bold;">Status: Pending</div>
            </div>
            
            <div class="control-group">
                <h3>Detection Info</h3>
                <div class="stats" id="detectionCount">Detections: 0</div>
                <div class="selection-info" id="detectionInfo">No detection selected</div>
            </div>
            
            <div class="control-group">
                <h3>Selection</h3>
                <div class="selection-info" id="selectionInfo">No detection selected</div>
                <select id="classSelect" onchange="changeSelectedClass()" disabled>
                    <option value="">Select class...</option>
                </select>
                <div class="layer-controls" style="margin-top: 10px; display: flex; gap: 5px;">
                    <button class="btn btn-secondary" onclick="sendToBack()" id="sendBackBtn" disabled>Send To Back</button>
                            <button class="btn btn-secondary" onclick="bringToFront()" id="bringFrontBtn" disabled>Bring To Front</button>
                </div>
                <button class="btn btn-danger" onclick="deleteSelected()" id="deleteBtn" disabled style="width: 100%; margin-top: 10px;">Delete Selected</button>
            </div>
            
            <div class="control-group">
                <h3>Class Legend</h3>
                <div class="class-legend" id="classLegend"></div>
            </div>
            
            <div class="control-group">
                <h3>Statistics</h3>
                <div class="stats" id="statistics">Loading...</div>
            </div>
            
            <div class="control-group finish-section">
                <h3>Finish Inspection</h3>
                <p style="margin: 10px 0; font-size: 12px;">Complete the inspection to proceed with the next iteration.</p>
                <button class="btn btn-warning" onclick="finishInspection()" style="width: 100%;">🏁 Finish Inspection</button>
            </div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let currentImage = null;
        let detections = [];
        let imageIndex = 0;
        let totalImages = 0;
        let classNames = [];
        let classColors = [];
        let currentStatus = 'pending';
        
        // Tool state
        let currentTool = 'select';
        let selectedDetection = null;
        let selectedDetections = []; // Array for multi-select
        let selectedCorner = null;
        
        // Interaction state
        let isDragging = false;
        let dragDetectionIndex = null;
        let dragPointIndex = null;
        let isDrawing = false;
        let drawingPoints = [];
        let drawingBbox = null;
        let isDragMode = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let originalShapeData = null; // Store original shape coordinates for proper dragging
        let copiedDetection = null;
        
        // Zoom state
        let zoomLevel = 1;
        let panX = 0;
        let panY = 0;
        let isPanning = false;
        let lastPanX = 0;
        let lastPanY = 0;
        
        // Initialize
        window.onload = function() {
            resizeCanvas();
            initInspection();
            setupEventListeners();
            updateStatistics();
            setTool('select');
        };
        
        // Resize canvas to fill available space
        function resizeCanvas() {
            const canvasContainer = document.querySelector('.canvas-container');
            
            // Get available space (subtract padding and margins)
            const availableWidth = canvasContainer.clientWidth - 20; // Account for border and padding
            const availableHeight = canvasContainer.clientHeight - 20; // Account for border and padding
            
            // Use full available space
            canvas.width = Math.max(availableWidth, 800); // Minimum reasonable size
            canvas.height = Math.max(availableHeight, 600); // Minimum reasonable size
            
            // Redraw if image is loaded
            if (currentImage) {
                drawCanvas();
            }
        }
        
        // Resize canvas when window resizes
        window.addEventListener('resize', function() {
            resizeCanvas();
        });
        
        async function initInspection() {
            try {
                const response = await fetch('/api/init/0');
                const data = await response.json();
                
                if (data.success) {
                    totalImages = data.total_images;
                    classNames = data.class_names;
                    classColors = data.class_colors;
                    setupClassSelect();
                    setupClassLegend();
                    await loadImage(0);
                } else {
                    console.error('Failed to initialize:', data.error);
                }
            } catch (error) {
                console.error('Failed to initialize:', error);
            }
        }
        
        async function loadImage(index) {
            try {
                const response = await fetch(`/api/image/${index}`);
                const data = await response.json();
                
                if (data.success) {
                    imageIndex = index;
                    detections = data.detections;
                    currentStatus = data.status;
                    selectedDetection = null;
                    
                    // Reset drag state when loading new image
                    isDragging = false;
                    dragDetectionIndex = null;
                    dragPointIndex = null;
                    selectedCorner = null;
                    originalShapeData = null;
                    dragStartX = 0;
                    dragStartY = 0;
                    
                    // Reset drawing state
                    isDrawing = false;
                    drawingPoints = [];
                    drawingBbox = null;
                    
                    const img = new Image();
                    img.onload = function() {
                        currentImage = img;
                        drawCanvas();
                        updateInfo(data.filename);
                        updateStatusDisplay();
                        updateSelectionInfo();
                        updateDetectionInfo();
                    };
                    img.src = data.image_data;
                } else {
                    console.error('Failed to load image:', data.error);
                }
            } catch (error) {
                console.error('Failed to load image:', error);
            }
        }
        
        function drawCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (currentImage) {
                // Save context state
                ctx.save();
                
                // Apply zoom and pan transformations
                ctx.translate(panX, panY);
                ctx.scale(zoomLevel, zoomLevel);
                
                // Calculate scale to fit image in canvas (before zoom)
                const scale = Math.min(canvas.width / currentImage.width, canvas.height / currentImage.height);
                const scaledWidth = currentImage.width * scale;
                const scaledHeight = currentImage.height * scale;
                const offsetX = (canvas.width - scaledWidth) / 2;
                const offsetY = (canvas.height - scaledHeight) / 2;
                
                // Draw image
                ctx.drawImage(currentImage, offsetX, offsetY, scaledWidth, scaledHeight);
                
                // Draw detections
                detections.forEach((detection, detIndex) => {
                    const isSelected = selectedDetection === detIndex;
                    const isMultiSelected = selectedDetections.includes(detIndex);
                    if (detection.type === 'polygon' && detection.points.length === 4) {
                        drawPolygon(detection, detIndex, offsetX, offsetY, scale, isSelected, isMultiSelected);
                    } else if (detection.type === 'bbox') {
                        drawBoundingBox(detection, detIndex, offsetX, offsetY, scale, isSelected, isMultiSelected);
                    }
                });
                
                // Draw current drawing
                if (isDrawing && currentTool === 'bbox' && drawingBbox) {
                    drawCurrentBbox(offsetX, offsetY, scale);
                }
                
                // Restore context state
                ctx.restore();
            }
        }
        
        function drawPolygon(detection, detIndex, offsetX, offsetY, scale, isSelected, isMultiSelected) {
            const points = detection.points.map(p => ({
                x: offsetX + p[0] * currentImage.width * scale,
                y: offsetY + p[1] * currentImage.height * scale
            }));
            
            const color = classColors[detection.class_id] || '#ff0000';
            const className = classNames[detection.class_id] || `Class ${detection.class_id}`;
            const isDegenerate = isPolygonDegenerate(points);
            
            // Draw polygon outline
            ctx.strokeStyle = isDegenerate ? '#ff6600' : color; // Orange for degenerate polygons
            ctx.lineWidth = isSelected ? 3 : (isMultiSelected ? 2.5 : 2);
            
            // Use dashed line for degenerate polygons or dotted for multi-selected
            if (isDegenerate) {
                ctx.setLineDash([5, 5]);
            } else if (isMultiSelected && !isSelected) {
                ctx.setLineDash([3, 3]);
            } else {
                ctx.setLineDash([]);
            }
            
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            for (let i = 1; i < points.length; i++) {
                ctx.lineTo(points[i].x, points[i].y);
            }
            ctx.closePath();
            ctx.stroke();
            
            // Reset line dash
            ctx.setLineDash([]);
            
            // Draw corner handles (only in select mode or if selected)
            if (currentTool === 'select' || isSelected) {
                points.forEach((point, pointIndex) => {
                    ctx.fillStyle = isDegenerate ? '#ff6600' : color;
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, isSelected ? 6 : 4, 0, 2 * Math.PI);
                    ctx.fill();
                });
            }
            
            // Draw class name with warning for degenerate polygons
            if (points.length > 0) {
                ctx.fillStyle = isDegenerate ? '#ff6600' : color;
                ctx.font = '12px Arial';
                const displayName = isDegenerate ? `${className} (DEGENERATE)` : className;
                ctx.fillText(displayName, points[0].x + 8, points[0].y - 8);
            }
        }
        
        function drawBoundingBox(detection, detIndex, offsetX, offsetY, scale, isSelected, isMultiSelected) {
            const x = offsetX + (detection.x_center - detection.width/2) * currentImage.width * scale;
            const y = offsetY + (detection.y_center - detection.height/2) * currentImage.height * scale;
            const width = detection.width * currentImage.width * scale;
            const height = detection.height * currentImage.height * scale;
            
            const color = classColors[detection.class_id] || '#ff0000';
            const className = classNames[detection.class_id] || `Class ${detection.class_id}`;
            
            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = isSelected ? 3 : (isMultiSelected ? 2.5 : 2);
            
            // Use dotted line for multi-selected (but not primary selected)
            if (isMultiSelected && !isSelected) {
                ctx.setLineDash([3, 3]);
            } else {
                ctx.setLineDash([]);
            }
            
            ctx.strokeRect(x, y, width, height);
            ctx.setLineDash([]); // Reset dash pattern
            
            // Draw resize handles (only in select mode or if selected)
            if (currentTool === 'select' || isSelected) {
                const handleSize = 8;
                ctx.fillStyle = color;
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 1;
                
                // Corner handles
                const corners = [
                    {x: x - handleSize/2, y: y - handleSize/2}, // nw
                    {x: x + width - handleSize/2, y: y - handleSize/2}, // ne
                    {x: x + width - handleSize/2, y: y + height - handleSize/2}, // se
                    {x: x - handleSize/2, y: y + height - handleSize/2} // sw
                ];
                
                // Side handles
                const sides = [
                    {x: x + width/2 - handleSize/2, y: y - handleSize/2}, // n
                    {x: x + width - handleSize/2, y: y + height/2 - handleSize/2}, // e
                    {x: x + width/2 - handleSize/2, y: y + height - handleSize/2}, // s
                    {x: x - handleSize/2, y: y + height/2 - handleSize/2} // w
                ];
                
                // Draw all handles
                [...corners, ...sides].forEach(handle => {
                    ctx.fillRect(handle.x, handle.y, handleSize, handleSize);
                    ctx.strokeRect(handle.x, handle.y, handleSize, handleSize);
                });
            }
            
            // Draw class name
            ctx.fillStyle = color;
            ctx.font = '12px Arial';
            ctx.fillText(className, x + 8, y - 8);
        }
        
        function drawCurrentBbox(offsetX, offsetY, scale) {
            if (!drawingBbox) return;
            
            ctx.strokeStyle = '#666666';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(
                drawingBbox.startX,
                drawingBbox.startY,
                drawingBbox.currentX - drawingBbox.startX,
                drawingBbox.currentY - drawingBbox.startY
            );
            ctx.setLineDash([]);
        }
        
        function setupEventListeners() {
            canvas.addEventListener('mousedown', onMouseDown);
            canvas.addEventListener('mousemove', onMouseMove);
            canvas.addEventListener('mouseup', onMouseUp);
            canvas.addEventListener('wheel', onWheel, { passive: false });
            canvas.addEventListener('touchstart', onTouchStart, { passive: false });
            canvas.addEventListener('touchmove', onTouchMove, { passive: false });
            canvas.addEventListener('touchend', onTouchEnd, { passive: false });
            
            // Prevent Safari's pinch-zoom gestures
            window.addEventListener('gesturestart', e => e.preventDefault(), { passive: false });
            window.addEventListener('gesturechange', e => e.preventDefault(), { passive: false });
            window.addEventListener('gestureend', e => e.preventDefault(), { passive: false });
            
            document.addEventListener('keydown', onKeyDown);
        }
        
        function handleSelectTool(x, y, offsetX, offsetY, scale, shiftKey = false) {
            // First check for corner handles if something is selected (only for single selection)
            if (selectedDetection !== null && selectedDetections.length <= 1) {
                const detection = detections[selectedDetection];
                if (detection.type === 'polygon') {
                    for (let pointIndex = 0; pointIndex < detection.points.length; pointIndex++) {
                        const point = {
                            x: offsetX + detection.points[pointIndex][0] * currentImage.width * scale,
                            y: offsetY + detection.points[pointIndex][1] * currentImage.height * scale
                        };
                        
                        const distance = Math.sqrt((x - point.x) ** 2 + (y - point.y) ** 2);
                        if (distance <= 8) {
                            isDragging = true;
                            dragDetectionIndex = selectedDetection;
                            dragPointIndex = pointIndex;
                            selectedCorner = pointIndex;
                            return;
                        }
                    }
                } else if (detection.type === 'bbox') {
                    const bx = offsetX + (detection.x_center - detection.width/2) * currentImage.width * scale;
                    const by = offsetY + (detection.y_center - detection.height/2) * currentImage.height * scale;
                    const bwidth = detection.width * currentImage.width * scale;
                    const bheight = detection.height * currentImage.height * scale;
                    const handleSize = 8;
                    
                    // Corner handles (0-3)
                    const corners = [
                        {x: bx - handleSize/2, y: by - handleSize/2}, // nw
                        {x: bx + bwidth - handleSize/2, y: by - handleSize/2}, // ne
                        {x: bx + bwidth - handleSize/2, y: by + bheight - handleSize/2}, // se
                        {x: bx - handleSize/2, y: by + bheight - handleSize/2} // sw
                    ];
                    
                    // Side handles (4-7)
                    const sides = [
                        {x: bx + bwidth/2 - handleSize/2, y: by - handleSize/2}, // n
                        {x: bx + bwidth - handleSize/2, y: by + bheight/2 - handleSize/2}, // e
                        {x: bx + bwidth/2 - handleSize/2, y: by + bheight - handleSize/2}, // s
                        {x: bx - handleSize/2, y: by + bheight/2 - handleSize/2} // w
                    ];
                    
                    const allHandles = [...corners, ...sides];
                    
                    for (let handleIndex = 0; handleIndex < allHandles.length; handleIndex++) {
                        const handle = allHandles[handleIndex];
                        if (x >= handle.x && x <= handle.x + handleSize && y >= handle.y && y <= handle.y + handleSize) {
                            isDragging = true;
                            dragDetectionIndex = selectedDetection;
                            dragPointIndex = handleIndex;
                            selectedCorner = pointIndex;
                            return;
                        }
                    }
                }
            }
            
            // Check for detection selection (iterate from back to front like drag tool)
            for (let detIndex = detections.length - 1; detIndex >= 0; detIndex--) {
                const detection = detections[detIndex];
                let isHit = false;
                
                if (detection.type === 'polygon') {
                    isHit = isPointInPolygon(x, y, detection, offsetX, offsetY, scale);
                } else if (detection.type === 'bbox') {
                    const bx = offsetX + (detection.x_center - detection.width/2) * currentImage.width * scale;
                    const by = offsetY + (detection.y_center - detection.height/2) * currentImage.height * scale;
                    const bwidth = detection.width * currentImage.width * scale;
                    const bheight = detection.height * currentImage.height * scale;
                    
                    isHit = (x >= bx && x <= bx + bwidth && y >= by && y <= by + bheight);
                }
                
                if (isHit) {
                    if (shiftKey) {
                        // Multi-select mode
                        const existingIndex = selectedDetections.indexOf(detIndex);
                        if (existingIndex >= 0) {
                            // Deselect if already selected
                            selectedDetections.splice(existingIndex, 1);
                            if (selectedDetection === detIndex) {
                                selectedDetection = selectedDetections.length > 0 ? selectedDetections[0] : null;
                            }
                        } else {
                            // Add to selection
                            selectedDetections.push(detIndex);
                            selectedDetection = detIndex; // Keep primary selection for UI
                        }
                    } else {
                        // Single select mode - clear previous selections
                        selectedDetections = [detIndex];
                        selectedDetection = detIndex;
                    }
                    selectedCorner = null;
                    updateSelectionInfo();
                    drawCanvas();
                    return;
                }
            }
            
            // Deselect if clicking empty area (unless shift is held)
            if (!shiftKey) {
                selectedDetection = null;
                selectedDetections = [];
                selectedCorner = null;
                updateSelectionInfo();
                drawCanvas();
            }
        }
        
        function handleBboxTool(x, y, offsetX, offsetY, scale) {
            // Convert to image coordinates
            const imgX = (x - offsetX) / (currentImage.width * scale);
            const imgY = (y - offsetY) / (currentImage.height * scale);
            
            if (imgX >= 0 && imgX <= 1 && imgY >= 0 && imgY <= 1) {
                isDrawing = true;
                drawingBbox = {
                    startX: x,
                    startY: y,
                    currentX: x,
                    currentY: y,
                    startImgX: imgX,
                    startImgY: imgY
                };
            }
        }
        
        function handlePolygonTool(x, y, offsetX, offsetY, scale) {
            // Convert to image coordinates
            const imgX = (x - offsetX) / (currentImage.width * scale);
            const imgY = (y - offsetY) / (currentImage.height * scale);
            
            if (imgX >= 0 && imgX <= 1 && imgY >= 0 && imgY <= 1) {
                if (!isDrawing) {
                    isDrawing = true;
                    drawingPoints = [[imgX, imgY]];
                } else {
                    drawingPoints.push([imgX, imgY]);
                    if (drawingPoints.length === 4) {
                        // Complete the polygon
                        const newDetection = {
                            type: 'polygon',
                            class_id: 0,
                            points: drawingPoints
                        };
                        detections.push(newDetection);
                        isDrawing = false;
                        drawingPoints = [];
                        drawCanvas();
                    }
                }
            }
        }
        
        function handleDragTool(x, y, offsetX, offsetY, scale) {
            // Check for detection selection for dragging
            for (let i = detections.length - 1; i >= 0; i--) {
                const detection = detections[i];
                let isInside = false;
                
                if (detection.type === 'polygon') {
                    // Check if point is inside polygon
                    isInside = isPointInPolygon(x, y, detection, offsetX, offsetY, scale);
                } else if (detection.type === 'bbox') {
                    // Check if point is inside bbox
                    const bx = offsetX + (detection.x_center - detection.width/2) * currentImage.width * scale;
                    const by = offsetY + (detection.y_center - detection.height/2) * currentImage.height * scale;
                    const bwidth = detection.width * currentImage.width * scale;
                    const bheight = detection.height * currentImage.height * scale;
                    
                    isInside = x >= bx && x <= bx + bwidth && y >= by && y <= by + bheight;
                }
                
                if (isInside) {
                    selectedDetection = i;
                    isDragging = true;
                    dragDetectionIndex = i;
                    dragStartX = x;
                    dragStartY = y;
                    
                    // Store original shape data for proper dragging
                    const detection = detections[i];
                    if (detection.type === 'polygon') {
                        originalShapeData = {
                            type: 'polygon',
                            points: detection.points.map(point => [point[0], point[1]])
                        };
                    } else if (detection.type === 'bbox') {
                        originalShapeData = {
                            type: 'bbox',
                            x_center: detection.x_center,
                            y_center: detection.y_center
                        };
                    }
                    
                    updateSelectionInfo();
                    drawCanvas();
                    return;
                }
            }
            
            // If no detection was clicked, deselect
            selectedDetection = null;
            updateSelectionInfo();
            drawCanvas();
        }
        
        // Helper function to convert screen coordinates to canvas coordinates accounting for zoom/pan
        function screenToCanvas(screenX, screenY) {
            return {
                x: (screenX - panX) / zoomLevel,
                y: (screenY - panY) / zoomLevel
            };
        }
        
        function onMouseDown(event) {
            if (!currentImage) return;
            
            const rect = canvas.getBoundingClientRect();
            const screenX = event.clientX - rect.left;
            const screenY = event.clientY - rect.top;
            
            // Convert to canvas coordinates accounting for zoom/pan
            const canvasCoords = screenToCanvas(screenX, screenY);
            const x = canvasCoords.x;
            const y = canvasCoords.y;
            
            const scale = Math.min(canvas.width / currentImage.width, canvas.height / currentImage.height);
            const offsetX = (canvas.width - currentImage.width * scale) / 2;
            const offsetY = (canvas.height - currentImage.height * scale) / 2;
            
            if (currentTool === 'select') {
                handleSelectTool(x, y, offsetX, offsetY, scale, event.shiftKey);
            } else if (currentTool === 'bbox') {
                handleBboxTool(x, y, offsetX, offsetY, scale);
            } else if (currentTool === 'polygon') {
                handlePolygonTool(x, y, offsetX, offsetY, scale);
            } else if (currentTool === 'drag') {
                handleDragTool(x, y, offsetX, offsetY, scale);
            }
        }
        
        function onMouseMove(event) {
            const rect = canvas.getBoundingClientRect();
            const screenX = event.clientX - rect.left;
            const screenY = event.clientY - rect.top;
            
            // Convert to canvas coordinates accounting for zoom/pan
            const canvasCoords = screenToCanvas(screenX, screenY);
            const x = canvasCoords.x;
            const y = canvasCoords.y;
            
            const scale = Math.min(canvas.width / currentImage.width, canvas.height / currentImage.height);
            const offsetX = (canvas.width - currentImage.width * scale) / 2;
            const offsetY = (canvas.height - currentImage.height * scale) / 2;
            
            if (isDragging && dragDetectionIndex !== null) {
                if (currentTool === 'drag' && originalShapeData) {
                    // Drag entire shape using original coordinates
                    const totalDeltaX = x - dragStartX;
                    const totalDeltaY = y - dragStartY;
                    const imgDeltaX = totalDeltaX / (currentImage.width * scale);
                    const imgDeltaY = totalDeltaY / (currentImage.height * scale);
                    
                    const detection = detections[dragDetectionIndex];
                    if (detection.type === 'polygon' && originalShapeData.type === 'polygon') {
                        detection.points = originalShapeData.points.map(point => [
                            Math.max(0, Math.min(1, point[0] + imgDeltaX)),
                            Math.max(0, Math.min(1, point[1] + imgDeltaY))
                        ]);
                    } else if (detection.type === 'bbox' && originalShapeData.type === 'bbox') {
                        detection.x_center = Math.max(detection.width/2, Math.min(1 - detection.width/2, originalShapeData.x_center + imgDeltaX));
                        detection.y_center = Math.max(detection.height/2, Math.min(1 - detection.height/2, originalShapeData.y_center + imgDeltaY));
                    }
                    
                    drawCanvas();
                } else if (dragPointIndex !== null) {
                    // Drag individual points/corners
                    const imgX = (x - offsetX) / (currentImage.width * scale);
                    const imgY = (y - offsetY) / (currentImage.height * scale);
                    
                    const detection = detections[dragDetectionIndex];
                    if (detection.type === 'polygon') {
                        detection.points[dragPointIndex] = [imgX, imgY];
                    } else if (detection.type === 'bbox') {
                        updateBboxCorner(detection, dragPointIndex, imgX, imgY);
                    }
                    
                    drawCanvas();
                }
            } else if (isDrawing && currentTool === 'bbox' && drawingBbox) {
                drawingBbox.currentX = x;
                drawingBbox.currentY = y;
                drawCanvas();
            }
        }
        
        function onMouseUp(event) {
            if (isDragging) {
                isDragging = false;
                if (currentTool === 'drag') {
                    // Save detections after dragging entire shape
                    saveDetections();
                }
                dragDetectionIndex = null;
                dragPointIndex = null;
                selectedCorner = null;
                originalShapeData = null; // Clear original shape data
            } else if (isDrawing && currentTool === 'bbox' && drawingBbox) {
                const rect = canvas.getBoundingClientRect();
                const screenX = event.clientX - rect.left;
                const screenY = event.clientY - rect.top;
                
                // Convert to canvas coordinates accounting for zoom/pan
                const canvasCoords = screenToCanvas(screenX, screenY);
                const x = canvasCoords.x;
                const y = canvasCoords.y;
                
                const scale = Math.min(canvas.width / currentImage.width, canvas.height / currentImage.height);
                const offsetX = (canvas.width - currentImage.width * scale) / 2;
                const offsetY = (canvas.height - currentImage.height * scale) / 2;
                
                const imgX = (x - offsetX) / (currentImage.width * scale);
                const imgY = (y - offsetY) / (currentImage.height * scale);
                
                if (imgX >= 0 && imgX <= 1 && imgY >= 0 && imgY <= 1) {
                    const startX = Math.min(drawingBbox.startImgX, imgX);
                    const startY = Math.min(drawingBbox.startImgY, imgY);
                    const endX = Math.max(drawingBbox.startImgX, imgX);
                    const endY = Math.max(drawingBbox.startImgY, imgY);
                    
                    const width = endX - startX;
                    const height = endY - startY;
                    
                    if (width > 0.01 && height > 0.01) { // Minimum size check
                        const newDetection = {
                            type: 'bbox',
                            class_id: 0,
                            x_center: startX + width / 2,
                            y_center: startY + height / 2,
                            width: width,
                            height: height
                        };
                        detections.push(newDetection);
                    }
                }
                
                isDrawing = false;
                drawingBbox = null;
                drawCanvas();
            }
        }
        
        // Trackpad two-finger pan and zoom handler
        function onWheel(event) {
            event.preventDefault();
            
            // On macOS, Ctrl+wheel or Cmd+wheel for zooming (pinch gestures), regular wheel for panning (two-finger swipe)
            if (event.ctrlKey || event.metaKey) {
                // Zoom mode - pinch gestures on trackpad
                const rect = canvas.getBoundingClientRect();
                const mouseX = event.clientX - rect.left;
                const mouseY = event.clientY - rect.top;
                
                const zoomFactor = event.deltaY > 0 ? 0.98 : 1.02;
                const newZoomLevel = Math.max(0.1, Math.min(5, zoomLevel * zoomFactor));
                
                if (newZoomLevel !== zoomLevel) {
                    const canvasCoords = screenToCanvas(mouseX, mouseY);
                    panX = panX - (canvasCoords.x * (newZoomLevel - zoomLevel));
                    panY = panY - (canvasCoords.y * (newZoomLevel - zoomLevel));
                    zoomLevel = newZoomLevel;
                    drawCanvas();
                }
            } else {
                // Pan mode - two-finger swipe gestures on trackpad
                panX -= event.deltaX;
                panY -= event.deltaY;
                drawCanvas();
            }
        }
        
        let touchStartDistance = 0;
        let touchStartZoom = 1;
        let touchStartPan = { x: 0, y: 0 };
        let touchStartCenter = { x: 0, y: 0 };
        let lastTouchCenter = { x: 0, y: 0 };
        let isZooming = false;
        
        function onTouchStart(event) {
            if (event.touches.length === 1) {
                // Single touch - start panning immediately
                event.preventDefault();
                const rect = canvas.getBoundingClientRect();
                lastTouchCenter.x = event.touches[0].clientX - rect.left;
                lastTouchCenter.y = event.touches[0].clientY - rect.top;
                isZooming = false;
            } else if (event.touches.length === 2) {
                // Two finger - zoom only
                event.preventDefault();
                const touch1 = event.touches[0];
                const touch2 = event.touches[1];
                
                // Calculate initial distance for zoom detection
                touchStartDistance = Math.sqrt(
                    Math.pow(touch2.clientX - touch1.clientX, 2) +
                    Math.pow(touch2.clientY - touch1.clientY, 2)
                );
                
                // Calculate center point
                const rect = canvas.getBoundingClientRect();
                touchStartCenter.x = (touch1.clientX + touch2.clientX) / 2 - rect.left;
                touchStartCenter.y = (touch1.clientY + touch2.clientY) / 2 - rect.top;
                lastTouchCenter = { ...touchStartCenter };
                
                touchStartZoom = zoomLevel;
                touchStartPan = { x: panX, y: panY };
                isZooming = true;
            }
        }
        
        function onTouchMove(event) {
            if (event.touches.length === 1 && !isZooming) {
                // Single touch pan
                event.preventDefault();
                const rect = canvas.getBoundingClientRect();
                const currentX = event.touches[0].clientX - rect.left;
                const currentY = event.touches[0].clientY - rect.top;
                
                const deltaX = currentX - lastTouchCenter.x;
                const deltaY = currentY - lastTouchCenter.y;
                
                panX += deltaX;
                panY += deltaY;
                
                lastTouchCenter.x = currentX;
                lastTouchCenter.y = currentY;
                
                drawCanvas();
            } else if (event.touches.length === 2 && isZooming) {
                // Two finger zoom
                event.preventDefault();
                const touch1 = event.touches[0];
                const touch2 = event.touches[1];
                
                const currentDistance = Math.sqrt(
                    Math.pow(touch2.clientX - touch1.clientX, 2) +
                    Math.pow(touch2.clientY - touch1.clientY, 2)
                );
                
                const rect = canvas.getBoundingClientRect();
                const currentCenter = {
                    x: (touch1.clientX + touch2.clientX) / 2 - rect.left,
                    y: (touch1.clientY + touch2.clientY) / 2 - rect.top
                };
                
                // Zoom gesture
                const zoomFactor = currentDistance / touchStartDistance;
                zoomLevel = Math.max(0.1, Math.min(5, touchStartZoom * zoomFactor));
                
                // Zoom around the center point
                const zoomCenterX = (touchStartCenter.x - touchStartPan.x) / touchStartZoom;
                const zoomCenterY = (touchStartCenter.y - touchStartPan.y) / touchStartZoom;
                panX = touchStartCenter.x - zoomCenterX * zoomLevel;
                panY = touchStartCenter.y - zoomCenterY * zoomLevel;
                
                drawCanvas();
            }
        }
        
        function onTouchEnd(event) {
            if (event.touches.length < 2) {
                touchStartDistance = 0;
                isZooming = false; // Reset zoom state when touches end
            }
        }
        
        function findCornerAt(x, y) {
            if (!currentImage) return null;
            
            const scale = Math.min(canvas.width / currentImage.width, canvas.height / currentImage.height);
            const offsetX = (canvas.width - currentImage.width * scale) / 2;
            const offsetY = (canvas.height - currentImage.height * scale) / 2;
            
            for (let detIndex = 0; detIndex < detections.length; detIndex++) {
                const detection = detections[detIndex];
                if (detection.type === 'polygon') {
                    for (let pointIndex = 0; pointIndex < detection.points.length; pointIndex++) {
                        const point = detection.points[pointIndex];
                        const screenX = offsetX + point[0] * currentImage.width * scale;
                        const screenY = offsetY + point[1] * currentImage.height * scale;
                        
                        const distance = Math.sqrt((x - screenX) ** 2 + (y - screenY) ** 2);
                        if (distance <= 8) {
                            return {
                                detectionIndex: detIndex,
                                pointIndex: pointIndex,
                                screenX: screenX,
                                screenY: screenY
                            };
                        }
                    }
                }
            }
            return null;
        }
        
        function updateCornerPosition(corner, screenX, screenY) {
            if (!currentImage) return;
            
            const scale = Math.min(canvas.width / currentImage.width, canvas.height / currentImage.height);
            const offsetX = (canvas.width - currentImage.width * scale) / 2;
            const offsetY = (canvas.height - currentImage.height * scale) / 2;
            
            // Convert screen coordinates back to normalized coordinates
            const normalizedX = (screenX - offsetX) / (currentImage.width * scale);
            const normalizedY = (screenY - offsetY) / (currentImage.height * scale);
            
            // Clamp to image bounds
            const clampedX = Math.max(0, Math.min(1, normalizedX));
            const clampedY = Math.max(0, Math.min(1, normalizedY));
            
            // Update the detection point
            detections[corner.detectionIndex].points[corner.pointIndex] = [clampedX, clampedY];
        }
        
        async function saveDetections() {
            try {
                const response = await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_index: imageIndex,
                        detections: detections
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    console.log('Detections saved successfully');
                } else {
                    console.error('Failed to save:', data.error);
                }
            } catch (error) {
                console.error('Failed to save:', error);
            }
        }
        
        async function previousImage() {
            if (imageIndex > 0) {
                await saveDetections();
                await loadImage(imageIndex - 1);
            }
        }
        
        async function nextImage() {
            if (imageIndex < totalImages - 1) {
                await saveDetections();
                await loadImage(imageIndex + 1);
            }
        }
        
        function updateInfo(filename) {
            document.getElementById('fileInfo').textContent = 
                `Image ${imageIndex + 1}/${totalImages}: ${filename}`;
            updateStatistics();
        }
        
        // Utility functions
        function isPointInPolygon(x, y, detection, offsetX, offsetY, scale) {
            const points = detection.points.map(p => ({
                x: offsetX + p[0] * currentImage.width * scale,
                y: offsetY + p[1] * currentImage.height * scale
            }));
            
            // Check if polygon is degenerate (all points are collinear or very close)
            if (isPolygonDegenerate(points)) {
                return isPointNearLine(x, y, points);
            }
            
            let inside = false;
            for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
                if (((points[i].y > y) !== (points[j].y > y)) &&
                    (x < (points[j].x - points[i].x) * (y - points[i].y) / (points[j].y - points[i].y) + points[i].x)) {
                    inside = !inside;
                }
            }
            return inside;
        }
        
        function isPolygonDegenerate(points) {
            if (points.length < 3) return true;
            
            // Calculate the area using the shoelace formula
            let area = 0;
            for (let i = 0; i < points.length; i++) {
                const j = (i + 1) % points.length;
                area += points[i].x * points[j].y;
                area -= points[j].x * points[i].y;
            }
            area = Math.abs(area) / 2;
            
            // If area is very small, consider it degenerate
            return area < 10; // 10 square pixels threshold
        }
        
        function isPointNearLine(x, y, points) {
            const threshold = 8; // 8 pixel threshold for line selection
            
            // Check distance to each edge of the polygon
            for (let i = 0; i < points.length; i++) {
                const j = (i + 1) % points.length;
                const p1 = points[i];
                const p2 = points[j];
                
                const distance = distanceToLineSegment(x, y, p1.x, p1.y, p2.x, p2.y);
                if (distance <= threshold) {
                    return true;
                }
            }
            return false;
        }
        
        function distanceToLineSegment(px, py, x1, y1, x2, y2) {
            const dx = x2 - x1;
            const dy = y2 - y1;
            const length = Math.sqrt(dx * dx + dy * dy);
            
            if (length === 0) {
                // Line segment is actually a point
                return Math.sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1));
            }
            
            // Calculate the parameter t for the closest point on the line
            const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (length * length)));
            
            // Find the closest point on the line segment
            const closestX = x1 + t * dx;
            const closestY = y1 + t * dy;
            
            // Return distance to closest point
            return Math.sqrt((px - closestX) * (px - closestX) + (py - closestY) * (py - closestY));
        }
        
        function updateBboxCorner(detection, handleIndex, imgX, imgY) {
            const currentLeft = detection.x_center - detection.width / 2;
            const currentTop = detection.y_center - detection.height / 2;
            const currentRight = detection.x_center + detection.width / 2;
            const currentBottom = detection.y_center + detection.height / 2;
            
            let newLeft = currentLeft;
            let newTop = currentTop;
            let newRight = currentRight;
            let newBottom = currentBottom;
            
            switch (handleIndex) {
                // Corner handles (0-3)
                case 0: // Top-left corner
                    newLeft = imgX;
                    newTop = imgY;
                    break;
                case 1: // Top-right corner
                    newRight = imgX;
                    newTop = imgY;
                    break;
                case 2: // Bottom-right corner
                    newRight = imgX;
                    newBottom = imgY;
                    break;
                case 3: // Bottom-left corner
                    newLeft = imgX;
                    newBottom = imgY;
                    break;
                // Side handles (4-7)
                case 4: // Top side
                    newTop = imgY;
                    break;
                case 5: // Right side
                    newRight = imgX;
                    break;
                case 6: // Bottom side
                    newBottom = imgY;
                    break;
                case 7: // Left side
                    newLeft = imgX;
                    break;
            }
            
            detection.x_center = (newLeft + newRight) / 2;
            detection.y_center = (newTop + newBottom) / 2;
            detection.width = Math.abs(newRight - newLeft);
            detection.height = Math.abs(newBottom - newTop);
        }
        
        function setupClassSelect() {
            const classSelect = document.getElementById('classSelect');
            classSelect.innerHTML = '';
            
            classNames.forEach((className, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = className;
                classSelect.appendChild(option);
            });
        }
        
        function setupClassLegend() {
            const classLegend = document.getElementById('classLegend');
            classLegend.innerHTML = '';
            
            classNames.forEach((className, index) => {
                const legendItem = document.createElement('div');
                legendItem.className = 'class-item';
                legendItem.innerHTML = `
                    <div class="color-box" style="background-color: ${classColors[index]}"></div>
                    <span>${className}</span>
                `;
                classLegend.appendChild(legendItem);
            });
        }
        
        function updateStatusDisplay() {
            const statusDisplay = document.getElementById('currentStatus');
            statusDisplay.textContent = `Status: ${currentStatus.charAt(0).toUpperCase() + currentStatus.slice(1)}`;
        }
        
        function updateDetectionInfo() {
            // Update detection count
            const detectionCount = document.getElementById('detectionCount');
            detectionCount.textContent = `Detections: ${detections.length}`;
            
            // Update detailed detection info
            const detectionInfo = document.getElementById('detectionInfo');
            if (selectedDetection !== null && selectedDetection < detections.length) {
                const detection = detections[selectedDetection];
                const className = classNames[detection.class_id] || `Class ${detection.class_id}`;
                let coordinateInfo = '';
                
                if (detection.type === 'polygon') {
                    const pointCount = detection.points ? detection.points.length : 0;
                    coordinateInfo = `${pointCount} points`;
                } else if (detection.type === 'bbox') {
                    coordinateInfo = `Center: (${detection.x_center.toFixed(3)}, ${detection.y_center.toFixed(3)}), Size: ${detection.width.toFixed(3)}×${detection.height.toFixed(3)}`;
                }
                
                detectionInfo.innerHTML = `
                    <strong>Detection Info</strong><br>
                    Selected: ${className}<br>
                    Class ID: ${detection.class_id}<br>
                    Type: ${detection.type}<br>
                    Coordinates: ${coordinateInfo}
                `;
            } else {
                detectionInfo.innerHTML = 'No detection selected';
            }
        }
        
        function updateSelectionInfo() {
            const selectionInfo = document.getElementById('selectionInfo');
            const classSelect = document.getElementById('classSelect');
            const deleteBtn = document.getElementById('deleteBtn');
            const sendBackBtn = document.getElementById('sendBackBtn');
            const bringFrontBtn = document.getElementById('bringFrontBtn');
            
            if (selectedDetections.length > 1) {
                // Multi-selection mode
                selectionInfo.textContent = `Multi-selected: ${selectedDetections.length} objects`;
                classSelect.disabled = false; // Enable for batch class change
                deleteBtn.disabled = false;
                sendBackBtn.disabled = true; // Disable layer operations for multi-select
                bringFrontBtn.disabled = true;
            } else if (selectedDetection !== null) {
                // Single selection mode
                const detection = detections[selectedDetection];
                const className = classNames[detection.class_id] || `Class ${detection.class_id}`;
                selectionInfo.textContent = `Selected: ${detection.type} (${className})`;
                classSelect.value = detection.class_id;
                classSelect.disabled = false;
                deleteBtn.disabled = false;
                sendBackBtn.disabled = false;
                bringFrontBtn.disabled = false;
            } else {
                selectionInfo.textContent = 'No detection selected';
                classSelect.disabled = true;
                deleteBtn.disabled = true;
                sendBackBtn.disabled = true;
                bringFrontBtn.disabled = true;
            }
            
            // Update detection info panel
            updateDetectionInfo();
        }
        
        async function updateStatistics() {
            try {
                const response = await fetch('/api/statistics');
                const data = await response.json();
                
                document.getElementById('statistics').innerHTML = `
                    <div class="stats-item"><span>Total:</span><span>${data.total_images}</span></div>
                    <div class="stats-item"><span>Approved:</span><span>${data.approved}</span></div>
                    <div class="stats-item"><span>Denied:</span><span>${data.denied}</span></div>
                    <div class="stats-item"><span>Pending:</span><span>${data.pending}</span></div>
                    <div class="stats-item"><span>Progress:</span><span>${data.progress_percent}%</span></div>
                `;
            } catch (error) {
                console.error('Error fetching statistics:', error);
            }
        }
        
        function setTool(tool) {
            currentTool = tool;
            
            // Update button states
            document.querySelectorAll('.btn').forEach(btn => {
                btn.classList.remove('btn-active');
                btn.classList.add('btn-secondary');
            });
            
            const toolButton = document.getElementById(tool + 'Tool');
            if (toolButton) {
                toolButton.classList.remove('btn-secondary');
                toolButton.classList.add('btn-active');
            }
            
            // Reset states
            selectedDetection = null;
            selectedDetections = []; // Clear multi-selection
            selectedCorner = null;
            isDrawing = false;
            drawingPoints = [];
            drawingBbox = null;
            isDragMode = (tool === 'drag');
            
            updateSelectionInfo();
            drawCanvas();
        }
        
        function changeSelectedClass() {
            const classSelect = document.getElementById('classSelect');
            const newClassId = parseInt(classSelect.value);
            
            if (selectedDetections.length > 1) {
                // Batch class change for multi-selected objects
                selectedDetections.forEach(detIndex => {
                    if (detIndex < detections.length) {
                        detections[detIndex].class_id = newClassId;
                    }
                });
                updateSelectionInfo();
                drawCanvas();
            } else if (selectedDetection !== null) {
                // Single object class change
                detections[selectedDetection].class_id = newClassId;
                updateSelectionInfo();
                drawCanvas();
            }
        }
        
        async function setImageStatus(status) {
            currentStatus = status;
            updateStatusDisplay();
            
            try {
                await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_index: imageIndex,
                        detections: detections,
                        status: status
                    })
                });
                
                await updateStatistics();
            } catch (error) {
                console.error('Error saving status:', error);
            }
        }
        
        async function finishInspection() {
            if (confirm('Are you sure you want to finish the inspection? This will mark the inspection as complete.')) {
                try {
                    const response = await fetch('/api/finish_inspection', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        alert('Inspection completed successfully!');
                    } else {
                        alert('Error finishing inspection: ' + data.error);
                    }
                } catch (error) {
                    console.error('Error finishing inspection:', error);
                    alert('Error finishing inspection');
                }
            }
        }
        
        function onKeyDown(event) {
            switch(event.key) {
                case 'ArrowLeft':
                    event.preventDefault();
                    previousImage();
                    break;
                case 'ArrowRight':
                    event.preventDefault();
                    nextImage();
                    break;
                case 's':
                case 'S':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        saveDetections();
                    } else {
                        event.preventDefault();
                        setTool('select');
                    }
                    break;
                case 'p':
                case 'P':
                    event.preventDefault();
                    setTool('polygon');
                    break;
                case 'b':
                case 'B':
                    event.preventDefault();
                    setTool('bbox');
                    break;
                case 'd':
                case 'D':
                    event.preventDefault();
                    setTool('drag');
                    break;
                case 'c':
                case 'C':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        copySelected();
                    }
                    break;
                case 'v':
                case 'V':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        pasteSelected();
                    }
                    break;
                case 'Delete':
                case 'Backspace':
                    if (selectedDetection !== null) {
                        deleteSelected();
                    }
                    break;
                case 'x':
                case 'X':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        cleanupDegeneratePolygons();
                    }
                    break;
            }
        }
        
        function deleteSelected() {
            if (selectedDetections.length > 1) {
                // Batch delete multi-selected objects (delete in reverse order to maintain indices)
                selectedDetections.sort((a, b) => b - a).forEach(detIndex => {
                    if (detIndex < detections.length) {
                        detections.splice(detIndex, 1);
                    }
                });
                selectedDetections = [];
                selectedDetection = null;
                selectedCorner = null;
                updateSelectionInfo();
                updateDetectionInfo();
                drawCanvas();
            } else if (selectedDetection !== null) {
                // Single object delete
                detections.splice(selectedDetection, 1);
                selectedDetection = null;
                selectedDetections = [];
                selectedCorner = null;
                updateSelectionInfo();
                updateDetectionInfo();
                drawCanvas();
            }
        }
        
        function cleanupDegeneratePolygons() {
            const scale = Math.min(canvas.width / currentImage.width, canvas.height / currentImage.height);
            const offsetX = (canvas.width - currentImage.width * scale) / 2;
            const offsetY = (canvas.height - currentImage.height * scale) / 2;
            
            let removedCount = 0;
            
            // Filter out degenerate polygons
            detections = detections.filter((detection, index) => {
                if (detection.type === 'polygon') {
                    const points = detection.points.map(p => ({
                        x: offsetX + p[0] * currentImage.width * scale,
                        y: offsetY + p[1] * currentImage.height * scale
                    }));
                    
                    if (isPolygonDegenerate(points)) {
                        if (selectedDetection === index) {
                            selectedDetection = null;
                            selectedCorner = null;
                        } else if (selectedDetection > index) {
                            selectedDetection--;
                        }
                        removedCount++;
                        return false; // Remove this detection
                    }
                }
                return true; // Keep this detection
            });
            
            if (removedCount > 0) {
                console.log(`Removed ${removedCount} degenerate polygon(s)`);
                updateSelectionInfo();
                updateDetectionInfo();
                drawCanvas();
                
                // Show a brief notification
                const notification = document.createElement('div');
                notification.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #ff6600;
                    color: white;
                    padding: 10px 15px;
                    border-radius: 5px;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    z-index: 1000;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                `;
                notification.textContent = `Removed ${removedCount} degenerate polygon(s)`;
                document.body.appendChild(notification);
                
                setTimeout(() => {
                    document.body.removeChild(notification);
                }, 3000);
            } else {
                console.log('No degenerate polygons found');
            }
        }
        
        function copySelected() {
            if (selectedDetection !== null) {
                copiedDetection = JSON.parse(JSON.stringify(detections[selectedDetection]));
                console.log('Copied detection:', copiedDetection);
            }
        }
        
        function pasteSelected() {
            if (copiedDetection !== null) {
                const newDetection = JSON.parse(JSON.stringify(copiedDetection));
                
                // Offset the pasted detection slightly
                if (newDetection.type === 'polygon') {
                    newDetection.points = newDetection.points.map(point => [
                        Math.min(0.95, point[0] + 0.05), // Offset by 5% and clamp to bounds
                        Math.min(0.95, point[1] + 0.05)
                    ]);
                } else if (newDetection.type === 'bbox') {
                    newDetection.x_center = Math.min(0.95, newDetection.x_center + 0.05);
                    newDetection.y_center = Math.min(0.95, newDetection.y_center + 0.05);
                }
                
                detections.push(newDetection);
                selectedDetection = detections.length - 1;
                updateSelectionInfo();
                updateDetectionInfo();
                drawCanvas();
                saveDetections();
                console.log('Pasted detection at index:', selectedDetection);
            }
        }
        
        function sendToBack() {
            console.log('sendToBack called, selectedDetection:', selectedDetection, 'detections.length:', detections.length);
            if (selectedDetection !== null && selectedDetection > 0) {
                console.log('Moving detection to back');
                const detection = detections.splice(selectedDetection, 1)[0];
                detections.unshift(detection);
                selectedDetection = 0;
                updateSelectionInfo();
                updateDetectionInfo();
                drawCanvas();
                saveDetections();
                console.log('Detection moved to back, new selectedDetection:', selectedDetection);
            } else {
                console.log('Cannot send to back: selectedDetection is null or already at back');
            }
        }
        
        function bringToFront() {
            console.log('bringToFront called, selectedDetection:', selectedDetection, 'detections.length:', detections.length);
            if (selectedDetection !== null && selectedDetection < detections.length - 1) {
                console.log('Moving detection to front');
                const detection = detections.splice(selectedDetection, 1)[0];
                detections.push(detection);
                selectedDetection = detections.length - 1;
                updateSelectionInfo();
                updateDetectionInfo();
                drawCanvas();
                saveDetections();
                console.log('Detection moved to front, new selectedDetection:', selectedDetection);
            } else {
                console.log('Cannot bring to front: selectedDetection is null or already at front');
            }
        }
        
        function toggleDarkMode() {
            document.body.classList.toggle('dark-mode');
            const isDarkMode = document.body.classList.contains('dark-mode');
            const toggleBtn = document.querySelector('.dark-mode-toggle');
            toggleBtn.textContent = isDarkMode ? '☀️ Light Mode' : '🌙 Dark Mode';
        }
    </script>
</body>
</html>
        '''
        return template.replace('/api/init/0', f'/api/init/{self.current_iteration}')
    
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