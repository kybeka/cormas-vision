#!/usr/bin/env python3
"""
Pseudo-Label Inspection Tool

This script allows you to visually inspect pseudo-labeled images before they are
added to the training dataset. It displays images with their predicted bounding boxes
and allows you to approve/reject individual pseudo-labels.
"""

import os
import cv2
import json
import shutil
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

class PseudoLabelInspector:
    def __init__(self, pseudo_dir: str = "pseudo_iterations"):
        self.pseudo_dir = Path(pseudo_dir)
        self.class_names = self.load_class_names()
        self.colors = self.generate_colors(len(self.class_names))
        
    def load_class_names(self) -> List[str]:
        """Load class names from data.yaml"""
        data_yaml = Path("frames.v3i.yolov11/data.yaml")
        if data_yaml.exists():
            import yaml
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            return data.get('names', [])
        else:
            # Default class names for chess pieces
            return ['blue-pawn', 'board', 'green-token', 'inner-board', 'red-pawn', 
                   'white-pawn', 'yellow-pawn', 'blue-token', 'red-token', 
                   'white-token', 'yellow-token', 'green-pawn']
    
    def generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def parse_label_file(self, label_path: Path) -> List[Dict]:
        """Parse YOLO OBB label file"""
        detections = []
        if not label_path.exists():
            return detections
            
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 9:  # class + 8 coordinates
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:9]]
                    detections.append({
                        'class_id': cls_id,
                        'coordinates': coords,
                        'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                    })
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on image"""
        h, w = image.shape[:2]
        annotated = image.copy()
        
        for det in detections:
            cls_id = det['class_id']
            coords = det['coordinates']
            class_name = det['class_name']
            
            # Convert normalized coordinates to pixel coordinates
            points = np.array(coords).reshape(-1, 2)
            points[:, 0] *= w  # x coordinates
            points[:, 1] *= h  # y coordinates
            points = points.astype(np.int32)
            
            # Draw polygon
            color = self.colors[cls_id % len(self.colors)]
            cv2.polylines(annotated, [points], True, color, 2)
            
            # Draw class label
            center_x = int(np.mean(points[:, 0]))
            center_y = int(np.mean(points[:, 1]))
            
            # Background for text
            text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, 
                         (center_x - text_size[0]//2 - 5, center_y - text_size[1] - 5),
                         (center_x + text_size[0]//2 + 5, center_y + 5),
                         color, -1)
            
            # Text
            cv2.putText(annotated, class_name, 
                       (center_x - text_size[0]//2, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def inspect_iteration(self, iteration: int) -> Dict[str, bool]:
        """Inspect pseudo-labels for a specific iteration"""
        iter_dir = self.pseudo_dir / f"iter_{iteration:02d}"
        pseudo_images_dir = iter_dir / "pseudo_images"
        pseudo_labels_dir = iter_dir / "pseudo_labels"
        
        if not pseudo_images_dir.exists() or not pseudo_labels_dir.exists():
            print(f"No pseudo-labels found for iteration {iteration}")
            return {}
        
        image_files = list(pseudo_images_dir.glob("*.jpg"))
        if not image_files:
            print(f"No images found in {pseudo_images_dir}")
            return {}
        
        print(f"\nInspecting {len(image_files)} pseudo-labeled images from iteration {iteration}")
        print("Controls:")
        print("  'a' - Approve image (add to training)")
        print("  'r' - Reject image (don't add to training)")
        print("  'q' - Quit inspection")
        print("  'n' - Next image")
        print("  'p' - Previous image")
        
        approved_images = {}
        current_idx = 0
        
        while current_idx < len(image_files):
            img_path = image_files[current_idx]
            label_path = pseudo_labels_dir / (img_path.stem + ".txt")
            
            # Load and display image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not load image: {img_path}")
                current_idx += 1
                continue
            
            # Parse detections
            detections = self.parse_label_file(label_path)
            
            # Draw detections
            annotated = self.draw_detections(image, detections)
            
            # Resize for display if too large
            h, w = annotated.shape[:2]
            if h > 800 or w > 1200:
                scale = min(800/h, 1200/w)
                new_h, new_w = int(h*scale), int(w*scale)
                annotated = cv2.resize(annotated, (new_w, new_h))
            
            # Display info
            title = f"Iteration {iteration} - Image {current_idx+1}/{len(image_files)} - {img_path.name}"
            info_text = f"Detections: {len(detections)}"
            
            # Add text overlay
            cv2.putText(annotated, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show current approval status
            status = approved_images.get(img_path.name, "Pending")
            status_color = (0, 255, 0) if status == "Approved" else (0, 0, 255) if status == "Rejected" else (0, 255, 255)
            cv2.putText(annotated, f"Status: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            cv2.imshow("Pseudo-Label Inspector", annotated)
            
            # Handle user input
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('a'):  # Approve
                approved_images[img_path.name] = True
                print(f"✓ Approved: {img_path.name}")
                current_idx += 1
            elif key == ord('r'):  # Reject
                approved_images[img_path.name] = False
                print(f"✗ Rejected: {img_path.name}")
                current_idx += 1
            elif key == ord('n'):  # Next
                current_idx = min(current_idx + 1, len(image_files) - 1)
            elif key == ord('p'):  # Previous
                current_idx = max(current_idx - 1, 0)
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        
        # Show summary
        approved_count = sum(1 for v in approved_images.values() if v)
        rejected_count = sum(1 for v in approved_images.values() if not v)
        pending_count = len(image_files) - len(approved_images)
        
        print(f"\nInspection Summary:")
        print(f"  Approved: {approved_count}")
        print(f"  Rejected: {rejected_count}")
        print(f"  Pending: {pending_count}")
        
        return approved_images
    
    def apply_approvals(self, iteration: int, approvals: Dict[str, bool]):
        """Apply user approvals by moving approved images to training set"""
        iter_dir = self.pseudo_dir / f"iter_{iteration:02d}"
        pseudo_images_dir = iter_dir / "pseudo_images"
        pseudo_labels_dir = iter_dir / "pseudo_labels"
        
        # Create approved/rejected directories
        approved_dir = iter_dir / "approved"
        rejected_dir = iter_dir / "rejected"
        approved_dir.mkdir(exist_ok=True)
        rejected_dir.mkdir(exist_ok=True)
        
        approved_images_dir = approved_dir / "images"
        approved_labels_dir = approved_dir / "labels"
        rejected_images_dir = rejected_dir / "images"
        rejected_labels_dir = rejected_dir / "labels"
        
        for dir_path in [approved_images_dir, approved_labels_dir, rejected_images_dir, rejected_labels_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Training directories
        train_images_dir = Path("frames.v3i.yolov11/train/images")
        train_labels_dir = Path("frames.v3i.yolov11/train/labels")
        
        approved_count = 0
        rejected_count = 0
        
        for img_name, approved in approvals.items():
            img_path = pseudo_images_dir / img_name
            label_path = pseudo_labels_dir / (Path(img_name).stem + ".txt")
            
            if not img_path.exists():
                continue
            
            if approved:
                # Move to approved directory
                shutil.copy2(img_path, approved_images_dir / img_name)
                if label_path.exists():
                    shutil.copy2(label_path, approved_labels_dir / (Path(img_name).stem + ".txt"))
                
                # Add to training set
                shutil.copy2(img_path, train_images_dir / img_name)
                if label_path.exists():
                    shutil.copy2(label_path, train_labels_dir / (Path(img_name).stem + ".txt"))
                
                # Remove from unlabeled pool
                unlabeled_frame = Path("unlabeled_pool") / img_name
                if unlabeled_frame.exists():
                    unlabeled_frame.unlink()
                
                approved_count += 1
            else:
                # Move to rejected directory
                shutil.copy2(img_path, rejected_images_dir / img_name)
                if label_path.exists():
                    shutil.copy2(label_path, rejected_labels_dir / (Path(img_name).stem + ".txt"))
                
                rejected_count += 1
        
        print(f"\nApplied approvals:")
        print(f"  Added {approved_count} images to training set")
        print(f"  Rejected {rejected_count} images")
        
        # Save approval log
        approval_log = {
            "iteration": iteration,
            "timestamp": str(datetime.now()),
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "approvals": approvals
        }
        
        log_file = iter_dir / "approval_log.json"
        with open(log_file, 'w') as f:
            json.dump(approval_log, f, indent=2)

def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Inspect pseudo-labeled images")
    parser.add_argument("--iteration", "-i", type=int, required=True,
                       help="Iteration number to inspect")
    parser.add_argument("--auto-apply", action="store_true",
                       help="Automatically apply approvals to training set")
    
    args = parser.parse_args()
    
    inspector = PseudoLabelInspector()
    approvals = inspector.inspect_iteration(args.iteration)
    
    if approvals and args.auto_apply:
        inspector.apply_approvals(args.iteration, approvals)
    elif approvals:
        print("\nTo apply these approvals to the training set, run:")
        print(f"python inspect_pseudo_labels.py -i {args.iteration} --auto-apply")

if __name__ == "__main__":
    main()