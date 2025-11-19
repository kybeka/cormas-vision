#!/usr/bin/env python3
"""
Iterative Pseudo-Labeling Pipeline for YOLO Training

This script implements an iterative pseudo-labeling approach:
1. Train model on current labeled data
2. Use trained model to predict on unlabeled frames
3. Select high-confidence predictions as pseudo-labels
4. Add pseudo-labeled frames to training set
5. Repeat until convergence or max iterations
"""

import os
import shutil
import random
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
from ultralytics import YOLO
import torch
import cv2

class PseudoLabelingPipeline:
    def __init__(self, config_path: str = "pseudo_config.json"):
        """Initialize the pseudo-labeling pipeline with configuration."""
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.iteration = 0
        self.metrics_history = []
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        default_config = {
            "max_iterations": 5,
            "confidence_threshold": 0.7,
            "frames_per_iteration": 25,
            "min_improvement": 0.01,
            "training_params": {
                "epochs": 30,
                "batch": 8,
                "lr0": 0.001,
                "patience": 10,
                "imgsz": 640
            },
            "data_yaml": "../frames.v3i.yolov11/data.yaml",
            "model_path": "yolo11m-obb.pt",
            "unlabeled_pool": "unlabeled_pool",
            "pseudo_dir": "pseudo_iterations"
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        else:
            # Create default config file
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default config at {config_path}")
            
        return default_config
    
    def setup_directories(self):
        """Create necessary directories for pseudo-labeling iterations."""
        self.pseudo_dir = Path(self.config["pseudo_dir"])
        self.pseudo_dir.mkdir(exist_ok=True)
        
        self.unlabeled_pool = Path(self.config["unlabeled_pool"])
        if not self.unlabeled_pool.exists():
            raise FileNotFoundError(f"Unlabeled pool directory not found: {self.unlabeled_pool}")
    
    def get_unlabeled_frames(self) -> List[Path]:
        """Get list of available unlabeled frames."""
        return list(self.unlabeled_pool.glob("*.jpg"))
    
    def train_model(self, iteration: int) -> Tuple[YOLO, Dict]:
        """Train YOLO model for current iteration."""
        print(f"\n=== Training Model - Iteration {iteration} ===")
        
        # Load model
        model_path = self.config["model_path"]
        if iteration > 0:
            # Use best model from previous iteration
            prev_iter_dir = self.pseudo_dir / f"iter_{iteration-1:02d}"
            best_model = prev_iter_dir / "weights" / "best.pt"
            if best_model.exists():
                model_path = str(best_model)
                print(f"Loading model from previous iteration: {model_path}")
        
        model = YOLO(model_path)
        
        # For iteration 0, use base model without training for initial pseudo-labeling
        if iteration == 0:
            print(f"Using base model {model_path} for initial pseudo-labeling (no training)")
            # Create a dummy results structure for consistency
            results = type('Results', (), {
                'results_dir': str(self.pseudo_dir / f"iter_{iteration:02d}"),
                'best_fitness': None
            })()
            # Create the iteration directory structure
            iter_dir = self.pseudo_dir / f"iter_{iteration:02d}"
            iter_dir.mkdir(exist_ok=True)
            weights_dir = iter_dir / "weights"
            weights_dir.mkdir(exist_ok=True)
            # Copy base model as "best.pt" for consistency
            shutil.copy2(model_path, weights_dir / "best.pt")
        else:
            # Train model for subsequent iterations
            results = model.train(
                data=self.config["data_yaml"],
                project=str(self.pseudo_dir),
                name=f"iter_{iteration:02d}",
                **self.config["training_params"],
                device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
            )
        
        # Extract metrics
        metrics = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "best_fitness": float(results.best_fitness) if hasattr(results, 'best_fitness') and results.best_fitness is not None else None,
            "final_epoch": results.epoch if hasattr(results, 'epoch') else None
        }
        
        return model, metrics
    
    def generate_pseudo_labels(self, model: YOLO, frames: List[Path], iteration: int) -> List[Path]:
        """Generate pseudo-labels for selected frames."""
        print(f"\nGenerating pseudo-labels for {len(frames)} frames...")
        
        iter_dir = self.pseudo_dir / f"iter_{iteration:02d}"
        pseudo_images_dir = iter_dir / "pseudo_images"
        pseudo_labels_dir = iter_dir / "pseudo_labels"
        
        pseudo_images_dir.mkdir(parents=True, exist_ok=True)
        pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        successful_frames = []
        confidence_threshold = self.config["confidence_threshold"]
        
        for frame_path in frames:
            try:
                # Run inference
                results = model(str(frame_path))
                
                # Check if any detections meet confidence threshold
                high_conf_detections = []
                for result in results:
                    if result.obb is not None:  # OBB detections
                        boxes = result.obb.xyxyxyxy.cpu().numpy()
                        confidences = result.obb.conf.cpu().numpy()
                        classes = result.obb.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(boxes, confidences, classes):
                            if conf >= confidence_threshold:
                                high_conf_detections.append((box, conf, cls))
                
                # If we have high-confidence detections, save as pseudo-label
                if high_conf_detections:
                    # Copy image
                    dest_image = pseudo_images_dir / frame_path.name
                    shutil.copy2(frame_path, dest_image)
                    
                    # Create label file
                    label_name = frame_path.stem + ".txt"
                    label_path = pseudo_labels_dir / label_name
                    
                    # Get image dimensions for normalization
                    img = cv2.imread(str(frame_path))
                    h, w = img.shape[:2]
                    
                    with open(label_path, 'w') as f:
                        for box, conf, cls in high_conf_detections:
                            # Normalize coordinates
                            norm_box = box.copy()
                            norm_box[:, 0] /= w  # x coordinates
                            norm_box[:, 1] /= h  # y coordinates
                            
                            # Format: class x1 y1 x2 y2 x3 y3 x4 y4
                            coords = norm_box.flatten()
                            f.write(f"{int(cls)} {' '.join(map(str, coords))}\n")
                    
                    successful_frames.append(dest_image)
                    
            except Exception as e:
                print(f"Error processing {frame_path.name}: {e}")
                continue
        
        print(f"Successfully generated {len(successful_frames)} pseudo-labels")
        return successful_frames
    
    def select_frames_for_labeling(self, available_frames: List[Path], iteration: int) -> List[Path]:
        """Select frames for pseudo-labeling in current iteration with progressive sizing."""
        # Progressive iteration sizes: 25, 50, 100, 200, 400...
        base_frames = self.config["frames_per_iteration"]  # 25
        frames_per_iter = base_frames * (2 ** iteration)
        
        print(f"Target frames for iteration {iteration}: {frames_per_iter}")
        
        if len(available_frames) <= frames_per_iter:
            print(f"Using all available frames: {len(available_frames)}")
            return available_frames
        
        # Random selection for now - could implement more sophisticated strategies
        random.seed(42 + iteration)  # Reproducible but different per iteration
        selected = random.sample(available_frames, frames_per_iter)
        print(f"Selected {len(selected)} frames from {len(available_frames)} available")
        return selected
    
    def update_training_data(self, pseudo_images: List[Path], iteration: int):
        """Add pseudo-labeled data to training set."""
        if not pseudo_images:
            print("No pseudo-labels to add to training set")
            return
        
        print(f"\nAdding {len(pseudo_images)} pseudo-labeled frames to training set...")
        
        # Paths to training directories
        train_images_dir = Path(self.config["data_yaml"]).parent / "train" / "images"
        train_labels_dir = Path(self.config["data_yaml"]).parent / "train" / "labels"
        
        iter_dir = self.pseudo_dir / f"iter_{iteration:02d}"
        pseudo_images_dir = iter_dir / "pseudo_images"
        pseudo_labels_dir = iter_dir / "pseudo_labels"
        
        # Copy pseudo-labeled data to training set
        for img_path in pseudo_images:
            # Copy image
            dest_img = train_images_dir / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            src_label = pseudo_labels_dir / label_name
            dest_label = train_labels_dir / label_name
            
            if src_label.exists():
                shutil.copy2(src_label, dest_label)
            
            # Remove from unlabeled pool
            original_frame = self.unlabeled_pool / img_path.name
            if original_frame.exists():
                original_frame.unlink()
    
    def save_metrics(self, metrics: Dict):
        """Save iteration metrics to file."""
        self.metrics_history.append(metrics)
        
        metrics_file = self.pseudo_dir / "iteration_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def check_convergence(self) -> bool:
        """Check if training has converged based on metrics."""
        if len(self.metrics_history) < 2:
            return False
        
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        if current["best_fitness"] is None or previous["best_fitness"] is None:
            return False
        
        improvement = current["best_fitness"] - previous["best_fitness"]
        min_improvement = self.config["min_improvement"]
        
        print(f"Fitness improvement: {improvement:.4f} (threshold: {min_improvement})")
        return improvement < min_improvement
    
    def wait_for_inspection(self, iteration: int):
        """Wait for user to inspect and adjust pseudo-labels through the web interface."""
        print(f"\n{'='*60}")
        print(f"PSEUDO-LABEL INSPECTION - ITERATION {iteration}")
        print(f"{'='*60}")
        print(f"Pseudo-labels have been generated for iteration {iteration}.")
        print(f"Please inspect and adjust the labels using the web interface at:")
        print(f"http://localhost:5000")
        print(f"")
        print(f"The simple_inspector should be running with iteration {iteration} data.")
        print(f"Review the {len(list((self.pseudo_dir / f'iter_{iteration:02d}' / 'pseudo_images').glob('*.jpg')))} pseudo-labeled frames.")
        print(f"Make any necessary corrections and save your changes.")
        print(f"")
        input("Press ENTER when you have finished inspecting and adjusting the pseudo-labels...")
        print("Continuing with pipeline...")
    
    def run_pipeline(self):
        """Run the complete iterative pseudo-labeling pipeline."""
        print("Starting Iterative Pseudo-Labeling Pipeline")
        print(f"Max iterations: {self.config['max_iterations']}")
        print(f"Confidence threshold: {self.config['confidence_threshold']}")
        print(f"Frames per iteration: {self.config['frames_per_iteration']}")
        
        for iteration in range(self.config["max_iterations"]):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{self.config['max_iterations']}")
            print(f"{'='*60}")
            
            # Train model
            model, metrics = self.train_model(iteration)
            self.save_metrics(metrics)
            
            # Check convergence (skip for first iteration)
            if iteration > 0 and self.check_convergence():
                print("\nConvergence detected. Stopping pipeline.")
                break
            
            # Get available unlabeled frames
            available_frames = self.get_unlabeled_frames()
            if not available_frames:
                print("\nNo more unlabeled frames available. Stopping pipeline.")
                break
            
            # Select frames for this iteration
            selected_frames = self.select_frames_for_labeling(available_frames, iteration)
            print(f"Selected {len(selected_frames)} frames for pseudo-labeling")
            
            # Generate pseudo-labels
            pseudo_images = self.generate_pseudo_labels(model, selected_frames, iteration)
            
            # Wait for user inspection and adjustment of pseudo-labels
            if pseudo_images:
                self.wait_for_inspection(iteration)
                self.update_training_data(pseudo_images, iteration)
            else:
                print("No high-confidence pseudo-labels generated. Stopping pipeline.")
                break
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED")
        print("="*60)
        print(f"Total iterations: {len(self.metrics_history)}")
        print(f"Metrics saved to: {self.pseudo_dir / 'iteration_metrics.json'}")

if __name__ == "__main__":
    pipeline = PseudoLabelingPipeline()
    pipeline.run_pipeline()