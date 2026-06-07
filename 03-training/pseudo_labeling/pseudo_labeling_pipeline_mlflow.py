#!/usr/bin/env python3
"""
Iterative Pseudo-Labeling Pipeline with MLflow Integration

This is an enhanced version of the pseudo-labeling pipeline that includes
optional MLflow experiment tracking. It maintains full backward compatibility
with the original pipeline.

Key features:
- Optional MLflow integration (graceful fallback if not available)
- All original functionality preserved
- Enhanced logging and experiment tracking
- Safe to use alongside existing pipeline
"""

import os
import sys
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

# Add parent directory to path for MLflow integration
sys.path.append(str(Path(__file__).parent.parent))

try:
    from mlflow_integration import MLflowTracker, log_training_session
    MLFLOW_INTEGRATION_AVAILABLE = True
except ImportError:
    MLFLOW_INTEGRATION_AVAILABLE = False
    print("⚠️  MLflow integration not available. Running without experiment tracking.")

class PseudoLabelingPipelineMLflow:
    def __init__(self, config_path: str = "pseudo_config.json", enable_mlflow: bool = True):
        """Initialize the pseudo-labeling pipeline with optional MLflow integration."""
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.iteration = 0
        self.metrics_history = []
        
        # Initialize MLflow tracker
        self.mlflow_enabled = enable_mlflow and MLFLOW_INTEGRATION_AVAILABLE
        if self.mlflow_enabled:
            try:
                self.mlflow_tracker = MLflowTracker(
                    experiment_name=f"yolo-pseudo-labeling-{datetime.now().strftime('%Y%m%d')}"
                )
                print("✅ MLflow experiment tracking enabled")
            except Exception as e:
                print(f"⚠️  MLflow initialization failed: {e}")
                self.mlflow_enabled = False
                self.mlflow_tracker = None
        else:
            self.mlflow_tracker = None
            if enable_mlflow:
                print("ℹ️  MLflow integration not available. Install with: pip install mlflow")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file {config_path} not found. Using default configuration.")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing config file: {e}")
            raise
    
    def get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "max_iterations": 5,
            "confidence_threshold": 0.75,
            "frames_per_iteration": 25,
            "min_improvement": 0.005,
            "manual_inspection": True,
            "training_params": {
                "epochs": 25,
                "batch": 8,
                "lr0": 0.001,
                "patience": 8,
                "imgsz": 640
            },
            "data_yaml": "frames.v3i.yolov11/data.yaml",
            "model_path": "yolo11m-obb.pt",
            "unlabeled_pool": "unlabeled_pool",
            "pseudo_dir": "pseudo_iterations"
        }
    
    def setup_directories(self):
        """Create necessary directories for the pipeline."""
        self.unlabeled_pool = Path(self.config["unlabeled_pool"])
        self.pseudo_dir = Path(self.config["pseudo_dir"])
        
        # Create directories if they don't exist
        self.unlabeled_pool.mkdir(exist_ok=True)
        self.pseudo_dir.mkdir(exist_ok=True)
        
        print(f"📁 Unlabeled pool: {self.unlabeled_pool}")
        print(f"📁 Pseudo iterations: {self.pseudo_dir}")
    
    def get_unlabeled_frames(self) -> List[Path]:
        """Get list of available unlabeled frames."""
        return list(self.unlabeled_pool.glob("*.jpg"))
    
    def train_model(self, iteration: int) -> Tuple[YOLO, Dict]:
        """Train YOLO model for current iteration with MLflow logging."""
        print(f"\n=== Training Model - Iteration {iteration} ===")
        
        # Start MLflow run for this iteration
        if self.mlflow_enabled and self.mlflow_tracker:
            self.mlflow_tracker.start_run(
                iteration=iteration,
                tags={
                    "stage": "training",
                    "pipeline_version": "mlflow_enhanced"
                }
            )
            
            # Log configuration parameters
            self.mlflow_tracker.log_params({
                "iteration": iteration,
                "confidence_threshold": self.config["confidence_threshold"],
                "frames_per_iteration": self.config["frames_per_iteration"],
                "training_params": self.config["training_params"],
                "data_yaml": self.config["data_yaml"]
            })
        
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
        
        # Log metrics to MLflow
        if self.mlflow_enabled and self.mlflow_tracker:
            mlflow_metrics = {}
            if metrics["best_fitness"] is not None:
                mlflow_metrics["best_fitness"] = metrics["best_fitness"]
            if metrics["final_epoch"] is not None:
                mlflow_metrics["final_epoch"] = metrics["final_epoch"]
            
            self.mlflow_tracker.log_metrics(mlflow_metrics, step=iteration)
            
            # Log model artifacts
            if iteration > 0 and hasattr(results, 'save_dir'):
                results_dir = Path(results.save_dir)
                if results_dir.exists():
                    # Log model weights
                    weights_dir = results_dir / "weights"
                    if weights_dir.exists():
                        self.mlflow_tracker.log_artifacts([str(weights_dir)], "weights")
                    
                    # Log training plots
                    for plot_file in results_dir.glob("*.png"):
                        self.mlflow_tracker.log_artifacts([str(plot_file)], "plots")
            
            # End MLflow run
            self.mlflow_tracker.end_run()
        
        return model, metrics
    
    def generate_pseudo_labels(self, model: YOLO, frames: List[Path], iteration: int) -> List[Path]:
        """Generate pseudo-labels for selected frames."""
        print(f"\nGenerating pseudo-labels for {len(frames)} frames...")
        
        # Create iteration directories
        iter_dir = self.pseudo_dir / f"iter_{iteration:02d}"
        pseudo_images_dir = iter_dir / "pseudo_images"
        pseudo_labels_dir = iter_dir / "pseudo_labels"
        
        pseudo_images_dir.mkdir(parents=True, exist_ok=True)
        pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        high_confidence_frames = []
        total_detections = 0
        confidence_scores = []
        
        for frame_path in frames:
            # Run inference
            results = model(str(frame_path), conf=self.config["confidence_threshold"])
            
            # Process results
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Copy image to pseudo_images directory
                dest_image = pseudo_images_dir / frame_path.name
                shutil.copy2(frame_path, dest_image)
                
                # Create YOLO format label file
                label_file = pseudo_labels_dir / (frame_path.stem + ".txt")
                
                with open(label_file, 'w') as f:
                    for box in results[0].boxes:
                        # Get normalized coordinates
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Write YOLO format: class x_center y_center width height
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
                        total_detections += 1
                        confidence_scores.append(confidence)
                
                high_confidence_frames.append(dest_image)
        
        # Log pseudo-labeling metrics to MLflow
        if self.mlflow_enabled and self.mlflow_tracker:
            self.mlflow_tracker.start_run(
                iteration=iteration,
                tags={
                    "stage": "pseudo_labeling",
                    "pipeline_version": "mlflow_enhanced"
                }
            )
            
            pseudo_metrics = {
                "frames_processed": len(frames),
                "high_confidence_frames": len(high_confidence_frames),
                "total_detections": total_detections,
                "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0,
                "min_confidence": np.min(confidence_scores) if confidence_scores else 0,
                "max_confidence": np.max(confidence_scores) if confidence_scores else 0
            }
            
            self.mlflow_tracker.log_metrics(pseudo_metrics, step=iteration)
            self.mlflow_tracker.end_run()
        
        print(f"Generated {len(high_confidence_frames)} high-confidence pseudo-labels")
        print(f"Total detections: {total_detections}")
        if confidence_scores:
            print(f"Average confidence: {np.mean(confidence_scores):.3f}")
        
        return high_confidence_frames
    
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
        """Run the complete iterative pseudo-labeling pipeline with MLflow tracking."""
        print("Starting Iterative Pseudo-Labeling Pipeline with MLflow Integration")
        print(f"Max iterations: {self.config['max_iterations']}")
        print(f"Confidence threshold: {self.config['confidence_threshold']}")
        print(f"Frames per iteration: {self.config['frames_per_iteration']}")
        print(f"MLflow tracking: {'✅ Enabled' if self.mlflow_enabled else '❌ Disabled'}")
        
        # Start overall experiment run
        if self.mlflow_enabled and self.mlflow_tracker:
            self.mlflow_tracker.start_run(
                run_name=f"pseudo_labeling_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={
                    "stage": "pipeline_overview",
                    "pipeline_version": "mlflow_enhanced",
                    "total_iterations": str(self.config['max_iterations'])
                }
            )
            
            # Log pipeline configuration
            self.mlflow_tracker.log_params(self.config)
        
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
        
        # End overall experiment run
        if self.mlflow_enabled and self.mlflow_tracker:
            # Log final summary metrics
            final_metrics = {
                "total_iterations_completed": len(self.metrics_history),
                "final_best_fitness": self.metrics_history[-1]["best_fitness"] if self.metrics_history else None,
                "pipeline_duration_minutes": 0  # Could calculate actual duration
            }
            self.mlflow_tracker.log_metrics(final_metrics)
            self.mlflow_tracker.end_run()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED")
        print("="*60)
        print(f"Total iterations: {len(self.metrics_history)}")
        print(f"Metrics saved to: {self.pseudo_dir / 'iteration_metrics.json'}")
        if self.mlflow_enabled:
            print("📊 All experiments logged to MLflow")
            print("   View results with: mlflow ui")

if __name__ == "__main__":
    # Allow enabling/disabling MLflow via command line
    import sys
    enable_mlflow = "--no-mlflow" not in sys.argv
    
    pipeline = PseudoLabelingPipelineMLflow(enable_mlflow=enable_mlflow)
    pipeline.run_pipeline()