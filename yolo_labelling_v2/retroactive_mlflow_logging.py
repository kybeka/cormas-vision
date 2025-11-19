#!/usr/bin/env python3
"""
Retroactive MLflow Logging Script
Logs existing pseudo-labeling iterations (iter_01, iter_02) as MLflow experiments
"""

import mlflow
import mlflow.pytorch
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path
import yaml

def log_iteration_to_mlflow(iteration_dir, iteration_num):
    """
    Log a single iteration's results to MLflow
    """
    print(f"\n=== Logging Iteration {iteration_num} ===")
    
    # Set experiment name
    experiment_name = "Pseudo-Labeling-Pipeline"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"iter_{iteration_num:02d}"):
        # Log basic parameters
        mlflow.log_param("iteration", iteration_num)
        mlflow.log_param("model_type", "YOLOv11-OBB")
        mlflow.log_param("task_type", "pseudo_labeling")
        
        # Load and log training arguments if available
        args_file = iteration_dir / "args.yaml"
        if args_file.exists():
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)
                
            # Log key training parameters
            mlflow.log_param("epochs", args.get('epochs', 'unknown'))
            mlflow.log_param("batch_size", args.get('batch', 'unknown'))
            mlflow.log_param("learning_rate", args.get('lr0', 'unknown'))
            mlflow.log_param("image_size", args.get('imgsz', 'unknown'))
            mlflow.log_param("optimizer", args.get('optimizer', 'unknown'))
            mlflow.log_param("data_path", args.get('data', 'unknown'))
            
            print(f"✓ Logged training parameters from args.yaml")
        
        # Load and log training metrics
        results_file = iteration_dir / "results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            
            # Log final epoch metrics (last row)
            final_metrics = df.iloc[-1]
            
            # Training metrics
            mlflow.log_metric("final_train_box_loss", final_metrics['train/box_loss'])
            mlflow.log_metric("final_train_cls_loss", final_metrics['train/cls_loss'])
            mlflow.log_metric("final_train_dfl_loss", final_metrics['train/dfl_loss'])
            
            # Validation metrics
            mlflow.log_metric("final_val_box_loss", final_metrics['val/box_loss'])
            mlflow.log_metric("final_val_cls_loss", final_metrics['val/cls_loss'])
            mlflow.log_metric("final_val_dfl_loss", final_metrics['val/dfl_loss'])
            
            # Performance metrics
            mlflow.log_metric("precision", final_metrics['metrics/precision(B)'])
            mlflow.log_metric("recall", final_metrics['metrics/recall(B)'])
            mlflow.log_metric("mAP50", final_metrics['metrics/mAP50(B)'])
            mlflow.log_metric("mAP50-95", final_metrics['metrics/mAP50-95(B)'])
            
            # Log best metrics across all epochs
            mlflow.log_metric("best_mAP50", df['metrics/mAP50(B)'].max())
            mlflow.log_metric("best_mAP50-95", df['metrics/mAP50-95(B)'].max())
            mlflow.log_metric("best_precision", df['metrics/precision(B)'].max())
            mlflow.log_metric("best_recall", df['metrics/recall(B)'].max())
            
            print(f"✓ Logged training metrics - Final mAP50: {final_metrics['metrics/mAP50(B)']:.3f}")
            
            # Log epoch-by-epoch metrics for visualization
            for idx, row in df.iterrows():
                epoch = int(row['epoch'])
                mlflow.log_metric("mAP50_by_epoch", row['metrics/mAP50(B)'], step=epoch)
                mlflow.log_metric("precision_by_epoch", row['metrics/precision(B)'], step=epoch)
                mlflow.log_metric("recall_by_epoch", row['metrics/recall(B)'], step=epoch)
                mlflow.log_metric("train_loss_by_epoch", row['train/box_loss'], step=epoch)
                mlflow.log_metric("val_loss_by_epoch", row['val/box_loss'], step=epoch)
        
        # Log artifacts (plots, model weights, etc.)
        artifacts_to_log = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxF1_curve.png",
            "BoxPR_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "labels.jpg",
            "labels_correlogram.jpg"
        ]
        
        logged_artifacts = []
        for artifact in artifacts_to_log:
            artifact_path = iteration_dir / artifact
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
                logged_artifacts.append(artifact)
        
        print(f"✓ Logged {len(logged_artifacts)} artifacts: {', '.join(logged_artifacts)}")
        
        # Log model weights if available
        weights_dir = iteration_dir / "weights"
        if weights_dir.exists():
            best_weights = weights_dir / "best.pt"
            last_weights = weights_dir / "last.pt"
            
            if best_weights.exists():
                mlflow.log_artifact(str(best_weights), "model_weights")
                print(f"✓ Logged best model weights")
            
            if last_weights.exists():
                mlflow.log_artifact(str(last_weights), "model_weights")
                print(f"✓ Logged last model weights")
        
        # Log pseudo-labeling statistics
        pseudo_images_dir = iteration_dir / "pseudo_images"
        pseudo_labels_dir = iteration_dir / "pseudo_labels"
        
        if pseudo_images_dir.exists():
            pseudo_image_count = len(list(pseudo_images_dir.glob("*.jpg")))
            mlflow.log_metric("pseudo_images_count", pseudo_image_count)
            print(f"✓ Logged pseudo-labeling stats - {pseudo_image_count} pseudo images")
        
        # Add tags for easy filtering
        mlflow.set_tag("stage", "completed")
        mlflow.set_tag("iteration_type", "pseudo_labeling")
        mlflow.set_tag("model_architecture", "YOLOv11-OBB")
        
        print(f"✓ Successfully logged iteration {iteration_num} to MLflow")
        
        return mlflow.active_run().info.run_id

def main():
    """
    Main function to log all existing iterations
    """
    print("🚀 Starting Retroactive MLflow Logging")
    print("This will log your existing iter_01 and iter_02 results as MLflow experiments")
    
    # Set MLflow tracking URI to local directory
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
    
    pseudo_iterations_dir = Path("pseudo_iterations")
    
    if not pseudo_iterations_dir.exists():
        print("❌ Error: pseudo_iterations directory not found")
        return
    
    logged_runs = []
    
    # Log iter_01
    iter_01_dir = pseudo_iterations_dir / "iter_01"
    if iter_01_dir.exists():
        try:
            run_id = log_iteration_to_mlflow(iter_01_dir, 1)
            logged_runs.append(("iter_01", run_id))
        except Exception as e:
            print(f"❌ Error logging iter_01: {e}")
    
    # Log iter_02
    iter_02_dir = pseudo_iterations_dir / "iter_02"
    if iter_02_dir.exists():
        try:
            run_id = log_iteration_to_mlflow(iter_02_dir, 2)
            logged_runs.append(("iter_02", run_id))
        except Exception as e:
            print(f"❌ Error logging iter_02: {e}")
    
    print("\n" + "="*50)
    print("🎉 Retroactive Logging Complete!")
    print(f"✓ Successfully logged {len(logged_runs)} iterations to MLflow")
    
    for iteration, run_id in logged_runs:
        print(f"  - {iteration}: {run_id}")
    
    print("\n📊 To view your experiments:")
    print("1. Run: mlflow ui")
    print("2. Open: http://localhost:5000")
    print("3. Navigate to 'Pseudo-Labeling-Pipeline' experiment")
    
    print("\n💡 Your historical data is now tracked without affecting current work!")

if __name__ == "__main__":
    main()