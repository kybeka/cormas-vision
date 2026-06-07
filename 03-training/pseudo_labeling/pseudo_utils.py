#!/usr/bin/env python3
"""
Utility functions for pseudo-labeling pipeline management and monitoring.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import shutil

def analyze_iteration_metrics(metrics_file: str = "pseudo_iterations/iteration_metrics.json"):
    """Analyze and visualize iteration metrics."""
    if not Path(metrics_file).exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics:
        print("No metrics data found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(metrics)
    
    # Print summary
    print("\n=== Pseudo-Labeling Iteration Summary ===")
    print(f"Total iterations: {len(df)}")
    print(f"Best fitness achieved: {df['best_fitness'].max():.4f}")
    print(f"Final fitness: {df['best_fitness'].iloc[-1]:.4f}")
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    
    # Fitness over iterations
    plt.subplot(2, 2, 1)
    plt.plot(df['iteration'], df['best_fitness'], 'b-o')
    plt.title('Model Fitness Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    
    # Epochs per iteration
    plt.subplot(2, 2, 2)
    plt.bar(df['iteration'], df['final_epoch'])
    plt.title('Training Epochs per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Final Epoch')
    plt.grid(True)
    
    # Fitness improvement
    if len(df) > 1:
        improvements = df['best_fitness'].diff().fillna(0)
        plt.subplot(2, 2, 3)
        plt.bar(df['iteration'], improvements)
        plt.title('Fitness Improvement per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Improvement')
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('pseudo_iterations/metrics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def count_dataset_sizes():
    """Count current dataset sizes across train/val/unlabeled."""
    base_path = Path("frames.v3i.yolov11")
    unlabeled_path = Path("unlabeled_pool")
    
    # If we're in pseudo_labeling directory, go up one level
    if Path.cwd().name == "pseudo_labeling":
        base_path = Path("..") / base_path
        unlabeled_path = Path("..") / unlabeled_path
    
    counts = {
        "train_images": len(list((base_path / "train" / "images").glob("*.jpg"))),
        "train_labels": len(list((base_path / "train" / "labels").glob("*.txt"))),
        "val_images": len(list((base_path / "valid" / "images").glob("*.jpg"))),
        "val_labels": len(list((base_path / "valid" / "labels").glob("*.txt"))),
        "unlabeled_frames": len(list(unlabeled_path.glob("*.jpg"))) if unlabeled_path.exists() else 0
    }
    
    print("\n=== Dataset Size Summary ===")
    print(f"Training images: {counts['train_images']}")
    print(f"Training labels: {counts['train_labels']}")
    print(f"Validation images: {counts['val_images']}")
    print(f"Validation labels: {counts['val_labels']}")
    print(f"Unlabeled frames: {counts['unlabeled_frames']}")
    print(f"Total labeled: {counts['train_images'] + counts['val_images']}")
    
    return counts

def backup_current_state(backup_name: str = None):
    """Create a backup of current training state."""
    if backup_name is None:
        from datetime import datetime
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    backup_dir = Path("backups") / backup_name
    
    # Adjust paths if we're in pseudo_labeling directory
    if Path.cwd().name == "pseudo_labeling":
        backup_dir = Path("..") / backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup training data
    train_dir = Path("frames.v3i.yolov11")
    if train_dir.exists():
        shutil.copytree(train_dir, backup_dir / "frames.v3i.yolov11")
    
    # Backup unlabeled pool
    unlabeled_dir = Path("unlabeled_pool")
    if unlabeled_dir.exists():
        shutil.copytree(unlabeled_dir, backup_dir / "unlabeled_pool")
    
    # Backup pseudo iterations
    pseudo_dir = Path("pseudo_iterations")
    if pseudo_dir.exists():
        shutil.copytree(pseudo_dir, backup_dir / "pseudo_iterations")
    
    # Backup config and scripts
    for file in ["pseudo_config.json", "train_v2.py", "labeled_frames_list.txt"]:
        if Path(file).exists():
            shutil.copy2(file, backup_dir / file)
    
    print(f"Backup created: {backup_dir}")
    return backup_dir

def restore_from_backup(backup_name: str):
    """Restore training state from backup."""
    backup_dir = Path("backups") / backup_name
    
    if not backup_dir.exists():
        print(f"Backup not found: {backup_dir}")
        return False
    
    print(f"Restoring from backup: {backup_dir}")
    
    # Restore training data
    if (backup_dir / "frames.v3i.yolov11").exists():
        if Path("frames.v3i.yolov11").exists():
            shutil.rmtree("frames.v3i.yolov11")
        shutil.copytree(backup_dir / "frames.v3i.yolov11", "frames.v3i.yolov11")
    
    # Restore unlabeled pool
    if (backup_dir / "unlabeled_pool").exists():
        if Path("unlabeled_pool").exists():
            shutil.rmtree("unlabeled_pool")
        shutil.copytree(backup_dir / "unlabeled_pool", "unlabeled_pool")
    
    # Restore pseudo iterations
    if (backup_dir / "pseudo_iterations").exists():
        if Path("pseudo_iterations").exists():
            shutil.rmtree("pseudo_iterations")
        shutil.copytree(backup_dir / "pseudo_iterations", "pseudo_iterations")
    
    # Restore config files
    for file in ["pseudo_config.json", "train_v2.py", "labeled_frames_list.txt"]:
        backup_file = backup_dir / file
        if backup_file.exists():
            shutil.copy2(backup_file, file)
    
    print("Restore completed")
    return True

def clean_pseudo_iterations(keep_best: bool = True):
    """Clean up pseudo iteration directories to save space."""
    pseudo_dir = Path("pseudo_iterations")
    
    if not pseudo_dir.exists():
        print("No pseudo iterations directory found")
        return
    
    # Get all iteration directories
    iter_dirs = [d for d in pseudo_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")]
    
    if not iter_dirs:
        print("No iteration directories found")
        return
    
    print(f"Found {len(iter_dirs)} iteration directories")
    
    if keep_best:
        # Find best iteration based on metrics
        metrics_file = pseudo_dir / "iteration_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            best_iter = max(metrics, key=lambda x: x.get('best_fitness', 0))
            best_iter_name = f"iter_{best_iter['iteration']:02d}"
            
            print(f"Keeping best iteration: {best_iter_name} (fitness: {best_iter['best_fitness']:.4f})")
            
            # Remove all except best
            for iter_dir in iter_dirs:
                if iter_dir.name != best_iter_name:
                    print(f"Removing {iter_dir.name}")
                    shutil.rmtree(iter_dir)
        else:
            print("No metrics file found, keeping last iteration")
            # Keep only the last iteration
            iter_dirs.sort()
            for iter_dir in iter_dirs[:-1]:
                print(f"Removing {iter_dir.name}")
                shutil.rmtree(iter_dir)
    else:
        # Remove all iteration directories
        for iter_dir in iter_dirs:
            print(f"Removing {iter_dir.name}")
            shutil.rmtree(iter_dir)
    
    print("Cleanup completed")

def list_available_backups():
    """List all available backups."""
    backup_dir = Path("backups")
    
    if not backup_dir.exists():
        print("No backups directory found")
        return []
    
    backups = [d.name for d in backup_dir.iterdir() if d.is_dir()]
    backups.sort(reverse=True)  # Most recent first
    
    print("\n=== Available Backups ===")
    for backup in backups:
        backup_path = backup_dir / backup
        size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        print(f"{backup} ({size_mb:.1f} MB)")
    
    return backups

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pseudo_utils.py <command>")
        print("Commands: analyze, count, backup, restore, clean, list_backups")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "analyze":
        analyze_iteration_metrics()
    elif command == "count":
        count_dataset_sizes()
    elif command == "backup":
        backup_name = sys.argv[2] if len(sys.argv) > 2 else None
        backup_current_state(backup_name)
    elif command == "restore":
        if len(sys.argv) < 3:
            print("Usage: python pseudo_utils.py restore <backup_name>")
            sys.exit(1)
        restore_from_backup(sys.argv[2])
    elif command == "clean":
        keep_best = len(sys.argv) < 3 or sys.argv[2].lower() != "all"
        clean_pseudo_iterations(keep_best)
    elif command == "list_backups":
        list_available_backups()
    else:
        print(f"Unknown command: {command}")