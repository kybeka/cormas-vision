#!/usr/bin/env python3
"""
Video Frame Extraction and YOLO Inference Script

This script:
1. Extracts frames from the mentor's video
2. Runs YOLO inference on each frame using the best available model
3. Saves results with visualizations for easy review
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import shutil
from datetime import datetime
import subprocess
import sys

def extract_frames_from_video_robust(video_path, output_dir, frames_per_second=1):
    """
    Extract frames from video using a more robust approach that handles codec issues.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frames_per_second: Number of frames to extract per second (default: 1)
    """
    print(f"Extracting frames from: {video_path}")
    
    # Create output directory
    frames_dir = Path(output_dir) / "extracted_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # First, get video info
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"Video info: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s duration")
    
    # Calculate how many frames we should extract
    expected_frames = int(duration * frames_per_second)
    print(f"Expected to extract ~{expected_frames} frames (1 per second)")
    
    # Use a more robust extraction approach
    # Try to seek to specific timestamps and extract frames
    extracted_count = 0
    failed_extractions = 0
    
    for second in range(int(duration)):
        # Try to extract frame at this second
        cap = cv2.VideoCapture(str(video_path))
        
        # Seek to the specific timestamp
        timestamp_ms = second * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_filename = frames_dir / f"frame_second_{second:06d}.jpg"
            try:
                success = cv2.imwrite(str(frame_filename), frame)
                if success:
                    extracted_count += 1
                else:
                    failed_extractions += 1
            except Exception as e:
                print(f"Warning: Error saving frame at {second}s: {e}")
                failed_extractions += 1
        else:
            failed_extractions += 1
            
        cap.release()
        
        # Progress update every 60 seconds
        if second % 60 == 0 and second > 0:
            print(f"Processed {second}/{int(duration)} seconds... (extracted: {extracted_count}, failed: {failed_extractions})")
    
    print(f"Extracted {extracted_count} frames to {frames_dir}")
    print(f"Failed extractions: {failed_extractions}")
    return frames_dir

def run_inference_on_frames(frames_dir, model_path, output_dir):
    """
    Run YOLO inference on extracted frames.
    
    Args:
        frames_dir: Directory containing extracted frames
        model_path: Path to YOLO model (.pt file)
        output_dir: Directory to save inference results
    """
    print(f"\nRunning inference with model: {model_path}")
    
    # Create output directories
    results_dir = Path(output_dir) / "inference_results"
    predictions_dir = results_dir / "predictions"
    visualizations_dir = results_dir / "visualizations"
    
    predictions_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Get all frame files
    frame_files = sorted(list(Path(frames_dir).glob("*.jpg")))
    print(f"Found {len(frame_files)} frames to process")
    
    if len(frame_files) == 0:
        print("No frames found to process!")
        return results_dir
    
    # Process each frame
    total_detections = 0
    frames_with_detections = 0
    detection_details = []
    
    for i, frame_file in enumerate(frame_files, 1):
        print(f"Processing frame {i}/{len(frame_files)}: {frame_file.name}")
        
        # Run inference
        results = model(str(frame_file))
        
        # Process results
        frame_detections = 0
        confidences = []
        classes = []
        
        for result in results:
            if result.obb is not None and len(result.obb) > 0:
                frame_detections = len(result.obb.conf)
                confidences = result.obb.conf.cpu().numpy().tolist()
                classes = result.obb.cls.cpu().numpy().astype(int).tolist()
                
                # Save annotated image
                annotated_frame = result.plot()
                output_path = visualizations_dir / f"{frame_file.stem}_annotated.jpg"
                cv2.imwrite(str(output_path), annotated_frame)
        
        # Save predictions
        pred_file = predictions_dir / f"{frame_file.stem}_predictions.txt"
        with open(pred_file, 'w') as f:
            f.write(f"Frame: {frame_file.name}\n")
            f.write(f"Detections: {frame_detections}\n")
            if confidences:
                f.write(f"Confidences: {', '.join([f'{c:.3f}' for c in confidences])}\n")
                f.write(f"Classes: {', '.join(map(str, classes))}\n")
        
        # Update statistics
        total_detections += frame_detections
        if frame_detections > 0:
            frames_with_detections += 1
            
        detection_details.append({
            'frame': frame_file.name,
            'detections': frame_detections,
            'confidences': confidences,
            'classes': classes
        })
    
    # Generate summary
    detection_rate = (frames_with_detections / len(frame_files)) * 100 if frame_files else 0
    
    summary_file = results_dir / "detection_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Video Inference Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Total frames processed: {len(frame_files)}\n\n")
        
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Frames with detections: {frames_with_detections}/{len(frame_files)}\n")
        f.write(f"Detection rate: {detection_rate:.1f}%\n\n")
        
        f.write("Per-frame details:\n")
        for detail in detection_details:
            f.write(f"{detail['frame']}: {detail['detections']} detections\n")
    
    print(f"\nInference complete! Results saved to {results_dir}")
    print(f"- Visualizations: {results_dir}/visualizations")
    print(f"- Predictions: {results_dir}/predictions")
    print(f"- Summary: {results_dir}/detection_summary.txt")
    
    return results_dir

def main():
    # Configuration
    video_path = "/Users/kylakaplan/Desktop/cormas-vision/game-data/oleks-new-vid.TS.mp4"
    output_dir = Path("./")
    
    # Find best model - try multiple locations
    model_candidates = [
        "../yolo_labelling_v2/pseudo_iterations/iter_02/weights/best.pt",
        "../yolo_labelling_v2/pseudo_iterations/iter_01/weights/best.pt", 
        "../yolo_labelling_v2/runs/obb/train10/weights/best.pt",
        "../yolo_labelling_v2/yolo11m-obb.pt"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if Path(candidate).exists():
            model_path = candidate
            print(f"Using model: {model_path}")
            break
    
    if not model_path:
        raise FileNotFoundError("No suitable model found!")
    
    # Check if video exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    print(f"Starting video inference pipeline...")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    
    # Step 1: Extract frames (1 frame per second)
    frames_dir = extract_frames_from_video_robust(video_path, output_dir, frames_per_second=1)
    
    # Step 2: Run inference
    results_dir = run_inference_on_frames(frames_dir, model_path, output_dir)
    
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Check the results in: {results_dir}")
    print(f"Visualizations are in: {results_dir}/visualizations/")
    print(f"Ready for your meeting!")

if __name__ == "__main__":
    main()