#!/usr/bin/env python3
import cv2
from pathlib import Path

video_path = "../../cormas-vision/game-data/oleks-new-vid.TS.mp4"

print(f"Checking video: {video_path}")
print(f"File exists: {Path(video_path).exists()}")
print(f"File size: {Path(video_path).stat().st_size / (1024*1024):.1f} MB")

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("ERROR: Could not open video")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Try to read a few frames to see if there are issues
    frame_count = 0
    readable_frames = 0
    
    while frame_count < 100:  # Check first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        if frame is not None:
            readable_frames += 1
        frame_count += 1
    
    print(f"\nFrame Reading Test:")
    print(f"  Attempted to read: {frame_count} frames")
    print(f"  Successfully read: {readable_frames} frames")
    
    cap.release()