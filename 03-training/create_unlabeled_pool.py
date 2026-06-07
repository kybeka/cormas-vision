import os
import shutil
from pathlib import Path

def create_unlabeled_pool():
    # Paths
    source_dir = Path("/Users/kylakaplan/Desktop/cormas-vision/game-data/videos/planet-c-frames")
    labeled_frames_file = Path("/Users/kylakaplan/Desktop/cormas-vision/yolo_labelling_v2/labeled_frames_list.txt")
    unlabeled_pool_dir = Path("/Users/kylakaplan/Desktop/cormas-vision/yolo_labelling_v2/unlabeled_pool")
    
    # Create unlabeled pool directory
    unlabeled_pool_dir.mkdir(exist_ok=True)
    
    # Read labeled frames list
    with open(labeled_frames_file, 'r') as f:
        labeled_frames = set(line.strip() for line in f)
    
    print(f"Found {len(labeled_frames)} already labeled frames")
    
    # Copy all frames except labeled ones
    copied_count = 0
    skipped_count = 0
    
    for frame_file in source_dir.glob("frame_*.jpg"):
        frame_name = frame_file.stem  # e.g., "frame_123"
        
        if frame_name in labeled_frames:
            skipped_count += 1
            continue
            
        # Copy to unlabeled pool
        dest_path = unlabeled_pool_dir / frame_file.name
        shutil.copy2(frame_file, dest_path)
        copied_count += 1
    
    print(f"Created unlabeled pool with {copied_count} frames")
    print(f"Skipped {skipped_count} already labeled frames")
    print(f"Unlabeled pool location: {unlabeled_pool_dir}")
    
    return unlabeled_pool_dir, copied_count

if __name__ == "__main__":
    create_unlabeled_pool()