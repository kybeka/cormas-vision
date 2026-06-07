#!/usr/bin/env python3
"""
Main script to run the iterative pseudo-labeling pipeline.

This script provides a simple interface to:
1. Check current dataset status
2. Create backup before starting
3. Run the pseudo-labeling pipeline
4. Monitor progress and results
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from pseudo_labeling_pipeline import PseudoLabelingPipeline

def main():
    print("=" * 60)
    print("YOLO Iterative Pseudo-Labeling Pipeline")
    print("=" * 60)
    
    # Get the main project directory (parent of pseudo_labeling)
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Check current dataset status
    print("\n1. Current Dataset Status:")
    # Import here after changing directory
    sys.path.append(str(Path(__file__).parent))
    from pseudo_utils import count_dataset_sizes, backup_current_state, analyze_iteration_metrics
    initial_counts = count_dataset_sizes()
    
    # Check if unlabeled pool exists and has frames
    unlabeled_pool = Path("unlabeled_pool")
    if not unlabeled_pool.exists() or initial_counts["unlabeled_frames"] == 0:
        print("\n❌ Error: No unlabeled frames found!")
        print("Please ensure the unlabeled_pool directory exists and contains frames.")
        print("You can create it using: python create_unlabeled_pool.py")
        return
    
    print(f"\n✅ Found {initial_counts['unlabeled_frames']} unlabeled frames ready for pseudo-labeling")
    
    # Create backup
    print("\n2. Creating Backup:")
    backup_name = f"pre_pseudo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir = backup_current_state(backup_name)
    print(f"✅ Backup created: {backup_dir}")
    
    # Confirm before starting
    print("\n3. Pipeline Configuration:")
    config_file = "pseudo_config.json"
    if Path(config_file).exists():
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"   Max iterations: {config['max_iterations']}")
        print(f"   Confidence threshold: {config['confidence_threshold']}")
        print(f"   Frames per iteration: {config['frames_per_iteration']}")
        print(f"   Training epochs per iteration: {config['training_params']['epochs']}")
    else:
        print("   Using default configuration")
    
    response = input("\nDo you want to start the pseudo-labeling pipeline? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Pipeline cancelled.")
        return
    
    # Run pipeline
    print("\n4. Starting Pipeline:")
    try:
        # Use full path to config file
        config_path = Path("pseudo_labeling") / config_file
        pipeline = PseudoLabelingPipeline(str(config_path))
        pipeline.run_pipeline()
        
        print("\n5. Pipeline Results:")
        final_counts = count_dataset_sizes()
        
        # Calculate changes
        added_frames = (final_counts["train_images"] + final_counts["val_images"]) - \
                      (initial_counts["train_images"] + initial_counts["val_images"])
        
        print(f"\n📊 Summary:")
        print(f"   Frames added to training: {added_frames}")
        print(f"   Remaining unlabeled: {final_counts['unlabeled_frames']}")
        print(f"   Total labeled frames: {final_counts['train_images'] + final_counts['val_images']}")
        
        # Analyze metrics if available
        print("\n6. Performance Analysis:")
        try:
            analyze_iteration_metrics()
        except Exception as e:
            print(f"Could not analyze metrics: {e}")
        
        print("\n✅ Pseudo-labeling pipeline completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Review the results in pseudo_iterations/")
        print(f"2. Run final training with expanded dataset")
        print(f"3. Evaluate model performance")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print(f"You can restore from backup using:")
        print(f"python pseudo_utils.py restore {backup_name}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print(f"You can restore from backup using:")
        print(f"python pseudo_utils.py restore {backup_name}")
        raise

if __name__ == "__main__":
    main()