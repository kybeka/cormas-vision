# GroundingDINO Exploration for Game Piece Detection

This directory documents the initial exploration phase using GroundingDINO for automated game piece detection and labeling, conducted prior to adopting the YOLO-based pseudo-labeling approach.

## 🎯 Project Overview

GroundingDINO was explored as a zero-shot object detection solution that combines natural language understanding with visual detection capabilities. The goal was to leverage its text-prompt-based detection to automatically identify and label game pieces without requiring extensive manual annotation.

## 🔬 The Journey: Three Phases of Exploration

### Phase 1: Basic Zero-Shot Detection

Started with GroundingDINO's zero-shot capabilities to avoid manually labeling hundreds of gameplay frames. Used simple prompts like "greencircle" to detect game pieces without training data. Integrated SAM for better segmentation boundaries.

### Phase 2: Fine-Tuning Attempt

Tried fine-tuning GroundingDINO with existing CSV annotations to improve consistency. Built a training pipeline (`quick_ft.py`) with modest settings suitable for Apple Silicon.

### Phase 3: COCO Format Integration

Converted to COCO format for standardized data handling. Built caption enhancement scripts (`augment_coco.py`, `patch_captions.py`) to bridge research tools with practical needs.

**Results:** Over-engineering of the whole solution. Managing COCO datasets and multi-modal inputs became more complex than the original manual labeling problem that I was trying to solve.


## 🚀 The Pivot: Why We Moved to YOLO

After three phases of exploration, the realization was that it was overcomplicated. The original goal was simple: avoid manually labeling hundreds of gameplay frames. Instead, it runed into a whole research project.

GroundingDINO required prompt engineering, multi-modal dataset management, and constant parameter tuning. It was going to require spending more time on the labeling system than it would have been just labeling frames manually.

YOLO offered a straightforward alternative: train on a few manual labels, use the model to suggest labels for new frames, correct mistakes, repeat. No natural language processing, no text-image alignment issues - just practical pseudo-labeling that actually reduced manual work.

## 📁 Directory Structure

```
exploration-labelling-with-groundingdino/
├── experiments/
│   ├── GroundingDINO_v1/          # Initial detection experiments
│   │   ├── 0_croptoboard.py       # Basic detection script
│   │   ├── autolabel.py           # SAM-based auto-labeling
│   │   └── img/                   # Test images
│   ├── GroundingDINO_v2/          # Fine-tuning experiments
│   │   ├── quick_ft.py            # Fine-tuning pipeline
│   │   └── data/                  # Training data
│   ├── Ready_GroundingDINO_v3/    # Production-ready setup
│   │   ├── augment_coco.py        # Caption enhancement
│   │   ├── patch_captions.py      # Caption formatting
│   │   └── Open-GroundingDino/    # Enhanced model version
│   └── segment-anything/          # SAM integration
└── 7frames.v2i.coco/             # COCO dataset
```

## 🛠️ Key Technologies

- **GroundingDINO**: Text-prompted object detection
- **Segment Anything Model (SAM)**: Instance segmentation
- **BERT**: Text encoding and understanding
- **Swin Transformer**: Visual backbone architecture
- **PyTorch**: Deep learning framework
- **COCO Format**: Standard annotation format
- **FiftyOne**: Dataset management and visualization

## 📚 What We Learned

**Simple problems need simple solutions.** Turning the problem of "label game pieces faster" into "build a multi-modal AI system." - the first is a practical problem, the second is a research project.

**Complexity has hidden costs.** Every additional feature adds debugging time and maintenance overhead.

**Iteration speed matters.** Complex systems take longer to experiment with. Simple systems let you try ideas quickly and see what works.

This exploration taught something: a straightforward way to reduce manual labeling work. The GroundingDINO experiments helped us appreciate YOLO's simplicity and guided the final pseudo-labeling approach.