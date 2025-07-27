## Week 05 – 2025-07-21

### Weekly Focus

**Context:**  
This week was all about pushing forward with Planet-C labeling and preparing the detection pipeline for the next training round. Also made progress on integration tasks and planning materials.

### Tasks
- Focused on Planet-C frame annotation:
  - Manually labeled 150 planet-c training data.
  - Tried fine-tuning Grounding DINO but ran into setup issues.
  - Shifted to improving YOLO11 model (with oriented bounding boxed):
    - Used Canny edge detection to find the board automatically.
    - Started building the filtering system (confidence + inside-board logic).
    - Ran low-threshold inference to generate pseudo-labels.
    - Now setting up a closed-loop feedback system of annotation in order to speed up the whole process
- Need to figure out how to set-up a basic log to track each round of model performance.
- Worked on Python/Cormas communication locally; exploring different ways to exchange data.

### Still have to do:
- `roadmap.md` with clear milestones.
- The webinar presentation showing progress and plans for the whole project