# ETH evaluation sample

26 dense gameplay frames from the main ETH session (`2025-11-21_08-53-26`), used to measure how the model actually performed — especially the **token vs pawn** gap.

## Layout
- `images/` — the 26 frames (git-ignored; raw data).
- `labels/` — YOLO-OBB labels. **Currently model pre-labels, not ground truth.** Built with `../make_eval_sample.py`.

## The job: correct the labels
The pre-labels are right about the flat tokens but miss almost every standing **pawn**. Correcting them mostly means **adding the missing pawns** (and fixing the odd wrong token). Two ways:

**A) Your inspector** (already wired):
```bash
conda activate ml
cd ../../03-training/pseudo_labeling
python simple_inspector.py -i 90      # iter_90 is symlinked to this sample
# open http://localhost:5000, correct, save
```

**B) Any YOLO-OBB tool** (Roboflow, LabelImg-OBB): edit `labels/*.txt` directly. Format per line: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (normalised 0–1). Class ids are in `../evaluate.py`.

## Then: get the numbers
```bash
conda activate ml
cd ..
python evaluate.py     # prints per-class precision / recall / mAP
```
