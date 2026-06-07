import json, csv, os


COCO_JSON = "train/_annotations.coco.json"  
IMG_DIR   = "img"                           # images are in current dir
OUT_CSV   = "train_7frames_gdino.csv"         

with open(COCO_JSON) as f:
    coco = json.load(f)

# building lookup tables
id2file = {img["id"]: img["file_name"] for img in coco["images"]}
id2size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
id2name = {cat["id"]: cat["name"] for cat in coco["categories"]}

# write to a CSV
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label_name", "x1", "y1", "x2", "y2", "image_name", "width", "height"])
    
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        image_id = ann["image_id"]
        writer.writerow([
            id2name[ann["category_id"]],
            int(x), int(y), int(x + w), int(y + h),
            os.path.join(IMG_DIR, id2file[image_id]),
            *id2size[image_id]
        ])

print(f"done: wrote to {OUT_CSV}")
