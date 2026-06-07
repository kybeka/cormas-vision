import json
from pathlib import Path
from collections import defaultdict

json_path = Path("data/_annotations.coco.json")       # adjust if needed
out_path  = json_path.with_name(json_path.stem + "_patched.json")

with json_path.open() as f:
    data = json.load(f)

# build a lookup: cat‑id → cat‑name
id2name = {c["id"]: c["name"] for c in data["categories"]}

# group annotation records by image_id
img2cats = defaultdict(list)
for ann in data["annotations"]:
    img2cats[ann["image_id"]].append(id2name[ann["category_id"]])

# patch captions
for img in data["images"]:
    cats = sorted(set(img2cats.get(img["id"], [])))
    img["caption"]  = " [SEP] ".join(cats)
    img["cap_list"] = cats   # still useful for sanity_check.py

with out_path.open("w") as f:
    json.dump(data, f, indent=2)

print(f"patched file written to: {out_path}")