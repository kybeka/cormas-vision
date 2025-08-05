import json, pathlib, collections

src = pathlib.Path("data/_annotations.coco.json")
dst = src.with_name("_annotations_with_caption.coco.json")

with src.open() as f:
    coco = json.load(f)

# map image_id -> list of category names in that picture
id2cats = collections.defaultdict(list)
cats     = {c["id"]: c["name"] for c in coco["categories"]}
for ann in coco["annotations"]:
    id2cats[ann["image_id"]].append(cats[ann["category_id"]])

for img in coco["images"]:
    # simple caption: space‑separated category names
    # (you can swap in anything more descriptive)
    img["caption"] = " ".join(sorted(set(id2cats[img["id"]])))

with dst.open("w") as f:
    json.dump(coco, f)
print(f"wrote {dst}")