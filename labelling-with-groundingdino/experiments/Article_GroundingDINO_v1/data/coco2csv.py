# coco2csv.py
import json, csv, argparse

def main(json_path, out, images_root):
    with open(json_path) as f:
        coco = json.load(f)

    id2img = {img["id"]: img for img in coco["images"]}
    id2cat = {cat["id"]: cat["name"] for cat in coco["categories"]}

    with open(out, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["label_name","bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                    "image_name","image_width","image_height"])
        for ann in coco["annotations"]:
            x, y, w_, h_ = ann["bbox"]
            w.writerow([
                id2cat[ann["category_id"]],
                x, y, x + w_, y + h_,
                id2img[ann["image_id"]]["file_name"],
                id2img[ann["image_id"]]["width"],
                id2img[ann["image_id"]]["height"],
            ])
    print(f"[✓] wrote {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", required=True, help="COCO annotations file")
    p.add_argument("--out", default="annotations_all.csv")
    p.add_argument("--images_root", default="images")
    main(**vars(p.parse_args()))