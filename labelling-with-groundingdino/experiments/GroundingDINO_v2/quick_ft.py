"""
quick_ft.py  -  few-shot fine-tune of Grounding DINO on a CSV.

CSV columns:
label_name,x1,y1,x2,y2,image_name,width,height
"""

import os, csv, cv2, torch
from torch.utils.data import Dataset, DataLoader
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.get_tokenlizer import get_tokenlizer          # NEW
from transformers import BertTokenizerFast                            # NEW

CSV_PATH   = "data/train_7frames_gdino.csv"
WEIGHTS    = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
OUT_CKPT   = "board_ft.pth"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS     = 5
LR         = 1e-5
BATCH_SIZE = 2
MAX_TOKENS = 256

tokenizer  = BertTokenizerFast.from_pretrained("bert-base-uncased")   # NEW

class CsvDataset(Dataset):
    def __init__(self, csv_path):
        rows = list(csv.DictReader(open(csv_path)))
        self.by_img = {}
        for r in rows:
            self.by_img.setdefault(r["image_name"], []).append(r)
        self.img_paths = sorted(self.by_img)

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        csv_path = self.img_paths[idx]
        fs_path  = csv_path if os.path.isfile(csv_path) else \
                   os.path.join("data", os.path.basename(csv_path))

        img = cv2.imread(fs_path)[:, :, ::-1] / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        boxes, phrases = [], []
        for r in self.by_img[csv_path]:
            boxes.append([float(r[k]) for k in ("x1","y1","x2","y2")])
            phrases.append(r["label_name"])

        # -------- positive_map construction --------
        caption = " . ".join(phrases)
        enc = tokenizer(caption, add_special_tokens=True,
                        max_length=MAX_TOKENS, truncation=True,
                        return_offsets_mapping=True)
        seq_len = len(enc["input_ids"])
        pos_map = torch.zeros((len(phrases), MAX_TOKENS), dtype=torch.float32)

        for box_idx, phrase in enumerate(phrases):
            phrase_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
            # find matching span
            for i in range(seq_len - len(phrase_ids) + 1):
                if enc["input_ids"][i:i+len(phrase_ids)] == phrase_ids:
                    pos_map[box_idx, i:i+len(phrase_ids)] = 1
                    break  # assume first match

        tgt = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "phrases": phrases,
            "caption": caption,
            "positive_map": pos_map
        }
        return img, tgt

# ----------------------------------------------------------------------
print("🔹 loading dataset")
dl = DataLoader(CsvDataset(CSV_PATH), batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=lambda x: tuple(zip(*x)))

print("🔹 building model")
cfg_path = os.path.join(os.path.dirname(
    __import__("groundingdino").__file__),
    "config", "GroundingDINO_SwinT_OGC.py")
cfg = SLConfig.fromfile(cfg_path); cfg.device = DEVICE
model = build_model(cfg)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=False)
model.to(DEVICE).train()

# freeze backbone to save VRAM
for p in model.backbone.parameters(): p.requires_grad_(False)

opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

print(f"🔹 starting training on {DEVICE} for {EPOCHS} epochs…")
for epoch in range(EPOCHS):
    running = 0
    for imgs, tgts in dl:
        imgs = nested_tensor_from_tensor_list([i.to(DEVICE) for i in imgs])
        loss_dict = model(imgs, tgts)
        loss = sum(loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
    print(f"epoch {epoch+1}/{EPOCHS}   mean_loss={running/len(dl):.4f}")

torch.save(model.state_dict(), OUT_CKPT)
print(f"✅ finished. Fine-tuned weights → {OUT_CKPT}")
