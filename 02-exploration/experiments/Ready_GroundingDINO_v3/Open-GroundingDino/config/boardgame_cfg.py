from config.cfg_coco import *        # or  cfg_odvg  (either is fine)

# Override ONLY the bits that differ
# train_dataset = dict(
#     type="coco_grounding",
#     ann_file="../data/_annotations.coco.json",   # your JSON
#     img_prefix="../data",                        # flat image folder
# )
# val_dataset = train_dataset

num_classes = 10
class_names = [
    "board", "inner-board", "green-token", "yellow-token", "orange-token",
    "blue-pawn", "red-pawn", "white-pawn", "yellow-pawn", "objects"
]

max_epoch      = 8     # tiny set
batch_size     = 2
train_backbone = False # freeze Swin (saves VRAM)
use_coco_eval  = False # skip mAP on 7 images
label_list = class_names