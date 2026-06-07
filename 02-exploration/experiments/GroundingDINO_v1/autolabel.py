import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'segment_anything'))

# === CONFIG ===
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"  # Download from: https://github.com/facebookresearch/segment-anything
IMAGE_PATH = "img/img1.jpeg"

# === LOAD MODEL ===
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# === LOAD IMAGE ===
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# === GENERATE MASKS ===
masks = mask_generator.generate(image_rgb)

# === HELPER FUNCTIONS ===
def average_color(mask, image):
    # mask: boolean numpy array (H, W)
    # image: RGB image (H, W, 3)
    masked_pixels = image[mask]
    return np.mean(masked_pixels, axis=0)  # [R, G, B]

def classify_color(rgb):
    r, g, b = rgb
    if r > 180 and g < 100 and b < 100:
        return "red"
    elif g > 180 and r < 100 and b < 100:
        return "green"
    elif b > 180 and r < 100 and g < 100:
        return "blue"
    elif r > 180 and g > 180 and b < 100:
        return "yellow"
    else:
        return "unknown"

# === PROCESS MASKS ===
annotated = image_bgr.copy()

for mask_data in masks:
    mask = mask_data["segmentation"]  # numpy array of shape (H, W), bool
    bbox = mask_data["bbox"]  # [x, y, w, h]
    
    avg_rgb = average_color(mask, image_rgb)
    label = classify_color(avg_rgb)
    
    x, y, w, h = map(int, bbox)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

# === SAVE & SHOW ===
cv2.imwrite("annotated_output.jpeg", annotated)
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title("Auto-Labeled Bounding Boxes")
plt.axis("off")
plt.show()
