from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import sys

import torch
print("torch.backends.mps.is_available(): ", torch.backends.mps.is_available(), "\n")

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the GroundingDINO directory
groundingdino_path = os.path.join(current_dir, '..', 'GroundingDINO')

# Add it to sys.path so we can import from it
sys.path.append(groundingdino_path)

# Now you can construct the config path or import from the module
config_file = os.path.join(groundingdino_path, 'config', 'GroundingDINO_SwinT_OGC.py')
print("Using config file at:", config_file)

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth", device="mps")
IMAGE_PATH = "/Users/kylakaplan/Desktop/experiment-groundingdino/GroundingDINO/img/img0_crop.jpeg"
TEXT_PROMPT = "greencircle"
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device="mps"
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)
print("image saved")