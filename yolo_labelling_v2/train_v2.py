from ultralytics import YOLO
import torch
import shutil
import random
from pathlib import Path
import os

# Change to the script's directory
os.chdir(Path(__file__).parent)

# Set up paths
data_root = Path("frames.v3i.yolov11")
train_img = data_root / "train/images"
train_lbl = data_root / "train/labels"
val_img = data_root / "valid/images"
val_lbl = data_root / "valid/labels"

# Create validation directories
val_img.mkdir(parents=True, exist_ok=True)
val_lbl.mkdir(parents=True, exist_ok=True)

# Get all training images
all_images = list(train_img.glob("*.jpg"))
print(f"Total images: {len(all_images)}")

# # Split: 80% train, 20% validation
# random.seed(42)  # For reproducible splits
# random.shuffle(all_images)
# val_count = int(len(all_images) * 0.2)
# val_images = all_images[:val_count]

# print(f"Moving {len(val_images)} images to validation set...")

# # Move validation images and labels
# for img_path in val_images:
#     # Move image
#     shutil.move(str(img_path), str(val_img / img_path.name))
    
#     # Find corresponding label with Roboflow naming convention
#     # Image: frame_X_jpg.rf.{hash}.jpg -> Label: frame_X_jpg.rf.{hash}.txt
#     # Just replace .jpg with .txt
#     label_name = img_path.name.replace('.jpg', '.txt')
#     label_path = train_lbl / label_name
    
#     if label_path.exists():
#         shutil.move(str(label_path), str(val_lbl / label_name))
#     else:
#         print(f"Warning: Label not found for {img_path.name} (expected: {label_name})")

print(f"Train images: {len(list(train_img.glob('*.jpg')))}")
print(f"Val images: {len(list(val_img.glob('*.jpg')))}")
print(f"Train labels: {len(list(train_lbl.glob('*.txt')))}")
print(f"Val labels: {len(list(val_lbl.glob('*.txt')))}")

# Initialize YOLO model
model = YOLO("yolo11m-obb.pt")  # Model file now in same directory

# Train the model with optimized parameters for small dataset (96 images)
results = model.train(
    data="frames.v3i.yolov11/data.yaml",
    epochs=30,  # Reduced from 100 - more appropriate for small dataset
    imgsz=640,
    batch=8,   # Reduced batch size for better gradient updates with small dataset
    lr0=0.01, # Already updated learning rate
    patience=10, # Reduced patience for earlier stopping
    save=True,
    plots=True,
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'),
    # Additional parameters for small dataset optimization
    dropout=0.1,  # Dropout for regularization
    # augment=True  # Data augmentation to increase effective dataset size
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")