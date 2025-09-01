# Chess USB Inference

Real-time chess piece detection using USB cameras with YOLO models.

## Features

- **USB Camera Detection**: Auto-detect available USB cameras
- **Real-time Inference**: Live chess piece detection with configurable confidence
- **Live Display**: Real-time visualization with FPS counter
- **Detection Saving**: Optional saving of frames with detections
- **Flexible Configuration**: Command-line arguments for all parameters

## Quick Start

```bash
# Install dependencies
pip install -r src/requirements.txt

# List available cameras
python src/usb_inference.py --list-cameras

# Run with default camera (ID 0)
python src/usb_inference.py

# Run with specific camera and confidence
python src/usb_inference.py --camera 1 --confidence 0.7

# Run without live display (headless)
python src/usb_inference.py --no-display
```

## Command Line Options

- `--camera ID`: Camera ID to use (default: 0)
- `--confidence FLOAT`: Detection confidence threshold (default: 0.5)
- `--no-display`: Disable live video display
- `--no-save`: Disable saving detection images
- `--max-frames INT`: Limit number of frames to process
- `--list-cameras`: Show available camera IDs

## Output

- **Live Display**: Real-time annotated video feed with bounding boxes
- **Saved Images**: Timestamped directory with detection frames
- **Performance**: FPS counter for monitoring performance

## Camera Setup

1. Connect USB camera to your system
2. Run `--list-cameras` to identify camera ID
3. Use the appropriate `--camera ID` parameter

## Performance Notes

- Default resolution: 640x480 at 30 FPS
- Performance varies with camera quality and system specs
- Use `--no-display` for better performance in headless environments