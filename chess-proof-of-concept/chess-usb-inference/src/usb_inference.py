import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import argparse

class USBChessInference:
    def __init__(self, camera_id=0, model_repo="yamero999/chess-piece-detection-yolo11n", 
                 model_filename="best.pt", confidence=0.5, save_detections=True):
        self.camera_id = camera_id
        self.confidence = confidence
        self.save_detections = save_detections
        
        # Create output directory
        self.output_dir = f"../output/usb_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.save_detections:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Download and load model
        print(f"Downloading model from {model_repo}...")
        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {camera_id} initialized successfully!")
    
    def detect_available_cameras(self):
        """Detect available USB cameras"""
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def run_inference(self, display_live=True, max_frames=None):
        """Run real-time inference on USB camera feed"""
        frame_count = 0
        fps_counter = 0
        start_time = time.time()
        
        print("Starting USB inference... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Run YOLO inference
                results = self.model(frame, conf=self.confidence)
                
                # Draw detections on frame
                annotated_frame = results[0].plot()
                
                # Save detection if enabled
                if self.save_detections and len(results[0].boxes) > 0:
                    filename = f"detection_{frame_count:06d}.jpg"
                    cv2.imwrite(os.path.join(self.output_dir, filename), annotated_frame)
                
                # Display live feed
                if display_live:
                    # Add FPS counter
                    fps = fps_counter / (time.time() - start_time) if time.time() - start_time > 0 else 0
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Chess USB Inference', annotated_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                fps_counter += 1
                
                # Reset FPS counter every second
                if time.time() - start_time >= 1.0:
                    fps_counter = 0
                    start_time = time.time()
                
                # Check max frames limit
                if max_frames and frame_count >= max_frames:
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
            print(f"Processed {frame_count} frames")
            if self.save_detections:
                print(f"Detections saved to: {self.output_dir}")
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='USB Chess Piece Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Disable live display')
    parser.add_argument('--no-save', action='store_true', help='Disable saving detections')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--list-cameras', action='store_true', help='List available cameras')
    
    args = parser.parse_args()
    
    # List available cameras if requested
    if args.list_cameras:
        inference = USBChessInference()
        cameras = inference.detect_available_cameras()
        print(f"Available cameras: {cameras}")
        return
    
    # Initialize inference
    try:
        inference = USBChessInference(
            camera_id=args.camera,
            confidence=args.confidence,
            save_detections=not args.no_save
        )
        
        # Run inference
        inference.run_inference(
            display_live=not args.no_display,
            max_frames=args.max_frames
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()