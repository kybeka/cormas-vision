## Chess Proof of Concept - RTMP Streaming & YOLO Detection

### Overview
Real-time chess piece detection system using YOLO models with RTMP video streaming capabilities.

### Architecture
- **RTMP Server**: Nginx-based streaming server for video input  
- **YOLO Inference**: Real-time chess piece detection using Ultralytics YOLO  
- **Model Source**: Pre-trained chess detection model from Hugging Face  
- **Output**: Timestamped detection results saved as images  

### Components
- `src/inference.py`: Main detection pipeline  
- `docker/Dockerfile.yolo`: Inference container setup  
- `docker-compose.yml`: Complete system orchestration  

### Configuration
- **Stream URL**: `rtmp://localhost:1935/live/chess`  
- **Frame stride**: Configurable processing interval  
- **Output directory**: Timestamped results in `output/`  

### To run:

```bash
git clone https://github.com/kybeka/cormas-vision.git
cd chess-proof-of-concept/chess-rmtp-streaming
cp .env.example .env            # edit if you like
docker compose up --build
```

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running in the background
- [Larix Broadcaster](https://apps.apple.com/app/larix-broadcaster/id1042474385) installed on your phone (iOS or Android)

### Instructions:

On your phone, open Larix Broadcaster and add a connection:
```bash
rtmp://<your-machine-ip>/live/chess
```

Press Record in Larix and the video stream will be picked up automatically by the running inference container, and a window will pop up showing live chess piece detections.

Press Q in the detection window to stop.