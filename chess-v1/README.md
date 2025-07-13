## To run:

```bash
git clone https://github.com/kybeka/cormas-vision.git
cd chess-v1
cp .env.example .env            # edit if you like
docker compose up --build
```

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running in the background
- [Larix Broadcaster](https://apps.apple.com/app/larix-broadcaster/id1042474385) installed on your phone (iOS or Android)

## Instructions:

On your phone, open Larix Broadcaster and add a connection:
```bash
rtmp://<your-machine-ip>/live/chess
```

Press Record in Larix and the video stream will be picked up automatically by the running inference container, and a window will pop up showing live chess piece detections.

Press Q in the detection window to stop.