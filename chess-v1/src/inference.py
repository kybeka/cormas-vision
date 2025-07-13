import os, time, cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# STREAM_URL   = os.getenv("STREAM_URL", "rtmp://localhost/live/chess")
STREAM = "rtmp://localhost:1935/live/chess"
MODEL_REPO   = os.getenv("MODEL_REPO", "dopaul/chess-piece-detector-merged")
MODEL_FILE   = os.getenv("MODEL_FILE", "best.pt")
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", 10))

print("🔄 downloading weights …")
# weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

weights_path = hf_hub_download(
    repo_id="dopaul/chess-piece-detector-merged",
    filename="best.pt",
    local_dir="../models",
    local_dir_use_symlinks=False
)

model = YOLO(weights_path)
print("✅ model ready:", MODEL_REPO, MODEL_FILE)

# cap = cv2.VideoCapture(STREAM_URL)
cap = cv2.VideoCapture(STREAM)
i = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.5); continue

    if i % FRAME_STRIDE == 0:
        res = model(frame)
        cv2.imshow("detections", res[0].plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    i += 1

cap.release()
cv2.destroyAllWindows()