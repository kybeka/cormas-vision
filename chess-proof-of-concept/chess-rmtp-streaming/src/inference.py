import os, time, cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from datetime import datetime

STREAM_URL   = os.getenv("STREAM_URL", "rtmp://localhost:1935/live/chess")
STREAM = "rtmp://localhost:1935/live/chess"
MODEL_REPO   = os.getenv("MODEL_REPO", "dopaul/chess-piece-detector-merged")
MODEL_FILE   = os.getenv("MODEL_FILE", "best.pt")
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", 10))

# to create some subfolders in the output folder
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_dir = os.path.join(OUTPUT_DIR, timestamp)
os.makedirs(session_dir, exist_ok=True)

print("🔄 downloading weights …")
# weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

weights_path = hf_hub_download(
    repo_id="dopaul/chess-board-segmentation",
    filename="best.pt",
    local_dir="../models"
)

model = YOLO(weights_path)
print("✅ model ready:", MODEL_REPO, MODEL_FILE)

cap = cv2.VideoCapture(STREAM_URL)
# cap = cv2.VideoCapture(STREAM)
i = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.5); continue

    if i % FRAME_STRIDE == 0:
        res = model(frame)
        # cv2.imshow("detections", res[0].plot())
        # cv2.imwrite(f"/app/output/frame_{i}.jpg", res[0].plot())
        cv2.imwrite(f"{session_dir}/frame_{i:03}.jpg", res[0].plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    i += 1

cap.release()
cv2.destroyAllWindows()