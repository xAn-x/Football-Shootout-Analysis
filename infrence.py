from ultralytics import YOLO
from pathlib import Path
from utils import process_frame, process_video
import cv2

MODEL_PATH = Path("weights/last.pt")
IMG_SIZE = 640
IMG_PATH = Path("test-data/image.png")
VIDEO_PATH = Path("test-data/video.mp4")

model = YOLO(MODEL_PATH, "detection")

img = cv2.imread(str(IMG_PATH))
annotated_frame = process_frame(model, img)

cv2.imwrite("output.png", annotated_frame)

# process_video(model, VIDEO_PATH, "output2.mp4")
