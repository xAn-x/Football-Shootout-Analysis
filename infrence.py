from ultralytics import YOLO
from pathlib import Path
from utils import process_video, process_frames, annotate_frame, annotate_video
import cv2

MODEL_PATH = Path("weights/last.pt")
IMG_PATH = Path("test/image1.png")
VIDEO_PATH = Path("test/video.mp4")

model = YOLO(MODEL_PATH, "detection")

img = cv2.imread(str(IMG_PATH))
detections = process_frames(model, img)[0]

annotated_frame = annotate_frame(img, detections, show_result=False)

cv2.imwrite(rf"./output/{IMG_PATH.stem}_output.png", annotated_frame)


# detections = process_video(model, str(VIDEO_PATH), 16)

# annotated_frames = annotate_video(
#     str(VIDEO_PATH), r"./output/output.mp4", detections)


print("Done!")
