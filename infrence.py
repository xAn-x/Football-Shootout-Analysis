from ultralytics import YOLO
from pathlib import Path
from utils import process_frames, process_video, annotate_frame, annotate_video, update_detections_with_teamid_in_a_frame, update_detections_with_teamid_in_video
import cv2

MODEL_PATH = Path("weights/last.pt")
IMG_PATH = Path("test/image3.png")
VIDEO_PATH = Path("test/video.mp4")

model = YOLO(MODEL_PATH, "detection")

# img = cv2.imread(str(IMG_PATH))
# detections = process_frames(model, img,verbose=False)[0]

# detections=update_detections_with_teamid_in_a_frame(img,detections,verbose=True)

# annotated_frame = annotate_frame(img, detections, show_result=True)

# cv2.imwrite(rf"./output/{IMG_PATH.stem}_output.png", annotated_frame)


detections = process_video(model, str(VIDEO_PATH),
                           16,save_stubs=False,verbose=False)


detections = update_detections_with_teamid_in_video(
    VIDEO_PATH, detections, save_stubs=False, verbose=False)


annotated_frames = annotate_video(
    str(VIDEO_PATH), r"./output/output.mp4", detections)


print("Done!")
