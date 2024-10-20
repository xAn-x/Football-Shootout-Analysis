import supervision as sv  # Make sure sv is the right module
import supervision as sv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import os
import math
import pickle

# ================================ ANNOTATIONS =======================================

color_palette = ['#FFFF00', '#DC143C', '#1AA7EC', '#FFFFFF', ]

# ANNOTATERS
player_annotater = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(color_palette), thickness=2)

ball_annotater = sv.TriangleAnnotator(
    sv.ColorPalette.from_hex(["#FFFF00"]), base=15, height=20)

player_with_ball_annotater = sv.BoxCornerAnnotator(
    color=sv.ColorPalette.from_hex(["#FFB500"]), thickness=2
)

label_annotater = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(color_palette), text_thickness=1, text_color=sv.ColorPalette.from_hex(["#000000"]))


def draw_detections(frame: np.ndarray, detections: sv.Detections, annotater, labels: list = None, *, pad: float = None) -> np.ndarray:
    """
    Annotate a frame with bounding boxes for detections using a given annotater.

    Parameters
    ----------
    frame : np.ndarray
        The frame to annotate.
    detections : sv.Detections
        The detections to annotate.
    annotater : sv.Annotater
        The annotater to use for drawing the bounding boxes.
    labels : list, optional
        The labels to use for the detections. If not given, no labels are drawn.
    pad : float, optional
        If given, the boxes are padded by this amount before drawing.

    Returns
    -------
    np.ndarray
        The annotated frame.
    """
    if detections.is_empty():
        return frame

    if pad is not None:
        detections.xyxy = sv.pad_boxes(detections.xyxy, pad)

    frame = annotater.annotate(frame, detections)
    if labels:
        frame = label_annotater.annotate(frame, detections, labels=labels)
    return frame


def annotate_frame(frame: np.ndarray, detections: sv.Detections, show_result=False):
    """
    Annotates a frame with detections and tracks of players and ball.

    Args:
        frame (np.ndarray): A frame to annotate.
        detections (dict): A dictionary with class names as keys and detections
            corresponding to each class as values.
        show_result (bool, optional): Whether to show the annotated frame. Defaults to False.

    Returns:
        np.ndarray: The annotated frame.
    """
    all_detections_but_ball = [v for k, v in detections.items() if k != 'ball']
    all_detections_but_ball = sv.Detections.merge(all_detections_but_ball)
    all_detections_but_ball = player_tracker.update_with_detections(
        all_detections_but_ball)

    # Ball-Detections
    ball_detection = detections['ball']
    if ball_detection:
        ball_detection = ball_detection[[np.argmax(ball_detection.confidence)]]

    # Player with ball
    player_with_ball = get_player_with_ball(
        detections['player'], ball_detection)

    # Annotate
    # men
    labels = [
        f'{tracker_id}' for tracker_id in all_detections_but_ball.tracker_id]

    annotated_frame = draw_detections(
        frame.copy(), all_detections_but_ball, player_annotater, labels)

    # ball
    annotated_frame = draw_detections(
        annotated_frame, ball_detection, ball_annotater, pad=10)

    # player with ball
    annotated_frame = draw_detections(
        annotated_frame, player_with_ball, player_with_ball_annotater, pad=40)

    if show_result:
        sv.plot_image(annotated_frame, size=(6, 8))

    return annotated_frame


def annotate_video(video_path: str, output_path: str, detections: list[defaultdict[str, sv.Detections]]) -> np.ndarray:
    """
    Annotates a video with detections and tracks of players and ball.

    Args:
        video_path (str): Path to the video to annotate.
        output_path (str): Path to save the annotated video.
        detections (list[dict]): A list of dictionaries with class names as keys and detections
            corresponding to each class as values.

    Returns:
        np.ndarray: The annotated video.
    """

    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path)
    annotated_frame = []
    with sv.VideoSink(output_path, video_info) as sink:
        for frame, frame_detections in tqdm(zip(frame_generator, detections), desc="Annotating Frames", total=video_info.total_frames):
            annotated_frame = annotate_frame(frame, frame_detections)
            sink.write_frame(annotated_frame)

    print(rf"Saved video to `{output_path}`")
    return annotated_frame


# =============================== Processing Utils =======================================
# Trackers
player_tracker = sv.ByteTrack()
player_tracker.reset()


def get_player_with_ball(player_detections: sv.Detections, ball_detection: sv.Detections, min_dist: float = 100) -> sv.Detections:
    """
    Identify the player closest to the ball within a specified distance.

    This function calculates the center points of bounding boxes for player detections
    and the ball detection, then finds the player whose center is closest to the ball's center.
    If the distance is less than the specified minimum distance (`min_dist`), the corresponding
    player detection is returned. Otherwise, an empty detection is returned.

    Args:
        player_detections: sv.Detections
            A set of detections corresponding to players.
        ball_detection: sv.Detections
            A detection corresponding to the ball.
        min_dist: float, optional
            The minimum distance threshold to consider a player as having the ball. Default is 100.

    Returns:
        sv.Detections
            A detection of the player closest to the ball if within `min_dist`, otherwise an empty detection.
    """
    if not player_detections or not ball_detection:
        return sv.Detections.empty()

    def get_bndbx_center(xyxy):
        return ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)

    player_centers = np.apply_along_axis(
        get_bndbx_center, 1, player_detections.xyxy)

    ball_centers = np.apply_along_axis(
        get_bndbx_center, 1, ball_detection.xyxy)

    dist = np.linalg.norm(player_centers-ball_centers, axis=1)
    idx = np.argmin(dist)
    if dist[idx] < min_dist:
        return player_detections[[idx]]
    return sv.Detections.empty()


def filter_detections(detections) -> dict[(str, sv.Detections)]:
    """
    Filters and groups detections by their class names.

    Args:
        detections (list): A list of detections where each detection is expected to be a dictionary with a 'class_name' key.

    Returns:
        defaultdict[(str, sv.Detections)]: A dictionary with class names as keys and detections corresponding to each class as values.
    """
    filtered_detections = defaultdict(list)
    for i, d in enumerate(detections):
        filtered_detections[d[-1]['class_name']].append(i)

    filtered_detections = defaultdict(
        sv.Detections.empty,  {k: detections[v]for k, v in filtered_detections.items()})

    return filtered_detections


def batch_generator(generator, batch_size=16):
    """
    Generates batches of frames from a given generator.

    Args:
        generator (iterable): An iterable that produces frames.
        batch_size (int, optional): The number of frames per batch. Defaults to 16.

    Yields:
        list: A batch of frames with a length of up to `batch_size`.
    """

    batch = []
    for frame in generator:
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    # If some frames are left
    if batch:
        yield batch


def process_frames(model, frames: np.ndarray | list[np.ndarray]) -> list[defaultdict[(str, sv.Detections)]]:
    """
    Runs object detection model on a frame and performs non-maximum suppression
    and filtering of detections.

    Args:
        model (YOLO): A YOLO object detection model.
        frame (np.ndarray): A frame to detect objects in.

    Returns:
        list[defaultdict[(str, sv.Detections)]]: A list of detections for each frame.
    """
    results = model(frames, conf=0.1, verbose=False)
    detections = map(sv.Detections.from_ultralytics, results)
    detections = [detection.with_nms(0.3, False) for detection in detections]

    # filter detections
    detections = map(filter_detections, detections)

    return list(detections)


def process_video(model, video_path: str, batch_size=16, save_stubs: bool = False, stubs_dir: str = "./stubs") -> list[defaultdict[(str, sv.Detections)]]:
    """
    Processes a video by running each frame through a detection model and returns the detections list.

    Args:
        model: The detection model to process each video frame.
        video_path (str): Path to the input source video.
        batch_size (int, optional): Number of frames processed in a batch. Defaults to 16.
        save_stubs (bool, optional): If True, saves the detection results (stubs) to a pickle file. Defaults to True.
        stubs_dir (str, optional): Directory to save the pickle file containing detection results. Defaults to "/stubs".

    Returns:
        list: A list of detection results for each processed video frame.
    """

    if not os.path.exists(stubs_dir):
        os.makedirs(stubs_dir)

    stubs_path = Path(stubs_dir) / f"{Path(video_path).stem}_output.pkl"

    # If stubs already exist and saving is disabled, load and return the detections
    if save_stubs == False and os.path.exists(stubs_path):
        with open(stubs_path, 'rb') as f:
            return pickle.load(f)

    # Get video information and frame generator
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path)
    batch_gen = batch_generator(frame_generator, batch_size=batch_size)
    print(f"Total Frames: {video_info.total_frames}")
    detections = []

    # Process frames in batches
    for batch in tqdm(batch_gen, desc="Processing Frames", total=math.ceil(video_info.total_frames / batch_size)):
        detection = process_frames(model, batch)
        detections.extend(detection)

    if save_stubs:
        print("Saving stubs...")
        with open(stubs_path, 'wb') as f:
            pickle.dump(detections, f)
        print(f"Saved stubs to: {stubs_path}")

    return detections
