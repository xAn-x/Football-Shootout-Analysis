import supervision as sv  # Make sure sv is the right module
import supervision as sv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import os
import math
import pickle
from TeamAssigner import TeamAssigner

# ================================= Team Assigner ===================================
team_assigner = TeamAssigner()


# ================================ ANNOTATIONS =======================================

color_palette = ["#000000", '#FFFF00', "#000000",
                 '#FFFFFF', "#000000", '#DC143C', '#4169e1', "#000000"]

# ANNOTATERS
ecclipse_annotater = sv.EllipseAnnotator(
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
    # Mens
    annotated_frame = frame.copy()
    for key, det in detections.items():
        if key == 'ball' or key == 'player_with_ball':
            continue

        annotated_frame = draw_detections(
            annotated_frame, det, ecclipse_annotater, [f"{ID}" for ID in det.tracker_id])

    # ball
    annotated_frame = draw_detections(
        annotated_frame, detections['ball'], ball_annotater, pad=10)

    # player with ball
    annotated_frame = draw_detections(
        annotated_frame, detections['player_with_ball'], player_with_ball_annotater, pad=40)

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
trackers = {obj: sv.ByteTrack() for obj in ['player', 'goalkeeper', 'referee']}


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

    # Track players,referee and goalkeeper
    for key in filtered_detections.keys():
        if key == "ball":
            filtered_detections[key] = filtered_detections[key][[
                np.argmax(filtered_detections[key].confidence)]]
        else:
            filtered_detections[key] = trackers[key].update_with_detections(
                filtered_detections[key])

    # Filter out player with ball
    filtered_detections["player_with_ball"] = get_player_with_ball(
        filtered_detections["player"], filtered_detections["ball"])

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


def process_frames(model, frames: np.ndarray | list[np.ndarray], verbose=False) -> list[defaultdict[(str, sv.Detections)]]:
    """
    Runs object detection model on a frame and performs non-maximum suppression
    and filtering of detections.

    Args:
        model (YOLO): A YOLO object detection model.
        frame (np.ndarray): A frame to detect objects in.

    Returns:
        list[defaultdict[(str, sv.Detections)]]: A list of detections for each frame.
    """

    if isinstance(frames, np.ndarray):
        frames = [frames]

    results = model(frames, conf=0.1, verbose=verbose)
    detections = [sv.Detections.from_ultralytics(result) for result in results]
    detections = [detection.with_nms(0.3, False) for detection in detections]

    # filter detections
    detections = [filter_detections(d) for d in detections]

    return detections


def process_video(model, video_path: str, batch_size=16, save_stubs: bool = False, stubs_dir: str = "./stubs", verbose=False) -> list[defaultdict[(str, sv.Detections)]]:
    """
    Processes a video by running each frame through a detection model and returns the detections list.

    Args:
        model: The detection model to process each video frame.
        video_path (str): Path to the input source video.
        batch_size (int, optional): Number of frames processed in a batch. Defaults to 16.
        save_stubs (bool, optional): If True, saves the detection results (stubs) to a pickle file. Defaults to True.
        stubs_dir (str, optional): Directory to save the pickle file containing detection results. Defaults to "/stubs".

    Returns:
        list[defaultdict[(str, sv.Detections)]]: A array of detection results for each processed video frame.
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

    # Process frames
    for frames in tqdm(batch_gen, desc="Detecting entities...", total=math.ceil(video_info.total_frames / batch_size)):
        frame_detections = process_frames(model, frames, verbose=verbose)
        detections.extend(frame_detections)

    if save_stubs:
        print("Saving stubs...")
        with open(stubs_path, 'wb') as f:
            pickle.dump(detections, f)
        print(f"Saved stubs to: {stubs_path}")

    return detections


def update_detections_with_teamid_in_a_frame(frame: np.ndarray, detections: defaultdict[(str, sv.Detections)], verbose=False) -> defaultdict[(str, sv.Detections)]:
    team_ids = team_assigner.get_team_ids(
        frame, detections['player'], verbose=verbose)
    team_ids += 5
    detections['player'].class_id = team_ids
    return detections


def update_detections_with_teamid_in_video(video_path: str, detections: np.ndarray[defaultdict[(str, sv.Detections)]], save_stubs=False, stubs_dir="./stubs/teams", verbose=False) -> np.ndarray[defaultdict[(str, sv.Detections)]]:
    if not os.path.exists(stubs_dir):
        os.makedirs(stubs_dir)

    stubs_path = Path(stubs_dir) / f"{Path(video_path).stem}_output.pkl"

    if save_stubs == False and os.path.exists(stubs_path):
        with open(stubs_path, 'rb') as f:
            return pickle.load(f)

    frame_generator = sv.get_video_frames_generator(video_path)
    n = len(detections)
    # assign team
    already_assigned_tracker_ids = set()
    tracker_id_to_team_id = {}

    for i in tqdm(range(n), total=n, desc="Analyzing Crops..."):
        frame, frame_detections = next(frame_generator), detections[i]

        tracker_ids = set(frame_detections['player'].tracker_id)

        if len(already_assigned_tracker_ids) == 0 or tracker_ids.issubset(already_assigned_tracker_ids) == False:
            track_ids = frame_detections['player'].tracker_id
            not_assigned_idx = [i for i in range(
                len(track_ids)) if track_ids[i] not in already_assigned_tracker_ids]

            already_assigned_tracker_ids.update(track_ids[not_assigned_idx])

            # Filter out detections that are not assigned to any teams
            not_assigned_detections = frame_detections['player'][not_assigned_idx]

            team_ids = team_assigner.get_team_ids(
                frame, not_assigned_detections, verbose=verbose)
            team_ids += 5
            tracker_id_to_team_id.update({
                tracker_id: team_id
                for tracker_id, team_id in zip(not_assigned_detections.tracker_id, team_ids)
            })

    for i in tqdm(range(n), total=n, desc="Assigning teams..."):
        detections[i]['player'].class_id = np.array(
            [tracker_id_to_team_id[tracker]
                for tracker in detections[i]['player'].tracker_id]
        )

    if save_stubs:
        print("Saving stubs...")
        with open(stubs_path, 'wb') as f:
            pickle.dump(detections, f)
        print(f"Saved stubs to: {stubs_path}")

    return detections
