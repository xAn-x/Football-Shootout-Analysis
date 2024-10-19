import supervision as sv
import numpy as np

color_palette = ['#FFFF00', '#FA8072', '#87CEEB', '#FFFFFF', ]

# ANNOTATERS
player_annotater = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(color_palette), thickness=2)

ball_annotater = sv.TriangleAnnotator(
    sv.ColorPalette.from_hex(["#FFFF00"]), base=15, height=20)

player_with_ball_annotater = sv.TriangleAnnotator(
    color=sv.ColorPalette.from_hex(["#3333FF"]), base=20, height=20
)

label_annotated = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(color_palette), text_thickness=1, text_color=sv.ColorPalette.from_hex(["#000000"]))


# Trackers
player_tracker = sv.ByteTrack()


def get_player_with_ball(player_detections, ball_detection, min_dist=100):
    if not player_detections or not ball_detection:
        return sv.Detections.empty()

    def get_bndbx_center(xyxy): return ((xyxy[0] + xyxy[2]) / 2,
                                        (xyxy[1] + xyxy[3]) / 2)

    player_centers = np.apply_along_axis(
        get_bndbx_center, 1, player_detections.xyxy)
    ball_centers =  np.apply_along_axis(
        get_bndbx_center, 1, ball_detection.xyxy)

    dist = np.linalg.norm(player_centers-ball_centers,axis=1)
    idx = np.argmin(dist)
    if dist[idx] < min_dist:
        return player_detections[[idx]]
    return sv.Detections.empty()


def process_frame(model, frame):
    results = model(frame, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Player-Tracking and detection
    all_detections_but_ball = detections[detections.class_id != 0]
    all_detections_but_ball.class_id -= 1  # since ball is not being used
    all_detections_but_ball = player_tracker.update_with_detections(
        all_detections_but_ball)

    # Ball Detections
    ball_detection = detections[(detections.class_id == 0)]
    if ball_detection:
        ball_detection=ball_detection[[np.argmax(ball_detection.confidence)]]
        ball_detection.xyxy = sv.pad_boxes(ball_detection.xyxy, 10)


    # Annotate
    # men
    labels = [str(tracker_id)
              for tracker_id in all_detections_but_ball.tracker_id]

    annotated_frame = player_annotater.annotate(
        frame.copy(), all_detections_but_ball)

    # ball
    annotated_frame = ball_annotater.annotate(annotated_frame, ball_detection)

    # player with ball
    only_players_detections = all_detections_but_ball[
        all_detections_but_ball.data['class_name'] == 'player']

    player_with_ball = get_player_with_ball(
        only_players_detections, ball_detection)
    player_with_ball.xyxy = sv.pad_boxes(player_with_ball.xyxy, 40)

    annotated_frame = player_with_ball_annotater.annotate(
        annotated_frame, player_with_ball)

    # labels
    annotated_frame = label_annotated.annotate(
        annotated_frame, all_detections_but_ball, labels=labels)

    return annotated_frame


def process_video(model, source_video_path, target_video_path):
    sv.process_video(source_video_path, target_video_path,
                     callback=lambda frame, idx: process_frame(model, frame))
