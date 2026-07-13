import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from ultralytics import YOLO

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32


@dataclass
class PlayerTrack:
    frame_number: int
    player_track_id: int
    team: int
    pixel_x: float
    pixel_y: float
    x_meters: float
    y_meters: float
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float

@dataclass
class BallTrack:
    frame_number: int
    pixel_x: float
    pixel_y: float
    x_meters: float
    y_meters: float
    confidence: float


def run_detection_and_tracking(
      frames: List[Tuple[int, np.ndarray]],
      model: YOLO,
      confidence_threshold: float,
      ball_confidence_threshold: float,
      clip_path: str,           # ← new parameter
      sample_fps: int,          # ← new parameter
  ) -> Tuple[List[PlayerTrack], List[BallTrack]]:

    """
    Two-pass approach:
    Pass 1 — Feed the clip VIDEO FILE directly to model.track() for player tracking.
             Ultralytics processes all frames internally in one continuous stream,
             which is the only way persist=True gives stable IDs across 280+ frames.
    Pass 2 — Run model() (plain inference, no tracker) on each sampled frame
             for ball detection at very low confidence. Ball is not tracked
             because its erratic motion breaks ByteTrack.
    """

    all_tracks: List[PlayerTrack] = []
    all_ball_tracks: List[BallTrack] = []

    # ── PASS 1: Player tracking via video file ────────────────────────────────
    # model.track() on a file path processes every frame internally in sequence.
    # This is fundamentally different from calling model.track(frame) in a loop —
    # the tracker state never gets interrupted between frames.
    print("Pass 1: Player tracking on full clip...")
    model.predictor = None  # clean reset before this run only

    # We need to know which source frames map to which sample frames.
    # model.track on a file processes at the VIDEO's native fps (e.g. 25fps).
    # We'll collect all results and then downsample to sample_fps ourselves.
    track_results = model.track(
        source=clip_path,    # pass the file path, not individual frames
        conf=confidence_threshold,
        tracker='/content/bytetrack.yaml',
        persist=True,
        verbose=False,
        classes=[PERSON_CLASS_ID],  # persons only for tracking pass
        stream=True,                # stream=True yields results frame by frame
                                    # without loading entire video into RAM
    )

    # Determine source video fps for downsampling
    cap_check = cv2.VideoCapture(clip_path)
    source_fps = cap_check.get(cv2.CAP_PROP_FPS)
    total_source_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_check.release()

    # We want 1 sample per (source_fps / sample_fps) source frames
    frame_interval = int(round(source_fps / sample_fps))
    print(f"  Source fps: {source_fps:.1f}, sampling every {frame_interval} frames")

    source_frame_idx = 0
    sample_frame_number = 0

    for result in track_results:
        # Only keep frames that correspond to our sample rate
        if source_frame_idx % frame_interval == 0:
            if result.boxes is not None and result.boxes.id is not None:
                tracker_ids = result.boxes.id.int().tolist()
                class_ids = result.boxes.cls.int().tolist()
                confidences = result.boxes.conf.tolist()
                xyxy_all = result.boxes.xyxy.tolist()

                for j in range(len(tracker_ids)):
                    x1, y1, x2, y2 = xyxy_all[j]
                    all_tracks.append(PlayerTrack(
                        frame_number=sample_frame_number,
                        player_track_id=int(tracker_ids[j]),
                        team=-1,
                        pixel_x=float((x1 + x2) / 2),
                        pixel_y=float(y2),
                        x_meters=0.0,
                        y_meters=0.0,
                        confidence=float(confidences[j]),
                        bbox_x1=float(x1),
                        bbox_y1=float(y1),
                        bbox_x2=float(x2),
                        bbox_y2=float(y2),
                    ))
            sample_frame_number += 1

        source_frame_idx += 1
        if source_frame_idx % 500 == 0:
            print(f"  Processed {source_frame_idx}/{total_source_frames} source frames...")

    unique_players = len(set(t.player_track_id for t in all_tracks))
    print(f"  ✅ Player tracking complete: {len(all_tracks)} records, {unique_players} unique IDs")

    # ── PASS 2: Ball detection on sampled frames (plain inference, no tracker) ─
    # We use model() not model.track() here — the ball moves too erratically
    # for ByteTrack to handle, and we only need one position per sample frame.
    # We run at conf=0.01 and filter manually because ball peaks at ~0.057.
    print("\nPass 2: Ball detection on sampled frames...")
    BALL_DETECT_CONF = 0.01  # cast wide net, filter below

    for frame_number, frame in frames:
        results = model(frame, conf=BALL_DETECT_CONF, classes=[BALL_CLASS_ID], verbose=False)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        confs = result.boxes.conf.tolist()
        xyxy = result.boxes.xyxy.tolist()

        # Keep only detections above our actual ball threshold (0.04)
        valid_balls = [(c, xy) for c, xy in zip(confs, xyxy) if c >= 0.04]
        if not valid_balls:
            continue

        # Take highest confidence ball detection this frame
        best_conf, best_xy = max(valid_balls, key=lambda x: x[0])
        x1, y1, x2, y2 = best_xy
        all_ball_tracks.append(BallTrack(
            frame_number=frame_number,
            pixel_x=float((x1 + x2) / 2),
            pixel_y=float((y1 + y2) / 2),
            x_meters=0.0,
            y_meters=0.0,
            confidence=float(best_conf),
        ))

    print(f"  ✅ Ball detected in {len(all_ball_tracks)}/{len(frames)} sampled frames")

    print(f"\n✅ Tracking complete")
    print(f"  Total player track records: {len(all_tracks)}")
    print(f"  Unique player IDs: {unique_players}")
    print(f"  Ball tracks: {len(all_ball_tracks)}")
    return all_tracks, all_ball_tracks

