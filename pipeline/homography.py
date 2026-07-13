import cv2
import numpy as np
from typing import List, Tuple, Optional
from pipeline.tracker import PlayerTrack, BallTrack

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0



def pixel_to_pitch(
    pixel_x: float,
    pixel_y: float,
    H: np.ndarray
) -> Tuple[float, float]:
    """
    Apply the homography matrix H to convert a single pixel coordinate
    to a real-world pitch coordinate in meters.

    Homography uses homogeneous coordinates (3D vectors for 2D points).
    The math: [x', y', w'] = H × [x, y, 1]
    Then divide by w' to get the Euclidean coordinates: (x'/w', y'/w')
    """
    # Create homogeneous coordinate: add a 1 as the third component
    pixel_point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float32)

    # Apply the homography matrix (matrix multiplication)
    transformed = H @ pixel_point
    # H is 3×3, pixel_point is 3×1, result is 3×1

    # Divide by the third component (w) to get Euclidean coordinates
    pitch_x = transformed[0] / transformed[2]
    pitch_y = transformed[1] / transformed[2]

    # Clamp to valid pitch boundaries
    # Points slightly outside the pitch (e.g. player near touchline) can
    # project to slightly negative or >105/>68 values — clamp them.
    pitch_x = float(np.clip(pitch_x, 0.0, PITCH_LENGTH_M))
    pitch_y = float(np.clip(pitch_y, 0.0, PITCH_WIDTH_M))

    return pitch_x, pitch_y


def apply_homography_to_tracks(
    tracks: List[PlayerTrack],
    frames: List[Tuple[int, np.ndarray]],
    H: np.ndarray
) -> List[PlayerTrack]:
    """
    Apply the homography matrix to every track's pixel position,
    filling in the x_meters and y_meters fields.

    Also recalculates H every HOMOGRAPHY_RECALC_INTERVAL frames
    (currently manual — in Phase 3 this will be automated).
    """
    print("Applying homography to all tracks...")

    for track in tracks:
        track.x_meters, track.y_meters = pixel_to_pitch(
            track.pixel_x, track.pixel_y, H
        )

    # Validate: check that projected coordinates look reasonable
    # Players should mostly be within the pitch boundaries
    valid_tracks = [
        t for t in tracks
        if 0 <= t.x_meters <= PITCH_LENGTH_M and 0 <= t.y_meters <= PITCH_WIDTH_M
    ]
    print(f"  ✅ Homography applied: {len(valid_tracks)}/{len(tracks)} tracks within pitch bounds")
    return tracks

def apply_homography_to_ball_tracks(
      ball_tracks: List[BallTrack],
      H: np.ndarray
  ) -> List[BallTrack]:
      """Apply homography to ball track pixel positions."""
      for track in ball_tracks:
          track.x_meters, track.y_meters = pixel_to_pitch(
              track.pixel_x, track.pixel_y, H
          )
      return ball_tracks


# Standard broadcast camera homography reference points.
  # These pixel coordinates are approximate averages for typical Premier League /
  # Champions League broadcast footage at 1920x1080 resolution.
  # They map to known FIFA pitch coordinates (meters).
  #
  # IMPORTANT: These are starting defaults. If your video has unusual framing,
  # you may need to adjust the SRC_POINTS values. The DST_POINTS (real-world
  # meters) never change — only SRC_POINTS (pixel positions) change per camera.
  #
  # How to find SRC_POINTS for a new camera angle:
  # 1. Take a screenshot of frame 0 from your video
  # 2. Open in any image editor (Preview on Mac works)
  # 3. Hover over each keypoint listed in DST_POINTS and note the pixel X,Y
  # 4. Update SRC_POINTS accordingly

DEFAULT_SRC_POINTS = np.float32([
    [190,  65],   # top-left corner of pitch
    [1750, 65],   # top-right corner of pitch
    [190,  980],  # bottom-left corner of pitch
    [1750, 980],  # bottom-right corner of pitch
    [960,  65],   # halfway line top touchline
    [960,  980],  # halfway line bottom touchline
    [960,  520],  # center spot
    [380,  280],  # left penalty area top-left
    [380,  760],  # left penalty area bottom-left
])

DEFAULT_DST_POINTS = np.float32([
    [0.0,   0.0],   # top-left corner
    [105.0, 0.0],   # top-right corner
    [0.0,   68.0],  # bottom-left corner
    [105.0, 68.0],  # bottom-right corner
    [52.5,  0.0],   # halfway line top
    [52.5,  68.0],  # halfway line bottom
    [52.5,  34.0],  # center spot
    [16.5,  13.84], # left penalty area top
    [16.5,  54.16], # left penalty area bottom
])

def compute_default_homography() -> np.ndarray:
    """
    Compute the homography matrix from the default broadcast reference points.
    Called once when the FastAPI server starts, result is cached in memory.
    """
    H, mask = cv2.findHomography(
        DEFAULT_SRC_POINTS,
        DEFAULT_DST_POINTS,
        cv2.RANSAC,
        5.0
    )
    if H is None:
        raise RuntimeError("Failed to compute default homography matrix")
    inliers = int(mask.sum())
    print(f"  Default homography computed: {inliers}/{len(DEFAULT_SRC_POINTS)} inliers")
    return H

