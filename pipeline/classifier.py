import cv2
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pipeline.tracker import PlayerTrack

NUM_TEAMS = 4  # as a module-level constant.

def extract_jersey_color(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Extract the dominant jersey color from a player's bounding box.

    We crop the top 40% of the bounding box (torso region) to avoid
    including pitch grass in the color sample (the grass is green and
    would contaminate the jersey color with a third cluster).

    Returns a 3-element array [mean_hue, mean_saturation, mean_value]
    representing the average HSV color of the torso region.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Clamp coordinates to frame boundaries to avoid out-of-bounds crops
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0])  # Invalid crop

    # Crop bounding box
    player_crop = frame[y1:y2, x1:x2]

    # Take only the top 40% (torso) — skip legs and feet
    torso_height = int(player_crop.shape[0] * 0.4)
    torso_crop = player_crop[:torso_height, :]

    if torso_crop.size == 0:
        return np.array([0.0, 0.0, 0.0])

    # Convert BGR (OpenCV default) to HSV
    # Why HSV? Because H (Hue) encodes the actual color as a single value (0-180 in OpenCV),
    # independent of brightness. A red jersey in shadow and a red jersey in sunlight
    # have different RGB values but nearly the same Hue value.
    hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)

    # Return the mean HSV values across all pixels in the torso crop
    return hsv_crop.mean(axis=(0, 1))  # Mean across height and width axes


def classify_teams(
    tracks: List[PlayerTrack],
    frames: List[Tuple[int, np.ndarray]]
) -> List[PlayerTrack]:
    """
    Assign team labels to all tracks using K-means clustering on jersey colors.

    Process:
    1. Build a frame lookup dictionary for fast access (frame_number → frame image)
    2. Extract jersey color for every track
    3. Run K-means with NUM_TEAMS=4 clusters
    4. Map cluster IDs to team labels (0=team_a, 1=team_b, 2=goalkeeper, 3=referee)
    5. Update the team field on every PlayerTrack object
    """

    # Build a lookup: frame_number → frame image
    # This avoids scanning the entire frames list for each track (O(n) → O(1))
    frame_lookup: Dict[int, np.ndarray] = {fn: frame for fn, frame in frames}

    # Extract jersey colors for all tracks
    colors = []
    for track in tracks:
        frame = frame_lookup.get(track.frame_number)
        if frame is None:
            colors.append(np.array([0.0, 0.0, 0.0]))
            continue

        bbox = (track.bbox_x1, track.bbox_y1, track.bbox_x2, track.bbox_y2)
        color = extract_jersey_color(frame, bbox)
        colors.append(color)

    colors_array = np.array(colors)
    # Shape: (num_tracks, 3) — one HSV color vector per track

    # Normalize features before K-means.
    # StandardScaler converts each feature to mean=0, std=1.
    # This prevents the Hue dimension (0-180) from dominating over
    # Saturation and Value (0-255) just because of its larger range.
    scaler = StandardScaler()
    colors_normalized = scaler.fit_transform(colors_array)
    # fit_transform: compute the mean/std (fit) and apply normalization (transform)

    # Run K-means clustering
    # n_clusters=NUM_TEAMS (4): we expect 4 color groups
    # n_init=10: run K-means 10 times with different starting points,
    #            keep the best result (avoids local optima)
    # random_state=42: seed for reproducibility (same result every run)
    kmeans = KMeans(n_clusters=NUM_TEAMS, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(colors_normalized)
    # cluster_labels: array of shape (num_tracks,) with values 0-3,
    # one cluster assignment per track

    # Map raw cluster IDs to meaningful team labels.
    # K-means doesn't know which cluster is "home team" vs "away team".
    # We identify the referee cluster by finding the cluster whose
    # center has the lowest Saturation in HSV — referees wear black/dark
    # outfits with low color saturation, while jersey colors are vivid.
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    # inverse_transform: convert normalized centers back to original HSV space
    saturation_per_cluster = cluster_centers[:, 1]  # Column 1 = Saturation channel
    referee_cluster = int(np.argmin(saturation_per_cluster))
    # argmin returns the index of the minimum value — that's our referee cluster

    # For the remaining 3 clusters, the 2 most populated ones are the playing teams,
    # and the least populated is goalkeepers (only 2 per game vs 10 outfield players).
    non_referee_clusters = [c for c in range(NUM_TEAMS) if c != referee_cluster]
    cluster_sizes = [np.sum(cluster_labels == c) for c in non_referee_clusters]
    sorted_by_size = sorted(zip(non_referee_clusters, cluster_sizes), key=lambda x: -x[1])
    # sorted by size descending: largest cluster = team_a, second = team_b, smallest = goalkeeper

    team_a_cluster = sorted_by_size[0][0]
    team_b_cluster = sorted_by_size[1][0]
    goalkeeper_cluster = sorted_by_size[2][0]

    # Build a mapping from K-means cluster ID to our team label (0, 1, 2, 3)
    cluster_to_team = {
        team_a_cluster: 0,       # Home team
        team_b_cluster: 1,       # Away team
        goalkeeper_cluster: 2,   # Goalkeepers
        referee_cluster: 3       # Referees
    }

    print(f"  Cluster mapping: {cluster_to_team}")
    # Assign initial team labels
    for i, track in enumerate(tracks):
        track.team = cluster_to_team[cluster_labels[i]]

    # ── Sanity check: referees should be rare ────────────────────────────
    # In a real match there are at most 4 referees. If we're seeing >15%
    # of all detections labelled as referees, the clustering misfired.
    # Reassign the excess referee-labelled detections to the closest
    # playing team cluster based on color distance.
    referee_tracks = [t for t in tracks if t.team == 3]
    referee_ratio = len(referee_tracks) / max(len(tracks), 1)

    if referee_ratio > 0.15:
        print(f"  ⚠️  Referee ratio {referee_ratio:.1%} too high — correcting...")

        ref_center = cluster_centers[referee_cluster]
        team_a_center = cluster_centers[team_a_cluster]
        team_b_center = cluster_centers[team_b_cluster]

        corrected = 0
        for i, track in enumerate(tracks):
            if track.team != 3:
                continue
            color = colors_array[i]
            dist_a = np.linalg.norm(color - team_a_center)
            dist_b = np.linalg.norm(color - team_b_center)
            # Only re-assign if this track's color is closer to a team
            # than to the referee center itself
            dist_ref = np.linalg.norm(color - ref_center)
            if min(dist_a, dist_b) < dist_ref:
                track.team = 0 if dist_a < dist_b else 1
                corrected += 1

        print(f"  Corrected {corrected} misclassified referee detections")

    team_counts = defaultdict(int)
    for track in tracks:
        team_counts[track.team] += 1
    print(f"  Team A detections: {team_counts[0]}")
    print(f"  Team B detections: {team_counts[1]}")
    print(f"  Goalkeeper detections: {team_counts[2]}")
    print(f"  Referee detections: {team_counts[3]}")

    return tracks
