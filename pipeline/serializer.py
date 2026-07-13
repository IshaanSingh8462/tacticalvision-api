from typing import List, Dict
from pipeline.tracker import PlayerTrack, BallTrack
from pipeline.events import TacticalEvent


def serialize_tracks_to_json(tracks: List[PlayerTrack], video_id: str = "TEST_VIDEO_ID") -> List[Dict]:
    """
    Convert PlayerTrack objects to the JSON format expected by Supabase.
    Each dict in the returned list corresponds to one row in player_tracks table.
    """
    rows = []
    for track in tracks:
        rows.append({
            "video_id": video_id,
            "player_track_id": track.player_track_id,
            "team": track.team,
            "frame_number": track.frame_number,
            "x_meters": round(track.x_meters, 3),  # Round to 3 decimal places (mm precision)
            "y_meters": round(track.y_meters, 3),
            "pixel_x": round(track.pixel_x, 1),
            "pixel_y": round(track.pixel_y, 1),
            "confidence": round(track.confidence, 4),
            "bbox_x1": round(track.bbox_x1, 1),
            "bbox_y1": round(track.bbox_y1, 1),
            "bbox_x2": round(track.bbox_x2, 1),
            "bbox_y2": round(track.bbox_y2, 1),
        })
    return rows


def serialize_events_to_json(events: List[TacticalEvent], video_id: str = "TEST_VIDEO_ID") -> List[Dict]:
    """
    Convert TacticalEvent objects to the JSON format expected by Supabase.
    Each dict corresponds to one row in the events table.
    """
    rows = []
    for event in events:
        rows.append({
            "video_id": video_id,
            "player_track_id": event.player_track_id,
            "type": event.event_type,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "start_x": round(event.start_x, 3),
            "start_y": round(event.start_y, 3),
            "end_x": round(event.end_x, 3),
            "end_y": round(event.end_y, 3),
            "successful": event.successful,
        })
    return rows


def serialize_ball_tracks_to_json(
    ball_tracks: List[BallTrack],
    video_id: str
) -> List[Dict]:
    rows = []
    for track in ball_tracks:
        rows.append({
            "video_id": video_id,
            "frame_number": track.frame_number,
            "x_meters": round(track.x_meters, 3),
            "y_meters": round(track.y_meters, 3),
            "pixel_x": round(track.pixel_x, 1),
            "pixel_y": round(track.pixel_y, 1),
            "confidence": round(track.confidence, 4),
        })
    return rows
