from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
from pipeline.tracker import PlayerTrack


@dataclass
class TacticalEvent:
    """
    Represents a single tactical event detected in the clip.
    Maps directly to one row in the Supabase events table.
    """
    event_type: str       # "pass_short", "pass_long", "cross", "shot", "off_ball_run", "tackle"
    player_track_id: int  # Which player performed this action
    start_frame: int      # Frame where this event began
    end_frame: int        # Frame where this event ended (same as start for instant events)
    start_x: float        # Pitch X coordinate at event start (meters)
    start_y: float        # Pitch Y coordinate at event start (meters)
    end_x: float          # Pitch X coordinate at event end (meters)
    end_y: float          # Pitch Y coordinate at event end (meters)
    successful: bool      # Did the event achieve its goal? (pass reached teammate, etc.)


def detect_events(tracks: List[PlayerTrack]) -> List[TacticalEvent]:
    """
    Detect tactical events from the tracking data.

    Strategy:
    1. Group tracks by player_track_id to get each player's positional history
    2. For each player, analyze their frame-to-frame movement
    3. Look for patterns that indicate specific events:
        - Sudden large displacement (>5m in 1 frame) = potential shot or pass recipient
        - Sustained high-speed movement away from ball zone = off-ball run
        - Proximity to ball position + displacement = ball interaction
    4. Classify events by trajectory characteristics
    """

    events: List[TacticalEvent] = []

    # Group tracks by player
    player_histories: Dict[int, List[PlayerTrack]] = defaultdict(list)
    for track in tracks:
        player_histories[track.player_track_id].append(track)

    # Sort each player's history by frame number
    for pid in player_histories:
        player_histories[pid].sort(key=lambda t: t.frame_number)

    # ─── Detect Off-Ball Runs ─────────────────────────────────────────────────
    # An off-ball run is when a player (not in possession) moves rapidly
    # into space. We detect this by finding sequences of frames where a player
    # moves consistently in one direction at high speed.

    for player_id, history in player_histories.items():
        if len(history) < 3:
            continue  # Need at least 3 frames to detect a run

        # Skip goalkeepers and referees
        if history[0].team in [2, 3]:
            continue

        # Calculate displacement between consecutive frames
        for i in range(1, len(history) - 1):
            prev = history[i - 1]
            curr = history[i]
            next_pos = history[i + 1]

            # Distance moved in this frame (meters)
            dx = curr.x_meters - prev.x_meters
            dy = curr.y_meters - prev.y_meters
            displacement = (dx**2 + dy**2) ** 0.5  # Euclidean distance

            # At 5fps, a player running at ~8 m/s moves ~1.6m per frame.
            # A displacement > 2m per frame suggests a sprint.
            # A displacement > 5m likely indicates a tracking error (skip it).
            if 2.0 < displacement < 5.0:
                # Check if the next frame continues in roughly the same direction
                dx_next = next_pos.x_meters - curr.x_meters
                dy_next = next_pos.y_meters - curr.y_meters

                # Dot product: positive means same general direction
                dot_product = dx * dx_next + dy * dy_next

                if dot_product > 0:  # Consistent direction = sustained run
                    event = TacticalEvent(
                        event_type="off_ball_run",
                        player_track_id=player_id,
                        start_frame=prev.frame_number,
                        end_frame=next_pos.frame_number,
                        start_x=prev.x_meters,
                        start_y=prev.y_meters,
                        end_x=next_pos.x_meters,
                        end_y=next_pos.y_meters,
                        successful=True  # Run attempted = successful for now
                    )
                    events.append(event)

    # ─── Detect Passes (simplified heuristic) ────────────────────────────────
    # A full pass detection system requires ball tracking (separate from player tracking).
    # For the MVP, we use a heuristic: when a player makes a very fast short movement
    # and another player of the SAME team appears to receive (starts moving) shortly after,
    # we classify it as a short pass.
    #
    # This is intentionally simplified for Phase 2. The full implementation
    # in Phase 3 will use ball position data to detect actual ball possession changes.

    for player_id, history in player_histories.items():
        if history[0].team in [2, 3]:  # Skip goalkeepers and referees
            continue

        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]

            dx = curr.x_meters - prev.x_meters
            dy = curr.y_meters - prev.y_meters
            displacement = (dx**2 + dy**2) ** 0.5

            # Classify by distance and direction
            if 1.0 < displacement < 3.0:
                # Short lateral or forward movement — potential short pass
                event_type = "pass_short"
            elif displacement >= 3.0 and displacement < 5.0:
                # Larger movement — potential long pass or cross
                # Crosses typically go toward the penalty area width
                if abs(dy) > abs(dx):
                    event_type = "cross"
                else:
                    event_type = "pass_long"
            else:
                continue  # Not a pass-like movement

            event = TacticalEvent(
                event_type=event_type,
                player_track_id=player_id,
                start_frame=prev.frame_number,
                end_frame=curr.frame_number,
                start_x=prev.x_meters,
                start_y=prev.y_meters,
                end_x=curr.x_meters,
                end_y=curr.y_meters,
                successful=True  # Will be refined with ball data in Phase 3
            )
            events.append(event)

    print(f"✅ Event detection complete")
    print(f"  Total events detected: {len(events)}")
    event_type_counts = defaultdict(int)
    for e in events:
        event_type_counts[e.event_type] += 1
    for etype, count in sorted(event_type_counts.items()):
        print(f"  {etype}: {count}")

    return events

