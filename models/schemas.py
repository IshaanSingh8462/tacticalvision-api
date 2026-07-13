from pydantic import BaseModel
from typing import Optional

class ProcessVideoRequest(BaseModel):
    """
    The JSON body sent by Next.js when the user submits a video for processing.

    video_id: The UUID of the video row already created in Supabase by the frontend.
            The frontend creates the videos row and uploads to Storage first,
            then sends this request to start processing.

    user_id: The authenticated user's UUID from Supabase Auth.
            The frontend reads this from the session and passes it here.
            The server verifies limits using this ID.

    storage_url: The full public URL to the uploaded video in Supabase Storage.
                FastAPI downloads the video from this URL before processing.

    clip_start_seconds: Where in the video the user's selected 2-min window starts.
    clip_duration_seconds: How long the clip is (default 120 for 2 minutes).

    is_paid: Whether this user is on a paid tier. The server ALSO checks this
            independently — this field is a hint for fast-path decisions,
            not a security control.
    """
    video_id: str
    user_id: str
    storage_url: str
    clip_start_seconds: float = 0.0
    clip_duration_seconds: float = 120.0
    is_paid: bool = False


class ProcessVideoResponse(BaseModel):
    """
    Returned immediately (within 200ms) when a job is accepted.
    The actual processing happens asynchronously in the background.
    """
    job_id: str
    status: str = "queued"
    message: str = "Processing started. Subscribe to job updates via Supabase Realtime."


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    homography_computed: bool


