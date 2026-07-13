import os
import uuid
import tempfile
import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from datetime import datetime, timezone

from models.schemas import ProcessVideoRequest, ProcessVideoResponse
from services.supabase_client import get_supabase
from services.freemium import check_and_enforce_limits, increment_upload_count
from pipeline.extractor import FFmpegClipExtractor, extract_frames_at_fps
from pipeline.tracker import run_detection_and_tracking
from pipeline.classifier import classify_teams
from pipeline.homography import apply_homography_to_tracks, apply_homography_to_ball_tracks
from pipeline.events import detect_events
from pipeline.serializer import (
    serialize_tracks_to_json,
    serialize_ball_tracks_to_json,
    serialize_events_to_json,
)

router = APIRouter()

# Configuration constants (same as Colab)
SAMPLE_FPS = 5
CONFIDENCE_THRESHOLD = 0.2
BALL_CONFIDENCE_THRESHOLD = 0.04


# ── Helper: Update Job Progress ───────────────────────────────────────────────

def update_job_progress(
    job_id: str,
    status: str,
    progress_pct: int,
    stage_label: str,
    supabase,
    error_message: str = None,
):
    """
    Write a progress update to the Supabase jobs table.

    Supabase Realtime broadcasts this change to the Next.js frontend
    automatically — no polling required. The frontend's useJobStatus hook
    receives the update and reveals the appropriate UI panel.

    This is the function that makes the "feels live" effect work.
    Each call corresponds to one panel reveal or loading state update
    in the frontend.
    """
    update_data = {
        "status": status,
        "progress_pct": progress_pct,
        "current_stage_label": stage_label,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error_message:
        update_data["error_message"] = error_message

    supabase.table("jobs").update(update_data).eq("id", job_id).execute()


# ── Background Processing Task ────────────────────────────────────────────────

async def run_pipeline_background(
    job_id: str,
    video_id: str,
    user_id: str,
    storage_url: str,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    is_paid: bool,
    model,       # YOLO model from app.state (passed in, not imported)
    H,           # Homography matrix from app.state
):
    """
    The full processing pipeline running asynchronously.

    FastAPI's BackgroundTasks runs this function AFTER the HTTP response
    has already been sent to the client. The client receives job_id immediately,
    then this function runs for 5-10 minutes in the background.

    Progress updates via update_job_progress() trigger Supabase Realtime
    events which the frontend receives in real time.
    """
    supabase = get_supabase()
    tmp_dir = tempfile.mkdtemp()
    # tempfile.mkdtemp() creates a unique temporary directory like /tmp/abc123/
    # We write all intermediate files here (downloaded video, extracted clip)
    # and clean up at the end.

    try:
        # ── Stage 1: Download video from Supabase Storage ─────────────────────
        update_job_progress(job_id, "downloading", 5,
                            "Downloading video...", supabase)

        video_path = os.path.join(tmp_dir, "source.mp4")

        # httpx downloads the video file from Supabase Storage URL.
        # follow_redirects=True handles Supabase's signed URL redirects.
        # timeout=300 allows up to 5 minutes for large video downloads.
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
            response = await client.get(storage_url)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download video: HTTP {response.status_code}")
            with open(video_path, "wb") as f:
                f.write(response.content)

        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  Downloaded: {file_size_mb:.1f} MB")

        # ── Stage 2: Extract clip ──────────────────────────────────────────────
        update_job_progress(job_id, "extracting_clip", 15,
                            "Extracting clip...", supabase)

        clip_path = os.path.join(tmp_dir, "clip.mp4")
        extractor = FFmpegClipExtractor()
        extractor.extract(video_path, clip_path, clip_start_seconds, clip_duration_seconds)

        # Sample frames from the clip
        frames = extract_frames_at_fps(clip_path, SAMPLE_FPS)
        print(f"  Sampled {len(frames)} frames at {SAMPLE_FPS}fps")

        # ── Stage 3: YOLO + ByteTrack ─────────────────────────────────────────
        update_job_progress(job_id, "running_yolo", 30,
                            "Identifying players...", supabase)

        tracks, ball_tracks = run_detection_and_tracking(
            frames=frames,
            model=model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            ball_confidence_threshold=BALL_CONFIDENCE_THRESHOLD,
            clip_path=clip_path,
            sample_fps=SAMPLE_FPS,
        )
        print(f"  Tracks: {len(tracks)}, Unique players: {len(set(t.player_track_id for t in tracks))}")

        # ── Stage 4: Team classification ──────────────────────────────────────
        update_job_progress(job_id, "running_yolo", 50,
                            "Classifying teams...", supabase)

        tracks = classify_teams(tracks, frames)

        # ── Stage 5: Homography projection ────────────────────────────────────
        update_job_progress(job_id, "running_homography", 65,
                            "Building 2D pitch map...", supabase)

        tracks = apply_homography_to_tracks(tracks, frames, H)
        ball_tracks = apply_homography_to_ball_tracks(ball_tracks, H)

        # ── Stage 6: Tactical event detection (paid only) ─────────────────────
        events = []
        if is_paid:
            update_job_progress(job_id, "running_homography", 75,
                                "Detecting tactical events...", supabase)
            events = detect_events(tracks)
            print(f"  Events detected: {len(events)}")
        else:
            print("  Skipping event detection (free tier)")

        # ── Stage 7: Gemini AI analysis (paid only, 1/week free) ──────────────
        # Phase 4 will add this. For now, skip with a placeholder.
        update_job_progress(job_id, "running_gemini", 85,
                            "Generating AI analysis...", supabase)
        # TODO: Phase 4 inserts Gemini analysis here

        # ── Stage 8: Write results to Supabase ────────────────────────────────
        update_job_progress(job_id, "running_gemini", 90,
                            "Saving results...", supabase)

        # Insert player tracks in batches of 500 to avoid request size limits.
        # Supabase's REST API has a default max payload of ~1MB per request.
        # 2920 tracks × ~200 bytes each = ~580KB, so batching is required.
        tracks_data = serialize_tracks_to_json(tracks, video_id)
        BATCH_SIZE = 500
        for i in range(0, len(tracks_data), BATCH_SIZE):
            batch = tracks_data[i:i + BATCH_SIZE]
            supabase.table("player_tracks").insert(batch).execute()

        # Insert ball tracks
        if ball_tracks:
            ball_data = serialize_ball_tracks_to_json(ball_tracks, video_id)
            for i in range(0, len(ball_data), BATCH_SIZE):
                batch = ball_data[i:i + BATCH_SIZE]
                supabase.table("ball_tracks").insert(batch).execute()

        # Insert events (paid tier only)
        if events:
            events_data = serialize_events_to_json(events, video_id)
            for i in range(0, len(events_data), BATCH_SIZE):
                batch = events_data[i:i + BATCH_SIZE]
                supabase.table("events").insert(batch).execute()

        # ── Mark job complete ──────────────────────────────────────────────────
        # This final update triggers the frontend to reveal all panels
        # simultaneously — the "analysis complete" moment the user sees.
        update_job_progress(job_id, "complete", 100,
                            "Analysis complete!", supabase)
        print(f"✅ Job {job_id} complete")

    except Exception as e:
        # If anything fails at any stage, mark the job as errored.
        # The frontend shows an error state and offers a retry button.
        error_msg = str(e)
        print(f"❌ Job {job_id} failed: {error_msg}")
        update_job_progress(job_id, "error", 0,
                            "Processing failed", supabase,
                            error_message=error_msg)

    finally:
        # Always clean up temporary files, even if processing failed.
        # Render's free tier has limited disk space (~512MB).
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  Cleaned up temp dir: {tmp_dir}")


# ── Endpoint: POST /api/process-video ─────────────────────────────────────────

@router.post("/process-video", response_model=ProcessVideoResponse)
async def process_video_endpoint(
    request_body: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Accept a video processing request and return a job_id immediately.

    The actual processing happens asynchronously via BackgroundTasks.
    The client polls for progress via Supabase Realtime (not this endpoint).

    Flow:
    1. Validate request
    2. Check freemium limits
    3. Create job row (status: queued)
    4. Add pipeline to BackgroundTasks
    5. Return job_id — HTTP response sent here, processing begins after
    """
    supabase = get_supabase()

    # ── Freemium check ────────────────────────────────────────────────────────
    # We need the video duration to check the length limit.
    # For now we use a proxy: clip_duration_seconds from the request.
    # In Phase 5, the frontend will send the actual video duration.
    limit_check = await check_and_enforce_limits(
        user_id=request_body.user_id,
        video_duration_seconds=request_body.clip_duration_seconds,
        supabase=supabase,
    )

    if not limit_check["allowed"]:
        raise HTTPException(status_code=403, detail=limit_check["reason"])

    # ── Create job row ────────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    # We generate the UUID here rather than letting Supabase auto-generate it
    # so we can return it to the client before the insert completes.

    job_data = {
        "id": job_id,
        "video_id": request_body.video_id,
        "user_id": request_body.user_id,
        "status": "queued",
        "progress_pct": 0,
        "current_stage_label": "Queued for processing...",
    }
    supabase.table("jobs").insert(job_data).execute()

    # ── Increment upload counter ───────────────────────────────────────────────
    await increment_upload_count(request_body.user_id, supabase)

    # ── Add pipeline to background tasks ──────────────────────────────────────
    # BackgroundTasks.add_task() schedules run_pipeline_background() to run
    # AFTER this function returns and the HTTP response is sent.
    # The client receives job_id within ~200ms.
    # Processing then runs for 5-10 minutes in the background.
    background_tasks.add_task(
        run_pipeline_background,
        job_id=job_id,
        video_id=request_body.video_id,
        user_id=request_body.user_id,
        storage_url=request_body.storage_url,
        clip_start_seconds=request_body.clip_start_seconds,
        clip_duration_seconds=request_body.clip_duration_seconds,
        is_paid=limit_check["tier"] in ("pro", "team"),
        model=request.app.state.model,
        H=request.app.state.H,
    )

    return ProcessVideoResponse(
        job_id=job_id,
        status="queued",
        message="Processing started. Track progress via Supabase Realtime.",
    )


# ── Endpoint: GET /api/job/{job_id} ───────────────────────────────────────────

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Fallback endpoint for clients that can't use Supabase Realtime.
    Returns current job status. The frontend primarily uses Realtime,
    but this provides a REST alternative.
    """
    supabase = get_supabase()
    response = supabase.table("jobs") \
        .select("id, status, progress_pct, current_stage_label, error_message") \
        .eq("id", job_id) \
        .single() \
        .execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Job not found")

    return response.data
