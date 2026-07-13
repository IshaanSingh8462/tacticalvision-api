import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from ultralytics import YOLO
from contextlib import asynccontextmanager

from pipeline.homography import compute_default_homography
from routers import process
from models.schemas import HealthResponse

load_dotenv()

# ── Application State ─────────────────────────────────────────────────────────
# We load the YOLO model and compute homography ONCE when the server starts,
# then store them in app.state so every request can access them without
# reloading. Loading YOLO takes ~3 seconds and uses ~200MB RAM —
# doing this per-request would make the server unusably slow.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — runs startup code before the server begins
    accepting requests, and cleanup code when it shuts down.

    This replaces the deprecated @app.on_event("startup") pattern.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    print("Starting TacticalVision AI server...")

    # Load YOLO model
    # yolo11s.pt downloads automatically from Ultralytics on first run (~22MB).
    # On subsequent starts (and in the Docker container after first build),
    # it loads from the local cache.
    print("  Loading YOLO11s model...")
    app.state.model = YOLO("yolo11s.pt")
    print("  ✅ YOLO model loaded")

    # Compute homography matrix from default broadcast reference points
    print("  Computing homography matrix...")
    app.state.H = compute_default_homography()
    print("  ✅ Homography computed")

    print("✅ Server ready")
    yield  # Server runs here — everything after yield is shutdown code

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    print("Server shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TacticalVision AI API",
    description="Soccer video analysis pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Configuration ────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) controls which domains can call this API.
# Without this, browsers block requests from your Vercel frontend to Render.
#
# allow_origins: list of frontend URLs that can make requests.
# In development this includes localhost. In production, your Vercel URL.
# The environment variable ALLOWED_ORIGINS lets you set this without
# changing code between environments.

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"   # default to local dev
).split(",")
# On Render, set ALLOWED_ORIGINS=https://tacticalvision-ai.vercel.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,   # Required for requests with auth cookies/headers
    allow_methods=["*"],      # Allow GET, POST, PUT, DELETE, OPTIONS, etc.
    allow_headers=["*"],      # Allow Content-Type, Authorization, etc.
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(process.router, prefix="/api")
# All routes in process.py will be prefixed with /api
# e.g. POST /api/process-video, GET /api/job/{job_id}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint. UptimeRobot pings this every 10 minutes
    to prevent Render's free tier from spinning down the server.

    Also useful for verifying the server started correctly —
    if model_loaded is False, the YOLO download failed.
    """
    return HealthResponse(
        status="ok",
        model_loaded=hasattr(app.state, "model") and app.state.model is not None,
        homography_computed=hasattr(app.state, "H") and app.state.H is not None,
    )


@app.get("/")
async def root():
    return {"message": "TacticalVision AI API", "docs": "/docs"}

