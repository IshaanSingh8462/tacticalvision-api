# Start from Python 3.11 slim — minimal Debian Linux with Python pre-installed.
# "slim" means no unnecessary packages, keeping the image small (~150MB base).
# We use 3.11 specifically because Ultralytics 8.4.x is tested against it.
FROM python:3.11-slim

# Set working directory inside the container.
# All subsequent commands run from /app.
WORKDIR /app

# Install system dependencies FIRST (before Python packages).
# These are OS-level tools that Python packages build on top of:
# - ffmpeg: the video processing binary our FFmpegClipExtractor calls
# - libgl1: OpenCV requires this graphics library even in headless mode
# - libglib2.0-0: required by OpenCV on Debian
# We combine into one RUN command to minimize Docker layers.
# apt-get clean and rm -rf /var/lib/apt/lists/* remove the package cache
# after install, keeping the image smaller.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt BEFORE copying the rest of the code.
# Why? Docker layer caching: if requirements.txt hasn't changed between builds,
# Docker skips the pip install step entirely (uses cached layer).
# This makes rebuilds after code-only changes much faster.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir: don't store pip's download cache in the image (saves space)
# --upgrade pip: ensure pip itself is current before installing
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container.
# The .dockerignore file (if present) controls what gets excluded.
COPY . .

# Create the output directory the pipeline writes temporary files to.
# exist_ok equivalent — won't fail if it exists.
RUN mkdir -p /app/output

# Expose port 8000. This tells Docker (and Render) that the container
# listens on this port. Render maps this to its own HTTPS port automatically.
EXPOSE 8000

# The command that runs when the container starts.
# uvicorn: the ASGI server that runs FastAPI
# main:app: look for the "app" object in main.py
# --host 0.0.0.0: listen on all network interfaces (required in containers —
#   127.0.0.1 would only accept connections from inside the container itself)
# --port 8000: match the EXPOSE above
# --workers 1: one worker process (Render free tier has 512MB RAM;
#   multiple workers would each load the YOLO model separately, crashing the server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
