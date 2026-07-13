import subprocess
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple
import cv2
import numpy as np

class ClipExtractorBase(ABC):
    """
    Abstract interface for video clip extraction.

    Any class that inherits from this MUST implement the extract() method.
    This means you can swap out FFmpegClipExtractor for any other implementation
    (MoviePyClipExtractor, CloudClipExtractor, etc.) without changing any code
    that CALLS the extractor — as long as the new class also inherits from this base.

    This is the "swappable interface" design that makes the pipeline future-proof.
    """

    @abstractmethod
    def extract(
        self,
        input_path: str,
        output_path: str,
        start_seconds: float,
        duration_seconds: float
    ) -> str:
        """
        Extract a clip from a video file.

        Args:
            input_path: Path to the source video file
            output_path: Where to save the extracted clip
            start_seconds: Start time of the clip in seconds
            duration_seconds: Length of the clip in seconds

        Returns:
            The output_path where the clip was saved (for chaining)
        """
        pass  # Abstract — no implementation here


class FFmpegClipExtractor(ClipExtractorBase):
    """
    Concrete implementation of ClipExtractorBase using FFmpeg.

    FFmpeg is the industry-standard video processing tool. It's fast, handles
    virtually every video format, and can extract clips without re-encoding
    (using -c copy), which is nearly instant regardless of clip length.
    """

    def extract(
        self,
        input_path: str,
        output_path: str,
        start_seconds: float,
        duration_seconds: float
    ) -> str:
        """
        Uses FFmpeg's subprocess interface to extract a clip.

        The -ss flag (seek) is placed BEFORE -i (input) intentionally.
        When -ss comes before -i, FFmpeg seeks to the position before
        opening the file — this is called "input seeking" and is much
        faster than "output seeking" (placing -ss after -i), especially
        for large files.

        -c copy means copy the video/audio streams without re-encoding.
        Re-encoding would take minutes and slightly degrade quality.
        Copy is instant and lossless.
        """
        print(f"  Extracting clip: {start_seconds}s → {start_seconds + duration_seconds}s")

        command = [
            'ffmpeg',
            '-y',                           # Overwrite output file if it exists
            '-ss', str(start_seconds),      # Seek to start position BEFORE input
            '-i', input_path,               # Input video file
            '-t', str(duration_seconds),    # Duration to extract
            '-c', 'copy',                   # Copy streams without re-encoding
            '-avoid_negative_ts', 'make_zero',  # Fix potential timestamp issues
            output_path                     # Output file path
        ]

        # subprocess.run executes the FFmpeg command as if you typed it in terminal.
        # capture_output=True captures stdout/stderr so they don't clutter the notebook.
        # check=True raises a Python exception if FFmpeg exits with an error code.
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Verify the output file actually exists and has size > 0
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"FFmpeg produced empty or missing output: {output_path}")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✅ Clip extracted: {output_path} ({file_size_mb:.1f} MB)")
        return output_path


def extract_frames_at_fps(video_path: str, target_fps: int) -> List[Tuple[int, np.ndarray]]:
    """
    Read a video and return sampled frames at the target frame rate.

    Instead of reading every frame (wasteful), we calculate which frame numbers
    correspond to our target sample rate and only decode those.

    Returns a list of (frame_number, frame_image) tuples.
    frame_number is the sequential index at TARGET fps (0, 1, 2, 3...)
    frame_image is a numpy array of shape (height, width, 3) in BGR color format.
    BGR is OpenCV's default color ordering (Blue, Green, Red — reversed from RGB).
    """
    cap = cv2.VideoCapture(video_path)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # How many source frames correspond to one sample frame?
    # e.g. source is 30fps, target is 5fps → sample every 6th frame
    frame_interval = int(round(source_fps / target_fps))

    frames = []
    sample_frame_number = 0  # Our own counter at target_fps

    for source_frame_idx in range(0, total_frames, frame_interval):
        # Jump directly to the frame we want (much faster than reading sequentially)
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
        # CAP_PROP_POS_FRAMES tells OpenCV to seek to this frame index.

        ret, frame = cap.read()
        # cap.read() returns (success_bool, frame_array)
        # ret is False if we've reached the end of the video or there's a read error

        if not ret:
            break

        frames.append((sample_frame_number, frame))
        sample_frame_number += 1

    cap.release()
    print(f"  ✅ Extracted {len(frames)} frames at {target_fps}fps from {Path(video_path).name}")
    return frames
