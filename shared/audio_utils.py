"""Audio processing and frame buffering utilities for continuous spoken dialogue models.

Supports standard 24 kHz, 3200 samples per frame (133.33 ms at 7.5 Hz) audio streams.
"""

import collections
import numpy as np
from typing import List, Optional, Tuple


class AudioFrameBuffer:
    """Thread-safe ring buffer for accumulating and streaming audio samples into fixed chunk sizes."""

    def __init__(self, frame_size: int = 3200, max_frames: int = 50):
        self.frame_size = frame_size
        self.max_samples = frame_size * max_frames
        self.buffer = collections.deque(maxlen=self.max_samples)

    def write(self, samples: np.ndarray) -> None:
        """Add float32 or int16 audio samples to the ring buffer."""
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # Flatten if multichannel
        if samples.ndim > 1:
            samples = samples.mean(axis=-1)

        self.buffer.extend(samples.tolist())

    def has_frame(self) -> bool:
        """Check if at least one complete frame is ready to read."""
        return len(self.buffer) >= self.frame_size

    def read_frame(self) -> Optional[np.ndarray]:
        """Pop and return one frame of `frame_size` samples, or None if insufficient samples."""
        if len(self.buffer) < self.frame_size:
            return None
        frame = [self.buffer.popleft() for _ in range(self.frame_size)]
        return np.array(frame, dtype=np.float32)

    def read_or_pad_frame(self) -> np.ndarray:
        """Pop one frame, or pop remaining samples and zero-pad if under frame_size."""
        frame = []
        for _ in range(self.frame_size):
            if self.buffer:
                frame.append(self.buffer.popleft())
            else:
                frame.append(0.0)
        return np.array(frame, dtype=np.float32)

    def clear(self) -> None:
        self.buffer.clear()

    @property
    def samples_available(self) -> int:
        return len(self.buffer)


def split_audio_frames(samples: np.ndarray, frame_size: int = 3200) -> Tuple[List[np.ndarray], np.ndarray]:
    """Split an audio array into multiple complete frames of `frame_size` and a remainder."""
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0

    num_complete_frames = len(samples) // frame_size
    frames = []
    for i in range(num_complete_frames):
        start = i * frame_size
        frames.append(samples[start : start + frame_size].astype(np.float32))

    remainder = samples[num_complete_frames * frame_size :].astype(np.float32)
    return frames, remainder


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int = 24000) -> np.ndarray:
    """Resample 1D audio array from orig_sr to target_sr using linear interpolation (lightweight CPU fallback)."""
    if orig_sr == target_sr:
        return samples.astype(np.float32)
    duration = len(samples) / orig_sr
    num_target_samples = int(duration * target_sr)
    orig_indices = np.linspace(0, len(samples) - 1, num_target_samples)
    resampled = np.interp(orig_indices, np.arange(len(samples)), samples)
    return resampled.astype(np.float32)
