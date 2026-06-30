"""Shared utilities for downloading, converting, and caching audio files.

Provides helpers used by both the recording audio cache and the ambient
noise cache to avoid duplicating download / ffmpeg / disk-cache logic.
"""

import asyncio
import os
import shutil
import tempfile
from typing import Literal, Optional

from loguru import logger

from api.constants import APP_ROOT_DIR

# ---------------------------------------------------------------------------
# Filesystem cache directory (shared by all audio caches)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(APP_ROOT_DIR), "dograh_pcm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


async def download_storage_file(
    storage_key: str,
    storage_backend: str,
    get_storage_fn,
) -> Optional[str]:
    """Download a file from object storage to a local temp file.

    Returns the temp file path on success, or None on failure.
    The caller is responsible for cleaning up the temp file.
    """
    ext = ext_from_key(storage_key)
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="dograh_dl_")
    os.close(fd)

    try:
        storage = get_storage_fn(storage_backend)
        success = await storage.adownload_file(storage_key, tmp_path)
        if not success:
            logger.error(f"Failed to download {storage_key}")
            _safe_unlink(tmp_path)
            return None
        return tmp_path
    except Exception:
        logger.exception(f"Error downloading {storage_key}")
        _safe_unlink(tmp_path)
        return None


# ---------------------------------------------------------------------------
# Audio conversion via ffmpeg
# ---------------------------------------------------------------------------


async def convert_audio_file(
    file_path: str,
    target_sample_rate: int,
    output_format: Literal["pcm", "wav"] = "pcm",
) -> Optional[bytes]:
    """Convert an audio file via ffmpeg.

    Args:
        file_path: Path to the source audio file.
        target_sample_rate: Desired output sample rate.
        output_format: ``"pcm"`` for raw s16le bytes, ``"wav"`` for a
            complete WAV file (16-bit mono).

    Returns:
        Converted audio bytes, or None on failure.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.error("ffmpeg not found on PATH - cannot convert audio")
        return None

    if output_format == "pcm":
        fmt_args = ["-f", "s16le", "-acodec", "pcm_s16le"]
    else:
        fmt_args = ["-f", "wav", "-acodec", "pcm_s16le"]

    cmd = [
        ffmpeg,
        "-i",
        file_path,
        *fmt_args,
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-loglevel",
        "error",
        "pipe:1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error(f"ffmpeg failed (rc={proc.returncode}): {stderr.decode()}")
            return None
        if not stdout:
            logger.error("ffmpeg produced no output")
            return None

        return stdout
    except Exception:
        logger.exception("ffmpeg subprocess error")
        return None


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def read_cached_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def write_cache_file(path: str, data: bytes) -> None:
    """Atomically write *data* to *path* (write-to-tmp then rename)."""
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    os.close(fd)
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def ext_from_key(storage_key: str) -> str:
    """Extract file extension from a storage key, defaulting to .wav."""
    _, ext = os.path.splitext(storage_key)
    return ext if ext else ".wav"


def _safe_unlink(path: str) -> None:
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Ambient noise file cache
# ---------------------------------------------------------------------------


def _ambient_noise_cache_path(storage_key: str, sample_rate: int) -> str:
    """Return the on-disk path for a cached ambient noise WAV file."""
    # Use a stable hash of the storage key so different uploads get different cache entries
    import hashlib

    key_hash = hashlib.sha256(storage_key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"ambient_{key_hash}_{sample_rate}.wav")


async def get_cached_ambient_noise_path(
    storage_key: str,
    storage_backend: str,
    target_sample_rate: int,
) -> Optional[str]:
    """Return a local WAV file path for a custom ambient noise file.

    Downloads from object storage and converts to mono WAV at
    *target_sample_rate* on the first call; subsequent calls return the
    cached path immediately.

    Args:
        storage_key: Object storage key for the uploaded audio file.
        storage_backend: Storage backend identifier (e.g. ``"minio"``, ``"s3"``).
        target_sample_rate: Target sample rate for the output WAV.

    Returns:
        Absolute path to the cached WAV file, or None on failure.
    """
    from api.services.storage import get_storage_for_backend

    cached = _ambient_noise_cache_path(storage_key, target_sample_rate)
    if os.path.exists(cached):
        logger.debug(f"Ambient noise served from cache: {cached}")
        return cached

    logger.info(f"Downloading custom ambient noise: {storage_key}")

    def _get_storage(backend: str):
        return get_storage_for_backend(backend)

    tmp_path = await download_storage_file(storage_key, storage_backend, _get_storage)
    if not tmp_path:
        return None

    try:
        wav_data = await convert_audio_file(
            tmp_path, target_sample_rate, output_format="wav"
        )
        if wav_data is None:
            return None

        write_cache_file(cached, wav_data)
        logger.info(f"Cached custom ambient noise: {cached} ({len(wav_data)} bytes)")
        return cached
    except Exception:
        logger.exception("Error caching ambient noise file")
        return None
    finally:
        _safe_unlink(tmp_path)


# ---------------------------------------------------------------------------
# TTS Phrase Cache (SSD-backed, MD5-keyed)
#
# Caches synthesised TTS audio on the 200 GB SSD volume so that repeated
# phrases (greetings, FAQs, confirmations) are served instantly without
# incurring CPU synthesis cost on the Kokoro-FastAPI service.
#
# Cache directory: /app/recordings/tts_cache/
# Key format:      <md5(text)>_<sample_rate>.wav
# ---------------------------------------------------------------------------

import hashlib as _hashlib

# Point at the SSD-mounted recordings volume (mapped via Docker volume in compose)
_TTS_PHRASE_CACHE_DIR = os.environ.get(
    "TTS_PHRASE_CACHE_DIR",
    os.path.join(os.path.dirname(APP_ROOT_DIR), "recordings", "tts_cache"),
)
os.makedirs(_TTS_PHRASE_CACHE_DIR, exist_ok=True)


def _tts_phrase_cache_path(text: str, sample_rate: int, voice: str = "") -> str:
    """Return the deterministic SSD cache path for a TTS phrase.

    Args:
        text: The exact TTS text to synthesise.
        sample_rate: Target audio sample rate (e.g. 24000).
        voice: Voice ID so different voices get separate cache files.

    Returns:
        Absolute path to the cache file (may or may not exist yet).
    """
    key = f"{text}|{voice}|{sample_rate}"
    md5 = _hashlib.md5(key.encode("utf-8")).hexdigest()
    safe_voice = "".join(c if c.isalnum() else "_" for c in voice)[:16]
    return os.path.join(_TTS_PHRASE_CACHE_DIR, f"tts_{safe_voice}_{md5}_{sample_rate}.wav")


def get_cached_tts_phrase(text: str, sample_rate: int, voice: str = "") -> Optional[bytes]:
    """Return cached WAV bytes for *text* if a cache hit exists, else None.

    Call this before invoking the Kokoro TTS service. A cache hit means
    0 ms synthesis latency — the audio bytes are read directly from SSD.

    Args:
        text: The TTS response text to look up.
        sample_rate: Audio sample rate for playback.
        voice: Voice ID used during synthesis.

    Returns:
        WAV bytes on cache hit, None on cache miss.
    """
    cache_path = _tts_phrase_cache_path(text, sample_rate, voice)
    if os.path.exists(cache_path):
        try:
            data = read_cached_file(cache_path)
            logger.debug(
                f"[tts_cache] HIT  | voice={voice} | rate={sample_rate} | "
                f"{len(text)} chars → {len(data)} bytes | {cache_path}"
            )
            return data
        except OSError:
            logger.warning(f"[tts_cache] Failed to read cache file: {cache_path}")
    return None


def save_tts_phrase_cache(
    text: str, sample_rate: int, audio_bytes: bytes, voice: str = ""
) -> None:
    """Atomically write synthesised TTS audio to the SSD phrase cache.

    Called immediately after a successful Kokoro synthesis so future
    requests for the same text are served from disk.

    Args:
        text: The TTS text that was synthesised.
        sample_rate: Audio sample rate of the output audio.
        audio_bytes: Raw WAV bytes to cache.
        voice: Voice ID used during synthesis.
    """
    cache_path = _tts_phrase_cache_path(text, sample_rate, voice)
    try:
        write_cache_file(cache_path, audio_bytes)
        logger.info(
            f"[tts_cache] MISS → saved | voice={voice} | rate={sample_rate} | "
            f"{len(text)} chars → {len(audio_bytes)} bytes | {cache_path}"
        )
    except OSError:
        logger.warning(f"[tts_cache] Failed to write cache file: {cache_path}")


def get_tts_cache_stats() -> dict:
    """Return basic stats about the TTS phrase cache for health/debug endpoints."""
    try:
        files = [
            f for f in os.listdir(_TTS_PHRASE_CACHE_DIR) if f.startswith("tts_")
        ]
        total_bytes = sum(
            os.path.getsize(os.path.join(_TTS_PHRASE_CACHE_DIR, f)) for f in files
        )
        return {
            "cache_dir": _TTS_PHRASE_CACHE_DIR,
            "cached_phrases": len(files),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        }
    except OSError:
        return {"cache_dir": _TTS_PHRASE_CACHE_DIR, "cached_phrases": 0, "total_size_mb": 0}
