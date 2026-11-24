"""Optional Demucs-based source separation for ChartForge."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def get_or_create_guitar_stem(audio_path: Path, cache_dir: Path | None = None) -> Path:
    """Return a local guitar stem, creating it with Demucs when needed."""
    # Quality-first cache: separation is expensive, so reuse it on later runs.
    destination = cache_dir or audio_path.parent / ".chartforge-stems-quality"
    destination.mkdir(parents=True, exist_ok=True)
    model_name = "htdemucs_6s"
    expected = destination / model_name / audio_path.stem / "guitar.wav"
    if expected.exists():
        return expected

    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model_name,
        "--shifts",
        "2",
        "-o",
        str(destination),
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Demucs source separation failed")
    if not expected.exists():
        raise RuntimeError(f"Demucs completed but did not create {expected}")
    return expected
