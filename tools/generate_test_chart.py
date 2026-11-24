"""Generate an internal comparison chart from a full-mix audio file."""

from __future__ import annotations

import shutil
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import analyze_audio, analyze_with_basic_pitch, audio_duration_seconds, build_chart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--separate", action="store_true")
    args = parser.parse_args()
    source = args.audio.resolve()
    destination = args.destination.resolve()
    analysis_source = source
    if args.separate:
        from source_separator import get_or_create_guitar_stem
        analysis_source = get_or_create_guitar_stem(source)
    try:
        bpm, stem_duration, onsets, chord_times, sustain_times, pitch_times = analyze_with_basic_pitch(analysis_source)
        print("Analysis: Basic Pitch AI")
    except Exception:
        bpm, stem_duration, onsets, chord_times, sustain_times = analyze_audio(analysis_source)
        pitch_times = None
        print("Analysis: classic audio analysis")
    duration = audio_duration_seconds(source) or stem_duration
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "notes.chart").write_text(
        build_chart("Song 2", "Blur", "RoboRiff", round(bpm), duration, onsets, chord_times, sustain_times, pitch_times),
        encoding="utf-8",
    )
    (destination / "song.ini").write_text(
        f"[song]\nname = Song 2\nartist = Blur\ncharter = RoboRiff\nsong_length = {int(duration * 1000)}\n",
        encoding="utf-8",
    )
    shutil.copy2(source, destination / "song.ogg")
    print(f"Created: {destination}")
    print(f"BPM: {bpm:.1f}  Duration: {duration:.1f}s  Onsets: {len(onsets)}")


if __name__ == "__main__":
    main()
