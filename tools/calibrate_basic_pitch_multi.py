"""Calibrate Basic Pitch across multiple clean guitar-chart references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from basic_pitch.inference import predict
from tools.calibrate_basic_pitch import grouped_events
from tools.compare_charts import read_midi, timing_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("references", nargs="+", type=Path)
    args = parser.parse_args()
    datasets = []
    for folder in args.references:
        audio = folder / "guitar.ogg"
        midi = folder / "notes.mid"
        if audio.exists() and midi.exists():
            print(f"Transcribing {folder.name}…")
            _, _, predicted = predict(audio, onset_threshold=0.35, frame_threshold=0.2, minimum_note_length=60)
            datasets.append((predicted, read_midi(midi, "Expert")))
    if not datasets:
        raise SystemExit("No reference folders with guitar.ogg and notes.mid were found")

    best = None
    for confidence in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        for minimum_length in (60, 90, 120, 160, 220):
            precisions, recalls, total_notes = [], [], 0
            for predicted, reference in datasets:
                generated = grouped_events(predicted, confidence, minimum_length)
                precision, recall = timing_match(reference, generated)
                precisions.append(precision)
                recalls.append(recall)
                total_notes += len(generated)
            precision = sum(precisions) / len(precisions)
            recall = sum(recalls) / len(recalls)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            candidate = (f1, confidence, minimum_length, precision, recall, total_notes)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    config = {"note_confidence": best[1], "onset_threshold": 0.35, "frame_threshold": 0.2, "minimum_note_length": best[2]}
    Path("ai_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Best average F1: {best[0] * 100:.1f}%")
    print(f"Average precision: {best[3] * 100:.1f}%  recall: {best[4] * 100:.1f}%")
    print(f"Saved multi-song calibration to {Path('ai_config.json').resolve()}")


if __name__ == "__main__":
    main()
