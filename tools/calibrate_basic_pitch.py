"""Calibrate Basic Pitch against a known clean guitar chart."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from basic_pitch.inference import predict
from tools.compare_charts import Note, read_midi, timing_match


def grouped_events(events, confidence: float, minimum_note_length: float) -> list[Note]:
    filtered = [event for event in events if float(event[3]) >= confidence and (float(event[1]) - float(event[0])) * 1000 >= minimum_note_length]
    filtered.sort(key=lambda event: float(event[0]))
    groups = []
    for event in filtered:
        if not groups or float(event[0]) - float(groups[-1][0][0]) > 0.11:
            groups.append([event])
        else:
            groups[-1].append(event)
    return [Note(float(group[0][0]) * 1000, frozenset(range(len(group))), max(0.0, (max(float(event[1]) for event in group) - float(group[0][0])) * 1000)) for group in groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print("Running Basic Pitch once on the clean guitar stem…")
    _, _, predicted = predict(args.audio, onset_threshold=0.35, frame_threshold=0.2, minimum_note_length=60)
    reference = read_midi(args.reference / "notes.mid", "Expert")
    best = None
    for confidence in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        for minimum_length in (60, 90, 120, 160, 220):
            generated = grouped_events(predicted, confidence, minimum_length)
            precision, recall = timing_match(reference, generated)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            candidate = (f1, confidence, minimum_length, precision, recall, len(generated))
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    config = {
        "note_confidence": best[1],
        "onset_threshold": 0.35,
        "frame_threshold": 0.2,
        "minimum_note_length": best[2],
    }
    Path("ai_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Best F1: {best[0] * 100:.1f}%")
    print(f"Precision: {best[3] * 100:.1f}%  Recall: {best[4] * 100:.1f}%  Notes: {best[5]}")
    print(f"Saved calibration to {Path('ai_config.json').resolve()}")


if __name__ == "__main__":
    main()
