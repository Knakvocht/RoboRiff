"""Internal ChartForge chart comparison tool.

Examples:
    py -3.13 tools/compare_charts.py \
      "Voorbeeld/Blur - Song 2 (Neversoft)" "output/Song 2"

The reference may contain notes.mid and the generated folder may contain notes.chart.
This tool is intentionally kept outside the user-facing application.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import mido


@dataclass(frozen=True)
class Note:
    time_ms: float
    lanes: frozenset[int]
    sustain_ms: float = 0.0


def grouped(notes: list[Note]) -> list[Note]:
    result: dict[int, tuple[set[int], float]] = {}
    for note in notes:
        key = round(note.time_ms)
        lanes, sustain = result.setdefault(key, (set(), 0.0))
        lanes.update(note.lanes)
        result[key] = (lanes, max(sustain, note.sustain_ms))
    return [Note(float(time), frozenset(lanes), sustain) for time, (lanes, sustain) in sorted(result.items())]


def read_chart(path: Path, difficulty: str) -> list[Note]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"\[{re.escape(difficulty)}Single\]\s*\{{(.*?)\n\}}", text, re.S | re.I)
    if not match:
        raise ValueError(f"Difficulty [{difficulty}Single] not found in {path}")
    notes = []
    for line in match.group(1).splitlines():
        event = re.match(r"\s*(\d+)\s*=\s*(\d+)\s+(\d+)", line)
        if event:
            lane, time_ms, sustain = map(int, event.groups())
            notes.append(Note(float(time_ms), frozenset({lane}), float(sustain)))
    return grouped(notes)


def midi_tempo_segments(mid: mido.MidiFile) -> list[tuple[int, int, float]]:
    events = []
    for message in mido.merge_tracks(mid.tracks):
        events.append(message)
    tick = 0
    tempo = 500000
    segments = [(0, tempo, 0.0)]
    for message in events:
        tick += message.time
        if message.type == "set_tempo":
            elapsed = segments[-1][2] + (tick - segments[-1][0]) * segments[-1][1] / 1_000_000 / mid.ticks_per_beat
            tempo = message.tempo
            segments.append((tick, tempo, elapsed))
    return segments


def tick_to_ms(tick: int, mid: mido.MidiFile, segments: list[tuple[int, int, float]]) -> float:
    segment = segments[0]
    for candidate in segments:
        if candidate[0] <= tick:
            segment = candidate
        else:
            break
    start_tick, tempo, start_seconds = segment
    return (start_seconds + (tick - start_tick) * tempo / 1_000_000 / mid.ticks_per_beat) * 1000


def read_midi(path: Path, difficulty: str) -> list[Note]:
    mid = mido.MidiFile(path)
    segments = midi_tempo_segments(mid)
    preferred = []
    for index, track in enumerate(mid.tracks):
        name = " ".join(str(message.name) for message in track if message.type == "track_name")
        if "guitar" in name.lower() and "rhythm" not in name.lower():
            preferred.append((index, track))
    tracks = preferred or list(enumerate(mid.tracks))
    # Clone Hero guitar gems use 96-100 for Expert, 84-88 for Hard, etc.
    difficulty_base = {"Easy": 60, "Medium": 72, "Hard": 84, "Expert": 96}[difficulty]
    notes: list[Note] = []
    for _, track in tracks:
        tick = 0
        active: dict[int, tuple[int, int]] = {}
        for message in track:
            tick += message.time
            if message.type == "note_on" and message.velocity > 0 and difficulty_base <= message.note <= difficulty_base + 4:
                active[message.note] = (tick, message.note - difficulty_base)
            elif message.type in {"note_off", "note_on"} and (message.type == "note_off" or message.velocity == 0) and message.note in active:
                start_tick, lane = active.pop(message.note)
                start_ms = tick_to_ms(start_tick, mid, segments)
                end_ms = tick_to_ms(tick, mid, segments)
                notes.append(Note(start_ms, frozenset({lane}), max(0.0, end_ms - start_ms)))
    return grouped(notes)


def metrics(notes: list[Note]) -> dict[str, float]:
    duration = max((note.time_ms + note.sustain_ms for note in notes), default=0.0)
    return {
        "notes": len(notes),
        "chords": sum(len(note.lanes) > 1 for note in notes),
        "sustains": sum(note.sustain_ms > 80 for note in notes),
        "duration_s": duration / 1000,
        "notes_per_minute": len(notes) / (duration / 60_000) if duration else 0.0,
    }


def timing_match(reference: list[Note], generated: list[Note], tolerance_ms: float = 80.0) -> tuple[float, float]:
    reference_times = [note.time_ms for note in reference]
    generated_times = [note.time_ms for note in generated]
    matches = sum(any(abs(candidate - value) <= tolerance_ms for value in reference_times) for candidate in generated_times)
    reverse = sum(any(abs(candidate - value) <= tolerance_ms for value in generated_times) for candidate in reference_times)
    precision = matches / len(generated_times) if generated_times else 0.0
    recall = reverse / len(reference_times) if reference_times else 0.0
    return precision, recall


def detailed_match(reference: list[Note], generated: list[Note], tolerance_ms: float = 80.0) -> dict[str, float]:
    """Compare one-to-one note events, including lanes and sustain lengths."""
    used: set[int] = set()
    matched = 0
    exact_lanes = 0
    overlap_scores: list[float] = []
    sustain_errors: list[float] = []
    for ref in reference:
        candidates = [
            (abs(candidate.time_ms - ref.time_ms), index, candidate)
            for index, candidate in enumerate(generated)
            if index not in used and abs(candidate.time_ms - ref.time_ms) <= tolerance_ms
        ]
        if not candidates:
            continue
        _, index, candidate = min(candidates)
        used.add(index)
        matched += 1
        if candidate.lanes == ref.lanes:
            exact_lanes += 1
        union = len(candidate.lanes | ref.lanes)
        overlap_scores.append(len(candidate.lanes & ref.lanes) / union if union else 1.0)
        sustain_errors.append(abs(candidate.sustain_ms - ref.sustain_ms))
    return {
        "matched": matched,
        "missing": len(reference) - matched,
        "extra": len(generated) - len(used),
        "exact_lane_rate": exact_lanes / matched if matched else 0.0,
        "lane_overlap": sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0,
        "sustain_mae_ms": sum(sustain_errors) / len(sustain_errors) if sustain_errors else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a reference Clone Hero chart with a ChartForge chart")
    parser.add_argument("reference", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("--difficulty", default="Expert", choices=("Easy", "Medium", "Hard", "Expert"))
    parser.add_argument("--detailed", action="store_true", help="Also compare matched lanes and sustain lengths")
    parser.add_argument("--report", type=Path, help="Optional output text file")
    args = parser.parse_args()

    reference_midi = args.reference / "notes.mid"
    generated_chart = args.generated / "notes.chart"
    if not reference_midi.exists():
        raise SystemExit(f"Reference MIDI not found: {reference_midi}")
    if not generated_chart.exists():
        raise SystemExit(f"Generated chart not found: {generated_chart}")

    reference = read_midi(reference_midi, args.difficulty)
    generated = read_chart(generated_chart, args.difficulty)
    reference_stats = metrics(reference)
    generated_stats = metrics(generated)
    precision, recall = timing_match(reference, generated)
    lines = [
        f"ChartForge comparison — {args.difficulty}",
        f"Reference: {args.reference}",
        f"Generated: {args.generated}",
        "",
        "Metric                 Reference       ChartForge",
        f"Notes                  {reference_stats['notes']:>9.0f}       {generated_stats['notes']:>10.0f}",
        f"Chords                 {reference_stats['chords']:>9.0f}       {generated_stats['chords']:>10.0f}",
        f"Sustains               {reference_stats['sustains']:>9.0f}       {generated_stats['sustains']:>10.0f}",
        f"Duration (s)           {reference_stats['duration_s']:>9.1f}       {generated_stats['duration_s']:>10.1f}",
        f"Notes per minute       {reference_stats['notes_per_minute']:>9.1f}       {generated_stats['notes_per_minute']:>10.1f}",
        "",
        f"Timing precision (±80 ms): {precision * 100:.1f}%",
        f"Timing recall (±80 ms):    {recall * 100:.1f}%",
    ]
    report = "\n".join(lines)
    if args.detailed:
        detail = detailed_match(reference, generated)
        report += "\n\nDetailed note comparison (one-to-one, ±80 ms)\n"
        report += f"Matched note events:       {detail['matched']:.0f}\n"
        report += f"Missing reference events:  {detail['missing']:.0f}\n"
        report += f"Extra generated events:    {detail['extra']:.0f}\n"
        report += f"Exact lane-set match:      {detail['exact_lane_rate'] * 100:.1f}%\n"
        report += f"Average lane overlap:      {detail['lane_overlap'] * 100:.1f}%\n"
        report += f"Sustain error (MAE):       {detail['sustain_mae_ms']:.0f} ms"
    print(report)
    if args.report:
        args.report.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
