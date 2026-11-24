"""Playable five-fret arrangement helpers.

This is intentionally deterministic and lightweight.  Audio analysis proposes
events; this module turns those events into a readable guitar chart.
"""

from __future__ import annotations


PATTERNS = {
    "Easy": (0, 1, 2, 1),
    "Medium": (0, 1, 2, 3, 2, 1),
    "Hard": (0, 1, 2, 3, 4, 3, 2, 1),
    "Expert": (0, 1, 2, 3, 4, 2, 1, 3),
}

MAX_JUMP = {"Easy": 1, "Medium": 2, "Hard": 3, "Expert": 4}

# Standard-tuned guitar, low to high.  This is used only as a fretboard
# heuristic; Clone Hero lanes are not literal guitar strings.
GUITAR_OPEN_MIDI = (40, 45, 50, 55, 59, 64)


def infer_lane_hints(pitch_times: dict[float, float]) -> dict[float, int]:
    """Infer playable fretboard positions from detected MIDI pitches."""
    if not pitch_times:
        return {}
    previous_fret: float | None = None
    hints: dict[float, int] = {}
    for onset, pitch in sorted(pitch_times.items()):
        candidates: list[tuple[float, int]] = []
        for open_pitch in GUITAR_OPEN_MIDI:
            fret = round(pitch - open_pitch)
            if 0 <= fret <= 20:
                movement = abs(fret - previous_fret) if previous_fret is not None else 0
                comfort = abs(fret - 5)
                candidates.append((movement * 2.2 + comfort * 0.35, fret))
        if not candidates:
            continue
        _, fret = min(candidates)
        previous_fret = float(fret)
        # Map a local five-fret hand position to Clone Hero's five lanes.
        hints[onset] = max(0, min(4, (fret - 1) % 5 if fret else 0))
    return hints


def infer_riff_lane_hints(pitch_times: dict[float, float]) -> dict[float, int]:
    """Reuse a lane contour when the same short pitch movement repeats."""
    items = sorted(pitch_times.items())
    if len(items) < 8:
        return {}
    hints: dict[float, int] = {}
    for length in (4, 5):
        occurrences: dict[tuple[int, ...], list[int]] = {}
        for index in range(len(items) - length + 1):
            intervals = tuple(round(items[index + offset + 1][1] - items[index + offset][1]) for offset in range(length - 1))
            if intervals and all(abs(interval) <= 12 for interval in intervals):
                occurrences.setdefault(intervals, []).append(index)
        for intervals, starts in occurrences.items():
            if len(starts) < 2:
                continue
            for start in starts:
                lane = 2
                for offset in range(length):
                    onset = items[start + offset][0]
                    hints.setdefault(onset, lane)
                    if offset < len(intervals):
                        if intervals[offset] >= 2:
                            lane += 1
                        elif intervals[offset] <= -2:
                            lane -= 1
                        lane = max(0, min(4, lane))
    return hints


def choose_lane(
    difficulty: str,
    event_index: int,
    previous_lane: int | None,
    pitch: float | None = None,
    pitch_min: float | None = None,
    pitch_max: float | None = None,
    lane_hint: int | None = None,
) -> int:
    """Choose a lane with controlled movement and no unnecessary repeats."""
    pattern = PATTERNS[difficulty]
    desired = pattern[event_index % len(pattern)]
    maximum_jump = MAX_JUMP[difficulty]
    candidates = range(5) if previous_lane is None else [
        lane for lane in range(5) if abs(lane - previous_lane) <= maximum_jump
    ]
    pitch_lane = None
    if pitch is not None and pitch_min is not None and pitch_max is not None and pitch_max > pitch_min:
        pitch_lane = round((pitch - pitch_min) / (pitch_max - pitch_min) * 4)
    return min(
        candidates,
        key=lambda lane: (
            abs(lane - desired) * (0.35 if pitch_lane is not None else 1.0)
            + (abs(lane - pitch_lane) * 0.65 if pitch_lane is not None else 0)
            + (abs(lane - lane_hint) * 0.15 if lane_hint is not None else 0)
            + (0.35 if previous_lane is not None and lane == previous_lane else 0)
            + (0.15 if lane in {0, 4} and event_index % 5 == 0 else 0)
        ),
    )


def choose_chord_lane(difficulty: str, lane: int, event_index: int) -> int:
    """Return a nearby second lane that is comfortable on a five-fret guitar."""
    options = [candidate for candidate in range(5) if candidate != lane and abs(candidate - lane) <= 2]
    if not options:
        return (lane + 1) % 5
    preferred_direction = 1 if event_index % 2 == 0 else -1
    return min(
        options,
        key=lambda candidate: (
            0 if (candidate - lane) * preferred_direction > 0 else 1,
            abs(candidate - lane),
        ),
    )


def chord_allowed(difficulty: str, event_index: int) -> bool:
    """Avoid stacking every detected chord on the lower difficulties."""
    if difficulty == "Easy":
        return False
    if difficulty == "Medium":
        return event_index % 2 == 0
    if difficulty == "Hard":
        return event_index % 4 != 1
    return True
