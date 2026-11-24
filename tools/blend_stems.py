"""Create a temporary guitar-focused blend for internal testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("guitar", type=Path)
    parser.add_argument("other", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--other-gain", type=float, default=0.15)
    args = parser.parse_args()
    guitar, sample_rate = sf.read(args.guitar, always_2d=True)
    other, other_rate = sf.read(args.other, always_2d=True)
    if sample_rate != other_rate:
        raise SystemExit("Stem sample rates do not match")
    frames = min(len(guitar), len(other))
    mix = guitar[:frames] + other[:frames] * args.other_gain
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix *= 0.98 / peak
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, mix, sample_rate, subtype="PCM_16")
    print(f"Created {args.output} with other gain {args.other_gain}")


if __name__ == "__main__":
    main()
