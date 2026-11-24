# Changelog

## RoboRiff 0.1.0 — Initial public release

### Included

- Local English Tkinter interface
- Automatic BPM and duration detection
- Riley/GAPS guitar-specific transcription mode
- Basic Pitch fallback mode
- Optional Demucs guitar separation
- Automatic album-art lookup
- Local song database search
- Easy, Medium, Hard, and Expert chart generation
- Playability-aware five-lane arrangement
- Detailed reference-chart comparison tool
- GPU toggle with safe CPU fallback

### Notes

- The first Riley/GAPS run downloads model weights locally.
- CPU analysis can take several minutes for a full song.
- Generated charts are drafts and should be playtested before distribution.
- Users must have the necessary rights to process and distribute their audio,
  charts, and artwork.

### Windows asset

- `RoboRiff.exe` can be attached to the GitHub Release when built with
  `build_release.ps1`.
