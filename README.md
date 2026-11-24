# RoboRiff

RoboRiff is a local AI-assisted Clone Hero chart generator. It analyzes an
audio file, detects guitar events, and exports a portable song folder with a
`notes.chart`, `song.ini`, audio, and album art.

## Features

- Local Tkinter GUI with English labels
- Riley/GAPS guitar-specific transcription mode
- Basic Pitch fallback transcription
- Optional Demucs guitar separation
- Automatic BPM and duration detection
- Playability-aware five-lane arrangement
- Easy, Medium, Hard, and Expert output
- Local song metadata and album-art lookup
- No API key required

## Development setup

Install the base dependencies:

```powershell
py -3.13 -m pip install -r requirements.txt
```

For AI modes, install the optional dependencies:

```powershell
py -3.13 -m pip install -r requirements-ai.txt
```

Run the development build:

```powershell
py -3.13 app.py
```

The Riley/GAPS model downloads its weights on first use and caches them
locally. The first guitar analysis can take several minutes on CPU.

## Testing against a reference chart

Reference songs and generated output are intentionally excluded from Git. Keep
them in local `Voorbeeld/` and `output/` folders, then run:

```powershell
py -3.13 tools\compare_charts.py `
  ".\Voorbeeld\Song Reference" `
  ".\output\Generated Song" `
  --difficulty Expert `
  --detailed
```

## Privacy and copyright

Audio files are analyzed locally. Users are responsible for having the rights
to process and distribute their audio, charts, and artwork. Do not commit
copyrighted songs, stems, charts, or album art to the public repository.

The optional Riley/GAPS implementation is vendored under `vendor/` and retains
its upstream license notice in `vendor/TABGEN_LICENSE`. Its model weights are
downloaded separately and are not included in this repository.

## License

RoboRiff code is released under the MIT License. See `LICENSE`. Third-party
code, models, and assets retain their own licenses.
