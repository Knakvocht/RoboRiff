"""RoboRiff - local AI-assisted Clone Hero chart generator.

This build creates a structurally valid Clone Hero song folder from an audio file.
Audio analysis is intentionally isolated behind a small generator function so it can
be replaced by BPM/instrument detection in the next iteration.
"""

from __future__ import annotations

import math
import json
import os
import re
import sqlite3
import tempfile
import configparser
import threading
import numpy as np
from pathlib import Path

# Prevent Numba cache issues when librosa is bundled or loaded from site-packages.
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "RoboRiffNumbaCache"))
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from playability import choose_chord_lane, choose_lane, chord_allowed, infer_lane_hints, infer_riff_lane_hints
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import wave

try:
    from PIL import Image, ImageTk
except ImportError:  # The app still runs without preview support.
    Image = None
    ImageTk = None

try:
    import librosa
except ImportError:  # The prototype keeps its synthetic fallback if unavailable.
    librosa = None

try:
    from mutagen import File as MutagenFile
except ImportError:
    MutagenFile = None


APP_NAME = "RoboRiff"
VERSION = "0.1.0"
AI_NOTE_CONFIDENCE = 0.35
AI_ONSET_THRESHOLD = 0.5
AI_FRAME_THRESHOLD = 0.3
AI_MINIMUM_NOTE_LENGTH_MS = 90
AI_MAX_NOTES_PER_MINUTE = 260
AI_MAX_SUSTAIN_RATIO = 0.10
# A wide grouping window turns guitar overtones and fast adjacent notes into
# false chords. Real strums are usually much tighter than this, especially in
# a separated guitar stem.
AI_CHORD_GROUP_WINDOW_SECONDS = 0.055


def load_ai_settings() -> dict[str, float]:
    settings = {
        "note_confidence": AI_NOTE_CONFIDENCE,
        "onset_threshold": AI_ONSET_THRESHOLD,
        "frame_threshold": AI_FRAME_THRESHOLD,
        "minimum_note_length": AI_MINIMUM_NOTE_LENGTH_MS,
    }
    config_path = Path(__file__).with_name("ai_config.json")
    if config_path.exists():
        try:
            import json as _json
            loaded = _json.loads(config_path.read_text(encoding="utf-8"))
            settings.update({key: float(value) for key, value in loaded.items() if key in settings})
        except (OSError, ValueError, TypeError):
            pass
    return settings


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9 _-]+", "", value).strip()
    return value or "RoboRiff Song"


def wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / float(audio.getframerate())
    except (OSError, EOFError, ZeroDivisionError):
        return None


def audio_duration_seconds(path: Path) -> float | None:
    """Read duration quickly without running full audio analysis."""
    detected = wav_duration(path)
    if detected is not None:
        return detected
    if MutagenFile is not None:
        try:
            audio = MutagenFile(str(path))
            if audio and audio.info and audio.info.length:
                return float(audio.info.length)
        except Exception:
            pass
    return None


def read_audio_tags(path: Path) -> dict[str, str]:
    """Read common title/artist/album tags from an audio file."""
    if MutagenFile is None:
        return {}
    try:
        audio = MutagenFile(str(path), easy=True)
        tags = audio.tags if audio else None
        if not tags:
            return {}
        values = {}
        for key in ("title", "artist", "album"):
            value = tags.get(key)
            if value:
                values[key] = str(value[0] if isinstance(value, list) else value).strip()
        return values
    except Exception:
        return {}


def read_song_ini(path: Path) -> dict[str, str]:
    """Read Clone Hero metadata from a nearby song.ini when available."""
    ini_path = path.parent / "song.ini"
    if not ini_path.exists():
        return {}
    try:
        parser = configparser.ConfigParser()
        parser.read(ini_path, encoding="utf-8")
        section = parser["song"] if parser.has_section("song") else parser[parser.sections()[0]]
        return {key: section.get(key, "").strip() for key in ("name", "artist", "album") if section.get(key, "").strip()}
    except (configparser.Error, OSError, IndexError):
        return {}


def download_album_art(title: str, artist: str, target: Path, album: str = "") -> str | None:
    """Find and save artwork using Apple's public iTunes Search API."""
    query = quote_plus(f"{title} {artist} {album}".strip())
    request = Request(
        f"https://itunes.apple.com/search?term={query}&media=music&entity=song&limit=1",
        headers={"User-Agent": "RoboRiff/0.1"},
    )
    with urlopen(request, timeout=12) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results", [])
    if not results or not results[0].get("artworkUrl100"):
        return None
    artwork_url = results[0]["artworkUrl100"].replace("100x100", "600x600")
    with urlopen(Request(artwork_url, headers={"User-Agent": "RoboRiff/0.1"}), timeout=12) as response:
        target.write_bytes(response.read())
    return results[0].get("collectionName") or results[0].get("trackName")


def search_song_database(query: str, database: Path) -> list[dict[str, str]]:
    """Search iTunes and cache the choices locally for later reuse."""
    query = query.strip()
    if not query:
        return []
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS song_search (query TEXT PRIMARY KEY, results TEXT NOT NULL)")
        cached = connection.execute("SELECT results FROM song_search WHERE query = ?", (query.casefold(),)).fetchone()
        if cached:
            return json.loads(cached[0])
        request = Request(
            f"https://itunes.apple.com/search?term={quote_plus(query)}&media=music&entity=song&limit=12",
            headers={"User-Agent": "RoboRiff/0.1"},
        )
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        results = []
        for item in payload.get("results", []):
            if item.get("trackName") and item.get("artistName"):
                results.append({
                    "title": item["trackName"],
                    "artist": item["artistName"],
                    "album": item.get("collectionName", ""),
                    "artwork": item.get("artworkUrl100", ""),
                })
        connection.execute("INSERT OR REPLACE INTO song_search(query, results) VALUES (?, ?)", (query.casefold(), json.dumps(results)))
        return results
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / float(audio.getframerate())
    except (OSError, EOFError, ZeroDivisionError):
        return None


def make_notes(bpm: int, duration: float, difficulty: str) -> list[str]:
    """Create a deliberately simple, valid guitar part for prototype testing."""
    resolution = 192
    beat_ms = 60_000 / bpm
    step_ms = beat_ms / 2
    patterns = {
        "Easy": ([0, 1, 2, 1], 1),
        "Medium": ([0, 1, 2, 3, 2, 1], 1),
        "Hard": ([0, 1, 2, 3, 4, 3, 2, 1], 1),
        "Expert": ([0, 1, 2, 3, 4, 2, 1, 3], 1),
    }
    pattern, _ = patterns[difficulty]
    end_ms = max(2_000, int(duration * 1_000))
    notes: list[str] = []
    index = 0
    for time_ms in range(0, end_ms, int(step_ms)):
        lane = pattern[index % len(pattern)]
        # Every fourth hit is a chord in Hard/Expert, keeping the chart readable.
        if difficulty in {"Hard", "Expert"} and index % 16 == 8:
            lanes = {lane, min(4, lane + 2)}
            for fret in sorted(lanes):
                notes.append(f"  {fret} = {time_ms} 0")
        else:
            notes.append(f"  {lane} = {time_ms} 0")
        index += 1
    return notes


def analyze_audio(path: Path) -> tuple[float, float, list[float], set[float], dict[float, int]]:
    """Return detected BPM, duration and musical onset times in seconds."""
    if librosa is None:
        raise RuntimeError("librosa is not installed")
    audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
    duration = float(librosa.get_duration(y=audio, sr=sample_rate))
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
    bpm = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=4096, hop_length=512)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        backtrack=False,
        delta=0.04,
        wait=1,
    )
    onset_times = [float(value) for value in librosa.frames_to_time(onset_frames, sr=sample_rate)]
    # Add musically plausible half-beat candidates where the onset detector is quiet.
    # This improves recall on compressed or heavily mixed audio without filling silence.
    beat_times = [float(value) for value in librosa.frames_to_time(beat_frames, sr=sample_rate)]
    envelope_floor = float(np.percentile(onset_envelope, 28))
    candidates = set(onset_times)
    for index, beat_time in enumerate(beat_times):
        candidates.add(beat_time)
        if index + 1 < len(beat_times):
            halfway = beat_time + (beat_times[index + 1] - beat_time) / 2
            frame = min(len(onset_envelope) - 1, max(0, int(round(halfway * sample_rate / 512))))
            if onset_envelope[frame] >= envelope_floor:
                candidates.add(halfway)
    onset_times = sorted(candidates)
    chord_times: set[float] = set()
    sustain_times: dict[float, int] = {}
    noise_floor = float(np.percentile(rms, 20))
    for index, onset in enumerate(onset_times):
        frame = min(chroma.shape[1] - 1, max(0, int(round(onset * sample_rate / 512))))
        profile = chroma[:, frame]
        peak = float(np.max(profile))
        active_pitches = int(np.sum(profile >= max(0.32, peak * 0.45))) if peak else 0
        if active_pitches >= 2:
            chord_times.add(round(onset, 3))
        energy = float(rms[min(len(rms) - 1, frame)])
        threshold = max(noise_floor * 1.7, energy * 0.42)
        end_frame = frame
        while end_frame + 1 < len(rms) and rms[end_frame + 1] >= threshold:
            end_frame += 1
        sustain_ms = int((end_frame - frame) * 512 / sample_rate * 1000)
        next_onset_ms = (onset_times[index + 1] - onset) * 1000 if index + 1 < len(onset_times) else 0
        available_ms = max(0.0, next_onset_ms - 90.0)
        if available_ms >= 500 and sustain_ms >= 280:
            sustain_times[round(onset, 3)] = int(min(sustain_ms, available_ms, 1200))
    return max(40.0, min(300.0, bpm)), duration, onset_times, chord_times, sustain_times


def _load_basic_pitch_model(use_gpu: bool):
    """Load Basic Pitch, preferring DirectML when the user requests GPU mode."""
    from basic_pitch import inference as basic_pitch

    model = basic_pitch.Model(basic_pitch.ICASSP_2022_MODEL_PATH)
    if not use_gpu or model.model_type.name != "ONNX":
        return model, "CPU"
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        gpu_provider = next((name for name in ("DmlExecutionProvider", "CUDAExecutionProvider") if name in available), None)
        if not gpu_provider:
            return model, "CPU fallback (GPU provider unavailable)"
        model.model = ort.InferenceSession(
            str(basic_pitch.ICASSP_2022_MODEL_PATH),
            providers=[gpu_provider, "CPUExecutionProvider"],
        )
        return model, gpu_provider.replace("ExecutionProvider", "")
    except Exception:
        return model, "CPU fallback (GPU initialization failed)"


def analyze_with_basic_pitch(path: Path, use_gpu: bool = False) -> tuple[float, float, list[float], set[float], dict[float, int], dict[float, float]]:
    """Use Spotify Basic Pitch for optional local polyphonic transcription."""
    from basic_pitch.inference import predict

    bpm, duration, classic_onsets, classic_chords, classic_sustains = analyze_audio(path)
    settings = load_ai_settings()
    model, _backend = _load_basic_pitch_model(use_gpu)
    _, _, predicted = predict(
        path,
        model_or_model_path=model,
        onset_threshold=settings["onset_threshold"],
        frame_threshold=settings["frame_threshold"],
        minimum_note_length=settings["minimum_note_length"],
        midi_tempo=bpm,
    )
    events = []
    for event in predicted:
        start, end, pitch, confidence = float(event[0]), float(event[1]), int(event[2]), float(event[3])
        if confidence >= settings["note_confidence"]:
            events.append((start, end, pitch, confidence))
    events.sort(key=lambda event: event[0])
    groups: list[list[tuple[float, float, int, float]]] = []
    for event in events:
        if not groups or event[0] - groups[-1][0][0] > AI_CHORD_GROUP_WINDOW_SECONDS:
            groups.append([event])
        else:
            groups[-1].append(event)
    # Basic Pitch can miss a clean rhythmic attack even when the audio onset
    # detector sees it. Add only clearly separated classical onsets as low-
    # confidence candidates; the density cap below still removes excess noise.
    ai_onset_times = [group[0][0] for group in groups]
    for onset in classic_onsets:
        if onset < 0.25 or any(abs(onset - candidate) < 0.075 for candidate in ai_onset_times):
            continue
        groups.append([(onset, onset + 0.08, 60, 0.18)])
    groups.sort(key=lambda group: float(group[0][0]))
    max_events = max(1, int(duration / 60 * AI_MAX_NOTES_PER_MINUTE))
    if len(groups) > max_events:
        groups = sorted(groups, key=lambda group: sum(float(event[3]) for event in group) / len(group), reverse=True)[:max_events]
        groups.sort(key=lambda group: float(group[0][0]))
    onsets = [group[0][0] for group in groups]
    chord_groups = [
        group for group in groups
        if len({event[2] for event in group}) >= 2
    ]
    # Polyphonic transcription can still over-call chords on distorted or
    # reverberant guitar. Keep the strongest chord attacks, while ensuring a
    # chart does not become almost entirely two-note chords.
    max_chords = max(1, int(len(groups) * 0.45))
    if len(chord_groups) > max_chords:
        chord_groups = sorted(
            chord_groups,
            key=lambda group: sum(float(event[3]) for event in group) / len(group),
            reverse=True,
        )[:max_chords]
        chord_groups.sort(key=lambda group: float(group[0][0]))
    chord_times = {round(group[0][0], 3) for group in chord_groups}
    # Do not merge classical chroma chord guesses into the AI result. On an
    # isolated guitar stem, chroma often interprets harmonics or overlapping
    # notes as chords and can turn most of the chart into two-note chords.
    # Basic Pitch's tightly grouped pitch events are the safer chord signal.
    sustain_groups = [
        group for group in groups
        if 0.35 <= max(event[1] for event in group) - group[0][0] <= 1.20
    ]
    # Keep sustains readable rather than attaching one to every detected
    # overlap. This preserves real held guitar notes without flooding the chart.
    max_sustains = max(1, int(len(groups) * AI_MAX_SUSTAIN_RATIO))
    if len(sustain_groups) > max_sustains:
        sustain_groups = sorted(
            sustain_groups,
            key=lambda group: (
                max(event[1] for event in group) - group[0][0],
                sum(float(event[3]) for event in group) / len(group),
            ),
            reverse=True,
        )[:max_sustains]
        sustain_groups.sort(key=lambda group: float(group[0][0]))
    sustain_times = {
        round(group[0][0], 3): int(min(1200, (max(event[1] for event in group) - group[0][0]) * 1000))
        for group in sustain_groups
    }
    pitch_times = {
        round(group[0][0], 3): sum(float(event[2]) for event in group) / len(group)
        for group in groups
    }
    return bpm, duration, onsets, chord_times, sustain_times, pitch_times


def analyze_with_riley(path: Path, use_gpu: bool = False) -> tuple[float, float, list[float], set[float], dict[float, int], dict[float, float]]:
    """Use the optional guitar-specific Riley/GAPS transcriber."""
    from guitar_ai import predict_guitar

    bpm, duration, classic_onsets, _, classic_sustains = analyze_audio(path)
    predicted = predict_guitar(path, use_gpu=use_gpu)
    events = [
        (float(event[0]), float(event[1]), int(event[2]), float(event[3]))
        for event in predicted
        if float(event[0]) >= 0.25
    ]
    events.sort(key=lambda event: event[0])
    groups: list[list[tuple[float, float, int, float]]] = []
    for event in events:
        if not groups or event[0] - groups[-1][0][0] > AI_CHORD_GROUP_WINDOW_SECONDS:
            groups.append([event])
        else:
            groups[-1].append(event)
    ai_onsets = [group[0][0] for group in groups]
    for onset in classic_onsets:
        if onset >= 0.25 and not any(abs(onset - candidate) < 0.075 for candidate in ai_onsets):
            groups.append([(onset, onset + 0.08, 60, 0.18)])
    groups.sort(key=lambda group: float(group[0][0]))
    max_events = max(1, int(duration / 60 * AI_MAX_NOTES_PER_MINUTE))
    if len(groups) > max_events:
        groups = sorted(groups, key=lambda group: sum(float(event[3]) for event in group) / len(group), reverse=True)[:max_events]
        groups.sort(key=lambda group: float(group[0][0]))
    onsets = [group[0][0] for group in groups]
    chord_groups = [group for group in groups if len({event[2] for event in group}) >= 2]
    max_chords = max(1, int(len(groups) * 0.45))
    if len(chord_groups) > max_chords:
        chord_groups = sorted(chord_groups, key=lambda group: sum(float(event[3]) for event in group) / len(group), reverse=True)[:max_chords]
    chord_times = {round(group[0][0], 3) for group in chord_groups}
    sustain_groups = [group for group in groups if 0.35 <= max(event[1] for event in group) - group[0][0] <= 1.20]
    max_sustains = max(1, int(len(groups) * AI_MAX_SUSTAIN_RATIO))
    if len(sustain_groups) > max_sustains:
        sustain_groups = sorted(sustain_groups, key=lambda group: max(event[1] for event in group) - group[0][0], reverse=True)[:max_sustains]
    sustain_times = {round(group[0][0], 3): int(min(1200, (max(event[1] for event in group) - group[0][0]) * 1000)) for group in sustain_groups}
    pitch_times = {round(group[0][0], 3): sum(float(event[2]) for event in group) / len(group) for group in groups}
    return bpm, duration, onsets, chord_times, sustain_times, pitch_times


def choose_analysis_source(audio_path: Path) -> tuple[Path, str]:
    """Use an adjacent isolated guitar stem when available; otherwise use the mix."""
    for stem_name in ("guitar.ogg", "guitar.wav", "guitar.mp3", "guitar.flac"):
        stem = audio_path.parent / stem_name
        if stem.exists():
            return stem, "isolated guitar stem"
    return audio_path, "full mix fallback"


def choose_full_song_source(audio_path: Path) -> Path:
    """Find the full-song file when the user selected an isolated stem."""
    if audio_path.stem.casefold() not in {"guitar", "rhythm", "vocals", "drums", "keys"}:
        return audio_path
    for song_name in ("song.ogg", "song.wav", "song.mp3", "song.flac"):
        candidate = audio_path.parent / song_name
        if candidate.exists():
            return candidate
    return audio_path


class ModernDropdown(tk.Frame):
    """Small custom dropdown with a consistent dark RoboRiff appearance."""

    def __init__(self, parent, variable: tk.StringVar, values: tuple[str, ...], width: int = 18):
        super().__init__(parent, bg="#252c38", highlightthickness=1, highlightbackground="#536174", highlightcolor="#36c98f")
        self.variable = variable
        self.values = values
        self.popup = None


class ModernToggle(tk.Frame):
    """Compact on/off control that matches the RoboRiff dark UI."""

    def __init__(self, parent, variable: tk.BooleanVar, text: str):
        super().__init__(parent, bg="#191e28")
        self.variable = variable
        self.text = text
        self.button = tk.Button(
            self,
            command=self._toggle,
            relief="flat",
            bd=0,
            highlightthickness=1,
            padx=10,
            pady=5,
            anchor="w",
            font=("Segoe UI", 9),
        )
        self.button.pack(fill="x")
        self.variable.trace_add("write", lambda *_: self._refresh())
        self._refresh()

    def _toggle(self) -> None:
        self.variable.set(not self.variable.get())

    def _refresh(self) -> None:
        enabled = bool(self.variable.get())
        self.button.configure(
            text=f"{self.text}  {'ON' if enabled else 'OFF'}",
            bg="#36c98f" if enabled else "#252c38",
            fg="#08120e" if enabled else "#e8edf5",
            activebackground="#56e0aa" if enabled else "#2d3644",
            activeforeground="#08120e" if enabled else "#ffffff",
            highlightbackground="#36c98f" if enabled else "#536174",
            highlightcolor="#36c98f",
        )
        self.button = tk.Button(
            self,
            text=self._label(),
            command=self._open,
            bg="#252c38",
            fg="#f5f7fb",
            activebackground="#2d3644",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            anchor="w",
            padx=9,
            font=("Segoe UI", 10),
            width=width,
        )
        self.button.pack(fill="both", expand=True)
        self.variable.trace_add("write", lambda *_: self.button.configure(text=self._label()))
        self.menu = tk.Menu(
            self,
            tearoff=False,
            bg="#252c38",
            fg="#f5f7fb",
            activebackground="#36c98f",
            activeforeground="#08120e",
            bd=0,
            relief="flat",
            font=("Segoe UI", 10),
        )
        for value in values:
            self.menu.add_command(label=value, command=lambda item=value: self._select(item))

    def _label(self) -> str:
        return f"{self.variable.get()}  ▼"

    def _select(self, value: str) -> None:
        self.variable.set(value)
        self._close_popup()

    def _open(self) -> None:
        if self.popup is not None and self.popup.winfo_exists():
            self._close_popup()
            return
        root = self.winfo_toplevel()
        root.update_idletasks()
        popup = tk.Frame(root, bg="#202833", highlightthickness=1, highlightbackground="#3a4657")
        self.popup = popup
        popup_width = max(self.winfo_width(), 220)
        popup_height = len(self.values) * 42 + 18
        relative_x = self.winfo_rootx() - root.winfo_rootx()
        relative_below_y = self.winfo_rooty() + self.winfo_height() + 4 - root.winfo_rooty()
        relative_above_y = self.winfo_rooty() - popup_height - 4 - root.winfo_rooty()
        y = relative_below_y if relative_below_y + popup_height <= root.winfo_height() - 8 else max(8, relative_above_y)
        x = min(max(8, relative_x), max(8, root.winfo_width() - popup_width - 8))
        popup.place(x=x, y=y, width=popup_width, height=popup_height)
        body = tk.Frame(popup, bg="#202833")
        body.pack(fill="both", expand=True, padx=8, pady=8)
        for value in self.values:
            selected = value == self.variable.get()
            row = tk.Button(
                body,
                text=("●  " if selected else "   ") + value,
                command=lambda item=value: self._select(item),
                bg="#36c98f" if selected else "#202833",
                fg="#08120e" if selected else "#e8edf5",
                activebackground="#36c98f",
                activeforeground="#08120e",
                relief="flat",
                bd=0,
                highlightthickness=0,
                anchor="w",
                padx=10,
                pady=8,
                font=("Segoe UI", 10),
            )
            row.pack(fill="x", pady=1)
        popup.bind("<Escape>", lambda _event: self._close_popup())
        popup.focus_force()

    def _close_popup(self) -> None:
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None


class ModernDropdownV2(tk.Frame):
    """Attached dark dropdown rendered inside the main RoboRiff window."""

    def __init__(self, parent, variable: tk.StringVar, values: tuple[str, ...], width: int = 18):
        super().__init__(parent, bg="#252c38", highlightthickness=1, highlightbackground="#536174")
        self.variable = variable
        self.values = values
        self.popup = None
        self.button = tk.Button(self, text=self._label(), command=self._open, bg="#252c38", fg="#f5f7fb", activebackground="#2d3644", activeforeground="#ffffff", relief="flat", bd=0, highlightthickness=0, anchor="w", padx=9, font=("Segoe UI", 10), width=width)
        self.button.pack(fill="both", expand=True)
        self.variable.trace_add("write", lambda *_: self.button.configure(text=self._label()))

    def _label(self) -> str:
        return f"{self.variable.get()}  v"

    def _open(self) -> None:
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
            self.popup = None
            return
        root = self.winfo_toplevel()
        root.update_idletasks()
        height = len(self.values) * 40 + 16
        popup = tk.Frame(root, bg="#202833", highlightthickness=1, highlightbackground="#3a4657")
        self.popup = popup
        x = self.winfo_rootx() - root.winfo_rootx()
        below = self.winfo_rooty() + self.winfo_height() + 4 - root.winfo_rooty()
        above = self.winfo_rooty() - height - 4 - root.winfo_rooty()
        y = below if below + height <= root.winfo_height() - 8 else max(8, above)
        width = max(self.winfo_width(), 220)
        x = min(max(8, x), max(8, root.winfo_width() - width - 8))
        popup.place(x=x, y=y, width=width, height=height)
        body = tk.Frame(popup, bg="#202833")
        body.pack(fill="both", expand=True, padx=7, pady=7)
        for value in self.values:
            selected = value == self.variable.get()
            button = tk.Button(body, text=("o  " if selected else "   ") + value, command=lambda item=value: self._select(item), bg="#36c98f" if selected else "#202833", fg="#08120e" if selected else "#e8edf5", activebackground="#36c98f", activeforeground="#08120e", relief="flat", bd=0, highlightthickness=0, anchor="w", padx=10, pady=7, font=("Segoe UI", 10))
            button.pack(fill="x", pady=1)

    def _select(self, value: str) -> None:
        self.variable.set(value)
        if self.popup is not None:
            self.popup.destroy()
            self.popup = None


class ModernToggleV2(tk.Frame):
    def __init__(self, parent, variable: tk.BooleanVar, text: str):
        super().__init__(parent, bg="#191e28")
        self.variable = variable
        self.text = text
        self.button = tk.Button(self, command=self._toggle, relief="flat", bd=0, highlightthickness=1, padx=10, pady=5, anchor="w", font=("Segoe UI", 9))
        self.button.pack(fill="x")
        self.variable.trace_add("write", lambda *_: self._refresh())
        self._refresh()

    def _toggle(self) -> None:
        self.variable.set(not self.variable.get())

    def _refresh(self) -> None:
        on = bool(self.variable.get())
        self.button.configure(text=f"{self.text}  {'ON' if on else 'OFF'}", bg="#36c98f" if on else "#252c38", fg="#08120e" if on else "#e8edf5", activebackground="#56e0aa" if on else "#2d3644", activeforeground="#08120e" if on else "#ffffff", highlightbackground="#36c98f" if on else "#536174")


def make_notes_from_onsets(onsets: list[float], bpm: int, difficulty: str, chord_times: set[float] | None = None, sustain_times: dict[float, int] | None = None, pitch_times: dict[float, float] | None = None) -> list[str]:
    """Quantize detected onsets into a readable five-fret guitar pattern."""
    step_ms = (60_000 / bpm) / 4
    density_skip = {"Easy": 3, "Medium": 2, "Hard": 5, "Expert": 1}[difficulty]
    notes: list[str] = []
    previous_time = -10_000.0
    previous_lane: int | None = None
    event_index = 0
    chart_pitches = list(pitch_times.values()) if pitch_times else []
    pitch_min = min(chart_pitches) if chart_pitches else None
    pitch_max = max(chart_pitches) if chart_pitches else None
    lane_hints = infer_lane_hints(pitch_times or {})
    lane_hints.update(infer_riff_lane_hints(pitch_times or {}))
    for index, onset in enumerate(onsets):
        if density_skip > 1 and index % density_skip == 1:
            continue
        time_ms = round((onset * 1_000) / step_ms) * step_ms
        if time_ms < 250 or time_ms - previous_time < max(75, step_ms * 0.45):
            continue
        pitch = next((value for key, value in (pitch_times or {}).items() if abs(onset - key) < 0.08), None)
        lane_hint = next((value for key, value in lane_hints.items() if abs(onset - key) < 0.08), None)
        lane = choose_lane(difficulty, event_index, previous_lane, pitch, pitch_min, pitch_max, lane_hint)
        sustain = 0
        if sustain_times:
            matching_sustain = next((value for key, value in sustain_times.items() if abs(onset - key) < 0.08), 0)
            sustain = int(round(matching_sustain / step_ms) * step_ms)
        notes.append(f"  {lane} = {int(time_ms)} {sustain}")
        is_chord = chord_times and any(abs(onset - chord_time) < 0.08 for chord_time in chord_times)
        if difficulty in {"Medium", "Hard", "Expert"} and is_chord and chord_allowed(difficulty, event_index):
            chord_lane = choose_chord_lane(difficulty, lane, event_index)
            notes.append(f"  {chord_lane} = {int(time_ms)} {sustain}")
        previous_time = time_ms
        previous_lane = lane
        event_index += 1
    return notes


def build_chart(title: str, artist: str, charter: str, bpm: int, duration: float, onsets: list[float] | None = None, chord_times: set[float] | None = None, sustain_times: dict[float, int] | None = None, pitch_times: dict[float, float] | None = None) -> str:
    sections = ["[Song]", "{", f'  Name = "{title}"', f'  Artist = "{artist}"',
                f'  Charter = "{charter}"', "  Album = \"\"", "  Year = \"2026\"",
                "  Genre = \"Other\"", "  MediaType = \"song\"", "  Resolution = 192",
                "  Offset = 0", "  PreviewStart = 0", "  PreviewEnd = 0", "}", ""]
    bpm_value = int(round(bpm * 1000))
    sections += ["[SyncTrack]", "{", f"  0 = B {bpm_value}", "}", ""]
    sections += ["[Events]", "{", '  0 = E "section Intro"', "}", ""]
    for difficulty in ("Easy", "Medium", "Hard", "Expert"):
        notes = make_notes_from_onsets(onsets, bpm, difficulty, chord_times, sustain_times, pitch_times) if onsets else make_notes(bpm, duration, difficulty)
        sections += [f"[{difficulty}Single]", "{", *notes, "}", ""]
    return "\n".join(sections)


class RoboRiff(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1120x820")
        self.minsize(980, 720)
        self.configure(bg="#10131a")
        self.audio_path = tk.StringVar()
        self.title_value = tk.StringVar(value="My Chart")
        self.artist = tk.StringVar(value="Unknown Artist")
        self.album = tk.StringVar()
        self.charter = tk.StringVar(value="RoboRiff")
        self.bpm = tk.StringVar(value="Auto")
        self.duration = tk.StringVar(value="Auto")
        self.difficulty = tk.StringVar(value="Expert")
        self.instrument = tk.StringVar(value="Lead Guitar")
        self.analysis_mode = tk.StringVar(value="Basic Pitch — balanced")
        self.use_gpu = tk.BooleanVar(value=False)
        self.output = tk.StringVar(value=str(Path.cwd() / "output"))
        self.art_status = tk.StringVar(value="No artwork selected")
        self.search_query = tk.StringVar()
        self.search_results: list[dict[str, str]] = []
        self.database_path = Path(__file__).with_name("roboriff.db")
        self.preview_art_path: Path | None = None
        self.search_after_id: str | None = None
        self.art_canvas: tk.Canvas | None = None
        self.art_image = None
        self._style()
        self._layout()

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#10131a")
        style.configure("Card.TFrame", background="#191e28")
        style.configure("TLabel", background="#10131a", foreground="#e8edf5", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#10131a", foreground="#ffffff", font=("Segoe UI", 24, "bold"))
        style.configure("Muted.TLabel", background="#10131a", foreground="#8e9aaa", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#191e28", foreground="#e8edf5", font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground="#252c38", foreground="#ffffff", insertcolor="#ffffff")
        style.configure(
            "TCombobox",
            fieldbackground="#252c38",
            background="#252c38",
            foreground="#ffffff",
            selectbackground="#36c98f",
            selectforeground="#08120e",
            arrowcolor="#ffffff",
        )
        style.configure(
            "Modern.TCombobox",
            fieldbackground="#252c38",
            background="#252c38",
            foreground="#ffffff",
            selectbackground="#36c98f",
            selectforeground="#08120e",
            arrowcolor="#d9e2ee",
            padding=(9, 6),
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#252c38"), ("disabled", "#202631")],
            foreground=[("readonly", "#ffffff"), ("disabled", "#8e9aaa")],
            selectbackground=[("readonly", "#36c98f")],
            selectforeground=[("readonly", "#08120e")],
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", "#252c38"), ("active", "#2d3644")],
            foreground=[("readonly", "#ffffff")],
            selectbackground=[("readonly", "#36c98f")],
            selectforeground=[("readonly", "#08120e")],
        )
        style.configure("Card.TCheckbutton", background="#191e28", foreground="#e8edf5", font=("Segoe UI", 9))
        style.map("Card.TCheckbutton", background=[("active", "#191e28")], foreground=[("disabled", "#8e9aaa")])
        style.configure("Accent.TButton", background="#36c98f", foreground="#08120e", padding=10, font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#56e0aa")])
        style.configure("TButton", background="#2b3442", foreground="#f5f7fb", padding=(12, 8), relief="flat", borderwidth=0, font=("Segoe UI", 9))
        style.map("TButton", background=[("active", "#3a4657"), ("pressed", "#202833")], foreground=[("disabled", "#8e9aaa")])
        style.configure("Treeview", background="#252c38", fieldbackground="#252c38", foreground="#e8edf5", rowheight=28, borderwidth=0, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background="#303846", foreground="#e8edf5", relief="flat", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#36c98f")], foreground=[("selected", "#08120e")])

    def _layout(self) -> None:
        root = ttk.Frame(self, padding=26)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 22))
        ttk.Label(header, text="ROBORIFF", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text=f"  AI CHART GENERATOR  /  v{VERSION}", style="Muted.TLabel").pack(side="left", pady=(9, 0))

        columns = ttk.Frame(root)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns, style="Card.TFrame", padding=20)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = ttk.Frame(columns, style="Card.TFrame", padding=20, width=290)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ttk.Label(left, text="CREATE A CHART", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(left, text="Start with a song file and tune the chart settings.", style="Card.TLabel").pack(anchor="w", pady=(3, 18))
        audio = ttk.Frame(left, style="Card.TFrame")
        audio.pack(fill="x", pady=(0, 16))
        ttk.Label(audio, text="AUDIO FILE", style="Card.TLabel").grid(row=0, column=0, sticky="w", columnspan=2)
        ttk.Entry(audio, textvariable=self.audio_path).grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(audio, text="Choose file…", command=self.choose_audio).grid(row=1, column=1, padx=(10, 0), pady=(12, 0))
        audio.columnconfigure(0, weight=1)
        ttk.Label(audio, text="OR SEARCH THE SONG DATABASE", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(18, 0), columnspan=2)
        search_row = ttk.Frame(audio, style="Card.TFrame")
        search_row.grid(row=3, column=0, sticky="ew", columnspan=2, pady=(6, 0))
        search_entry = ttk.Entry(search_row, textvariable=self.search_query)
        search_entry.pack(side="left", fill="x", expand=True)
        search_entry.bind("<Return>", lambda _event: self.search_songs())
        search_entry.bind("<KeyRelease>", self.queue_search)
        ttk.Button(search_row, text="Search", command=self.search_songs).pack(side="left", padx=(8, 0))
        results_row = ttk.Frame(audio, style="Card.TFrame")
        results_row.grid(row=4, column=0, sticky="ew", columnspan=2, pady=(9, 0))
        self.search_tree = ttk.Treeview(results_row, columns=("title", "artist", "album"), show="headings", height=3, selectmode="browse")
        self.search_tree.heading("title", text="TITLE")
        self.search_tree.heading("artist", text="ARTIST")
        self.search_tree.heading("album", text="ALBUM")
        self.search_tree.column("title", width=190, minwidth=130, anchor="w")
        self.search_tree.column("artist", width=140, minwidth=100, anchor="w")
        self.search_tree.column("album", width=190, minwidth=120, anchor="w")
        self.search_tree.pack(fill="both", expand=True)
        self.search_tree.bind("<Double-Button-1>", lambda _event: self.use_selected_song())
        action_row = ttk.Frame(audio, style="Card.TFrame")
        action_row.grid(row=5, column=0, sticky="e", columnspan=2, pady=(7, 0))
        ttk.Button(action_row, text="Use selected song", command=self.use_selected_song).pack(anchor="e")

        form = ttk.Frame(left, style="Card.TFrame")
        form.pack(fill="x")
        fields = [("Title", self.title_value), ("Artist", self.artist), ("Charter", self.charter)]
        for col, (label, variable) in enumerate(fields):
            ttk.Label(form, text=label, style="Card.TLabel").grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else 12, 0))
            ttk.Entry(form, textvariable=variable).grid(row=1, column=col, sticky="ew", padx=(0 if col == 0 else 12, 0), pady=(5, 0))
            form.columnconfigure(col, weight=1)
        ttk.Label(form, text="BPM (detected)", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(14, 0))
        tk.Entry(form, textvariable=self.bpm, state="readonly", readonlybackground="#252c38", fg="#f5f7fb", relief="flat", bd=1, highlightthickness=1, highlightbackground="#536174", highlightcolor="#36c98f", font=("Segoe UI", 10), width=12).grid(row=3, column=0, sticky="ew", pady=(5, 0))
        ttk.Label(form, text="Duration (detected)", style="Card.TLabel").grid(row=2, column=1, sticky="w", pady=(14, 0))
        tk.Entry(form, textvariable=self.duration, state="readonly", readonlybackground="#252c38", fg="#f5f7fb", relief="flat", bd=1, highlightthickness=1, highlightbackground="#536174", highlightcolor="#36c98f", font=("Segoe UI", 10), width=12).grid(row=3, column=1, sticky="ew", pady=(5, 0))
        ttk.Label(form, text="Instrument", style="Card.TLabel").grid(row=2, column=2, sticky="w", pady=(14, 0))
        ModernDropdownV2(form, self.instrument, ("Lead Guitar", "Bass"), width=18).grid(row=3, column=2, sticky="ew", pady=(5, 0))
        ttk.Label(form, text="Difficulty preview", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=(14, 0))
        ModernDropdownV2(form, self.difficulty, ("Easy", "Medium", "Hard", "Expert"), width=18).grid(row=5, column=0, sticky="ew", pady=(5, 0))
        ttk.Label(form, text="Output folder", style="Card.TLabel").grid(row=4, column=1, sticky="w", pady=(14, 0))
        ttk.Entry(form, textvariable=self.output).grid(row=5, column=1, sticky="ew", pady=(5, 0))
        ttk.Label(form, text="Analysis mode", style="Card.TLabel").grid(row=4, column=2, sticky="w", pady=(14, 0))
        ModernDropdownV2(form, self.analysis_mode, ("Riley Guitar — best quality", "Basic Pitch — balanced", "Guitar Separation — slow", "Classic — fast"), width=22).grid(row=5, column=2, sticky="ew", pady=(5, 0))
        ModernToggleV2(form, self.use_gpu, "GPU acceleration").grid(row=6, column=2, sticky="ew", pady=(12, 0))

        ttk.Label(right, text="ALBUM ART", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.art_canvas = tk.Canvas(right, width=205, height=205, bg="#252c38", highlightthickness=0)
        self.art_canvas.pack(pady=(16, 12))
        self._show_art_placeholder()
        ttk.Button(right, text="Find album art", command=self.fetch_art).pack(fill="x")
        ttk.Label(right, textvariable=self.art_status, style="Card.TLabel", wraplength=205).pack(anchor="w", pady=(10, 22))
        ttk.Separator(right).pack(fill="x", pady=(0, 18))
        ttk.Label(right, text="EXPORT", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(right, text="Portable song folder with notes.chart, song.ini and album art.", style="Card.TLabel", wraplength=205).pack(anchor="w", pady=(8, 12))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(right, variable=self.progress_var, maximum=100, mode="indeterminate")
        self.generate_button = ttk.Button(right, text="Generate chart", style="Accent.TButton", command=self.generate)
        self.generate_button.pack(fill="x")
        self.status = tk.StringVar(value="Ready — choose an audio file or generate a test chart.")
        ttk.Label(left, textvariable=self.status, style="Muted.TLabel", wraplength=500).pack(anchor="w", pady=(22, 0))

    def choose_audio(self) -> None:
        selected = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg"), ("All files", "*.*")])
        if selected:
            self.audio_path.set(selected)
            path = Path(selected)
            tags = read_audio_tags(path)
            ini_tags = read_song_ini(path)
            if ini_tags.get("name"):
                tags["title"] = ini_tags["name"]
            if ini_tags.get("artist"):
                tags["artist"] = ini_tags["artist"]
            if ini_tags.get("album"):
                tags["album"] = ini_tags["album"]
            # Prefer a clear "Artist - Title" filename over unreliable uploader tags.
            filename_parts = [part.strip() for part in path.stem.split(" - ")]
            generic_audio_name = path.stem.casefold() in {"song", "guitar", "rhythm", "vocals", "drums", "keys"}
            if len(filename_parts) >= 2 and not ini_tags:
                detected_artist = filename_parts[0]
                detected_title = filename_parts[1]
                self.artist.set(detected_artist)
                self.title_value.set(detected_title)
                tags["artist"], tags["title"] = detected_artist, detected_title
            else:
                if tags.get("title"):
                    self.title_value.set(tags["title"])
                elif generic_audio_name and " - " in path.parent.name:
                    folder_parts = [part.strip() for part in path.parent.name.split(" - ", 1)]
                    self.artist.set(folder_parts[0])
                    self.title_value.set(folder_parts[1].split("(", 1)[0].strip())
                    tags["artist"], tags["title"] = self.artist.get(), self.title_value.get()
                else:
                    self.title_value.set(path.stem.replace("_", " "))
                if tags.get("artist"):
                    self.artist.set(tags["artist"])
            if tags.get("album"):
                self.album.set(tags["album"])
            detected = audio_duration_seconds(path)
            if detected:
                self.duration.set(round(detected, 1))
            self.status.set(f"Selected: {path.name} — ready to generate")
            if tags.get("title") or tags.get("artist"):
                self.search_query.set(f"{self.artist.get().strip()} {self.title_value.get().strip()}".strip())
                self.search_songs()
                self.fetch_art()

    def search_songs(self) -> None:
        query = self.search_query.get().strip()
        if not query:
            self.status.set("Type a song title or artist first.")
            return
        self.status.set("Searching song database…")
        self.update_idletasks()
        try:
            self.search_results = search_song_database(query, self.database_path)
        except Exception as exc:
            self.status.set("Song search failed.")
            messagebox.showerror(APP_NAME, f"Search failed:\n{exc}")
            return
        for item in self.search_tree.get_children():
            self.search_tree.delete(item)
        for index, result in enumerate(self.search_results):
            self.search_tree.insert("", "end", iid=str(index), values=(result["title"], result["artist"], result["album"]))
        if self.search_results:
            self.search_tree.selection_set("0")
            self.search_tree.focus("0")
        self.status.set(f"{len(self.search_results)} song(s) found. Choose one from the list.")

    def queue_search(self, _event=None) -> None:
        """Autocomplete after the user pauses typing, rather than on every key."""
        if self.search_after_id:
            self.after_cancel(self.search_after_id)
        if len(self.search_query.get().strip()) < 2:
            for item in self.search_tree.get_children():
                self.search_tree.delete(item)
            return
        self.search_after_id = self.after(650, self.search_songs)

    def use_selected_song(self) -> None:
        selection = self.search_tree.selection()
        if not selection or not self.search_results:
            self.status.set("Select a song from the search results first.")
            return
        result = self.search_results[int(selection[0])]
        self.title_value.set(result["title"])
        self.artist.set(result["artist"])
        self._show_art_placeholder()
        preview_dir = Path(tempfile.gettempdir()) / "RoboRiff"
        preview_dir.mkdir(parents=True, exist_ok=True)
        self.preview_art_path = preview_dir / "album_art_preview.jpg"
        try:
            download_album_art(result["title"], result["artist"], self.preview_art_path)
            self._show_art(self.preview_art_path)
            self.art_status.set("Preview loaded — saved when chart is generated")
        except Exception:
            self.preview_art_path = None
            self.art_status.set("No artwork found")
        self.status.set(f"Selected: {result['title']} — {result['artist']}")

    def fetch_art(self) -> None:
        title = safe_name(self.title_value.get())
        artist = self.artist.get().strip() or "Unknown Artist"
        preview_dir = Path(tempfile.gettempdir()) / "RoboRiff"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / "album_art_preview.jpg"
        self.art_status.set("Searching…")
        self.update_idletasks()
        try:
            album_name = self.album.get().strip()
            album = download_album_art(title, artist, preview_path, album_name)
        except Exception as exc:
            self.art_status.set("Could not find artwork")
            messagebox.showerror(APP_NAME, f"Could not fetch album artwork:\n{exc}")
            return
        if album:
            self.preview_art_path = preview_path
            self._show_art(preview_path)
            self.art_status.set(f"Preview loaded: {album}")
            self.status.set("Album artwork found. It will be saved when you generate the chart.")
        else:
            self.art_status.set("No matching artwork found")

    def _show_art_placeholder(self) -> None:
        if self.art_canvas is None:
            return
        self.art_canvas.delete("all")
        self.art_canvas.create_text(102, 94, text="♪", fill="#36c98f", font=("Segoe UI", 52, "bold"))
        self.art_canvas.create_text(102, 145, text="NO ARTWORK", fill="#8e9aaa", font=("Segoe UI", 9, "bold"))

    def _show_art(self, path: Path) -> None:
        if self.art_canvas is None or Image is None or ImageTk is None:
            return
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((205, 205), Image.Resampling.LANCZOS)
            self.art_image = ImageTk.PhotoImage(image)
            self.art_canvas.delete("all")
            self.art_canvas.create_image(102, 102, image=self.art_image)
        except Exception:
            self._show_art_placeholder()

    def generate(self) -> None:
        try:
            bpm = max(40, min(300, int(float(self.bpm.get())))) if self.bpm.get() != "Auto" else 120
            duration = max(2.0, float(self.duration.get())) if self.duration.get() != "Auto" else 30.0
        except (tk.TclError, ValueError):
            messagebox.showerror(APP_NAME, "Detected BPM and duration are not valid yet. Choose an audio file and try again.")
            return
        title = safe_name(self.title_value.get())
        folder = Path(self.output.get()).expanduser() / title
        source = Path(self.audio_path.get())
        artist = self.artist.get().strip() or "Unknown Artist"
        charter = self.charter.get().strip() or "RoboRiff"
        album = self.album.get().strip()
        mode = self.analysis_mode.get()
        use_gpu = bool(self.use_gpu.get())
        self.generate_button.configure(state="disabled")
        self.progress_bar.pack(fill="x", pady=(0, 10), before=self.generate_button)
        self.progress_bar.start(12)
        self.status.set("Starting chart generation…")
        threading.Thread(target=self._generate_worker, args=(source, folder, title, artist, charter, album, mode, use_gpu, bpm, duration), daemon=True).start()

    def _worker_status(self, message: str) -> None:
        self.after(0, lambda: self.status.set(message))

    def _generate_worker(self, source: Path, folder: Path, title: str, artist: str, charter: str, album: str, mode: str, use_gpu: bool, bpm: int, duration: float) -> None:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            onsets = None
            chord_times = None
            sustain_times = None
            pitch_times = None
            if source.is_file():
                self._worker_status("Preparing audio analysis…")
                analysis_source, source_kind = choose_analysis_source(source)
                if mode.startswith("Guitar Separation") and source_kind == "full mix fallback":
                    from source_separator import get_or_create_guitar_stem
                    self._worker_status("Separating guitar stem locally…")
                    analysis_source = get_or_create_guitar_stem(source)
                    source_kind = "Demucs guitar stem"
                try:
                    if mode.startswith("Riley Guitar"):
                        self._worker_status(f"Running Riley guitar AI on {source_kind}…")
                        detected_bpm, stem_duration, onsets, chord_times, sustain_times, pitch_times = analyze_with_riley(analysis_source, use_gpu=use_gpu)
                        analysis_label = "Riley guitar AI"
                    elif mode.startswith("Basic Pitch") or mode.startswith("Guitar Separation"):
                        self._worker_status(f"Running Basic Pitch AI on {source_kind}…")
                        detected_bpm, stem_duration, onsets, chord_times, sustain_times, pitch_times = analyze_with_basic_pitch(analysis_source, use_gpu=use_gpu)
                        analysis_label = "Basic Pitch AI"
                    else:
                        raise RuntimeError("Classic mode selected")
                except Exception:
                    if mode.startswith("Riley Guitar"):
                        self._worker_status("Riley unavailable; falling back to Basic Pitch…")
                        detected_bpm, stem_duration, onsets, chord_times, sustain_times, pitch_times = analyze_with_basic_pitch(analysis_source, use_gpu=use_gpu)
                        analysis_label = "Basic Pitch fallback"
                    else:
                        self._worker_status("AI unavailable; using classic audio analysis…")
                        detected_bpm, stem_duration, onsets, chord_times, sustain_times = analyze_audio(analysis_source)
                        analysis_label = "classic audio analysis"
                bpm = round(detected_bpm)
                duration = audio_duration_seconds(choose_full_song_source(source)) or stem_duration
                self.after(0, lambda: (self.bpm.set(bpm), self.duration.set(round(duration, 1))))
                self._worker_status(f"Analysis source: {source_kind} · {analysis_label}")
            self._worker_status("Writing Clone Hero files…")
            (folder / "notes.chart").write_text(build_chart(title, artist, charter, bpm, duration, onsets, chord_times if onsets else None, sustain_times if onsets else None, pitch_times if onsets else None), encoding="utf-8")
            ini = "\n".join([f"name = {title}", f"artist = {artist}", f"charter = {charter}", f"song_length = {int(duration * 1000)}", "preview_start_time = 0", "icon = roboriff", "diff_guitar = 0", "diff_bass = 0", "loading_phrase = Generated by RoboRiff"]) + "\n"
            (folder / "song.ini").write_text(ini, encoding="utf-8")
            if self.preview_art_path and self.preview_art_path.exists():
                (folder / "album_art.jpg").write_bytes(self.preview_art_path.read_bytes())
            elif not (folder / "album_art.jpg").exists():
                download_album_art(title, artist, folder / "album_art.jpg", album)
            if source.is_file():
                target = folder / source.name
                if source.resolve() != target.resolve():
                    target.write_bytes(source.read_bytes())
            self.after(0, lambda: self._generation_finished(folder, None))
        except Exception as exc:
            self.after(0, lambda: self._generation_finished(folder, exc))

    def _generation_finished(self, folder: Path, error: Exception | None) -> None:
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.generate_button.configure(state="normal")
        if error:
            self.status.set(f"Generation failed: {error}")
            messagebox.showerror(APP_NAME, f"Chart generation failed:\n{error}")
            return
        self.status.set(f"Chart created: {folder}  ·  ready for testing")
        if (folder / "album_art.jpg").exists():
            self._show_art(folder / "album_art.jpg")
        self.art_status.set("Album art saved")
        messagebox.showinfo(APP_NAME, f"The Clone Hero song folder was created at:\n{folder}")


if __name__ == "__main__":
    RoboRiff().mainloop()
