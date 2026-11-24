"""Download and verify the optional ChartForge AI model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_manager import ModelDownloadError, download_model, is_model_ready, load_manifest


def main() -> None:
    manifest = load_manifest()
    if is_model_ready(manifest):
        print("AI model is already installed.")
        return

    def progress(downloaded: int, total: int) -> None:
        if total:
            print(f"\rDownloading AI model: {downloaded / total * 100:5.1f}%", end="", flush=True)
        else:
            print(f"\rDownloaded {downloaded / 1_048_576:.1f} MB", end="", flush=True)

    try:
        target = download_model(manifest, progress=progress)
    except ModelDownloadError as error:
        print(f"\nModel download unavailable: {error}")
        return
    print(f"\nAI model installed in {target}")


if __name__ == "__main__":
    main()
