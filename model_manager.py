"""Optional local AI model downloader for ChartForge."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


class ModelDownloadError(RuntimeError):
    pass


def load_manifest(path: Path | None = None) -> dict:
    manifest_path = path or Path(__file__).with_name("models") / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def model_path(manifest: dict, models_dir: Path | None = None) -> Path:
    root = models_dir or Path(__file__).with_name("models")
    return root / manifest["model"]["directory"]


def is_model_ready(manifest: dict, models_dir: Path | None = None) -> bool:
    target = model_path(manifest, models_dir)
    required = manifest["model"].get("required_files", [])
    return target.is_dir() and all((target / name).is_file() for name in required)


def download_model(manifest: dict, models_dir: Path | None = None, progress=None) -> Path:
    model = manifest["model"]
    url = model.get("url", "").strip()
    expected_hash = model.get("sha256", "").lower().strip()
    if not url or not expected_hash:
        raise ModelDownloadError("The AI model is not configured yet.")

    root = models_dir or Path(__file__).with_name("models")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chartforge-model-") as temp_dir:
        archive = Path(temp_dir) / "model.download"
        request = Request(url, headers={"User-Agent": "ChartForge/0.1"})
        with urlopen(request, timeout=60) as response, archive.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)

        digest = hashlib.sha256(archive.read_bytes()).hexdigest().lower()
        if digest != expected_hash:
            raise ModelDownloadError("The downloaded model failed its SHA-256 check.")

        target = model_path(manifest, root)
        staging = Path(temp_dir) / "staging"
        staging.mkdir()
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as package:
                package.extractall(staging)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, target / model.get("filename", "model.onnx"))
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        return target
