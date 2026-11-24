"""Optional local guitar-specific transcription backend.

The vendored Riley/GAPS implementation is used only when the user selects
the experimental guitar model. Basic Pitch remains the normal fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path


VENDOR_ROOT = Path(__file__).with_name("vendor")
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))


def predict_guitar(path: Path, use_gpu: bool = False):
    import torch
    from tabgen.core.transcription.riley import predict

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    return predict(path, device=device)
