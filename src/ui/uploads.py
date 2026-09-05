from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


DEFAULT_UPLOAD_ROOT = Path("workspace/uploads")


def save_zip_upload(
    file_bytes: bytes,
    original_name: str,
    upload_root: str | Path = DEFAULT_UPLOAD_ROOT,
) -> Path:
    """Persist an uploaded ZIP safely so the normal repository pipeline can use it."""
    if not file_bytes:
        raise ValueError("The uploaded ZIP file is empty.")

    if not original_name.lower().endswith(".zip"):
        raise ValueError("Repository upload must be a .zip archive.")

    digest = hashlib.sha256(file_bytes).hexdigest()
    root = Path(upload_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    destination = root / f"repository-{digest[:20]}.zip"
    destination.write_bytes(file_bytes)

    if not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise ValueError("The uploaded file is not a valid ZIP archive.")

    return destination
