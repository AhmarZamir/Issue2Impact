import io
import zipfile

import pytest

from src.ui.uploads import save_zip_upload


def make_zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("project/app.py", "print('hello')")
    return buffer.getvalue()


def test_save_zip_upload_persists_valid_archive(tmp_path):
    saved_path = save_zip_upload(
        make_zip_bytes(),
        "project.zip",
        upload_root=tmp_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".zip"
    assert zipfile.is_zipfile(saved_path)


def test_save_zip_upload_rejects_invalid_archive(tmp_path):
    with pytest.raises(ValueError, match="valid ZIP"):
        save_zip_upload(
            b"not-a-zip",
            "project.zip",
            upload_root=tmp_path,
        )


def test_save_zip_upload_rejects_non_zip_name(tmp_path):
    with pytest.raises(ValueError, match=".zip"):
        save_zip_upload(
            make_zip_bytes(),
            "project.txt",
            upload_root=tmp_path,
        )
