from __future__ import annotations

from pathlib import Path

import pytest

from lightshap.config import DatasetConfig
from lightshap.data.download import ensure_raw_dataset, verify_sha256, write_sha256_yaml
from lightshap.exceptions import DataIntegrityError
from lightshap.utils.hashing import sha256_file
from lightshap.utils.io import atomic_write_text


def _tiny(path: Path, payload: bytes = b"lightshap-bytes") -> Path:
    path.write_bytes(payload)
    return path


@pytest.mark.unit
def test_first_seen_hash_is_locked_into_yaml(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    blob = _tiny(raw / "reviews_Beauty_5.json.gz")
    digest = sha256_file(blob)
    yml = tmp_path / "beauty.yaml"
    atomic_write_text(yml, 'name: beauty\nsha256: ""\n')
    rec = ensure_raw_dataset(
        DatasetConfig(
            name="beauty",
            url="http://example.invalid/x",
            filename="reviews_Beauty_5.json.gz",
            expected_sha256=None,
            implicit_threshold=None,
            batch_size=8,
        ),
        raw,
        allow_missing_sha256=False,
        yaml_path=yml,
    )
    assert rec["verified"] is True
    assert rec["recorded_first_seen"] is True
    assert rec["sha256"] == digest
    assert f'sha256: "{digest}"' in yml.read_text()


@pytest.mark.unit
def test_mismatch_still_fails(tmp_path: Path) -> None:
    p = _tiny(tmp_path / "ml-1m.zip")
    with pytest.raises(DataIntegrityError, match="mismatch"):
        verify_sha256(p, "0" * 64, allow_missing=False)


@pytest.mark.unit
def test_matching_hash_passes(tmp_path: Path) -> None:
    p = _tiny(tmp_path / "ml-1m.zip")
    digest = sha256_file(p)
    got, recorded = verify_sha256(p, digest, allow_missing=False)
    assert got == digest
    assert recorded is False


@pytest.mark.unit
def test_write_sha256_yaml_roundtrip(tmp_path: Path) -> None:
    y = tmp_path / "ml1m.yaml"
    y.write_text('name: ml1m\nsha256: ""\n')
    write_sha256_yaml(y, "abc123")
    assert 'sha256: "abc123"' in y.read_text()
