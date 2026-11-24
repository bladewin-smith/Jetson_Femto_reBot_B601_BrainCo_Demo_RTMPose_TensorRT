import hashlib
from pathlib import Path

import pytest

from arm_control.rebot_b601 import StartupGuardError
from scripts.b601_desktop_app import (
    required_engines,
    validate_engine_files,
    validate_table_flat_geometry,
)


def test_table_flat_geometry_accepts_validated_startup_shape():
    validate_table_flat_geometry(
        [-1.473068, -0.004005, -0.002861, 0.003624, -0.045968, -0.348097]
    )


@pytest.mark.parametrize(
    "positions,joint",
    [
        ([0.0, -0.19, 0.0, 0.0, 0.0, 0.0], 2),
        ([0.0, 0.0, -0.19, 0.0, 0.0, 0.0], 3),
        ([0.0, 0.0, 0.0, 0.26, 0.0, 0.0], 4),
    ],
)
def test_table_flat_geometry_rejects_unsupported_shape(positions, joint):
    with pytest.raises(StartupGuardError, match=rf"joints \[{joint}\]"):
        validate_table_flat_geometry(positions)


def test_verified_engine_bundle_requires_three_nonempty_files(tmp_path: Path):
    engines = required_engines(tmp_path)
    manifest_lines = []
    for path in engines.values():
        path.write_bytes(b"engine")
        manifest_lines.append(f"{hashlib.sha256(b'engine').hexdigest()}  {path.name}")
    (tmp_path / "verified_engines.sha256").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    validate_engine_files(engines)

    engines["HAND_ENGINE"].write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty"):
        validate_engine_files(engines)
