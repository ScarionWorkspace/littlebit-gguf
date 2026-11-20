import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def env_path(name):
    v = os.environ.get(name)
    return Path(v) if v else None


@pytest.fixture
def littlebit_dir():
    p = env_path("LITTLEBIT_DIR")
    if not p or not (p / "quantization").exists():
        pytest.skip("LITTLEBIT_DIR not set (checkout of SamsungLabs/LittleBit)")
    return p
