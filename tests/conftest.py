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


@pytest.fixture(scope="session")
def gguf_py():
    """gguf-py from the llama.cpp checkout (LLAMA_CPP_DIR, default ../llama.cpp)."""
    p = env_path("LLAMA_CPP_DIR") or ROOT.parent / "llama.cpp"
    if not (p / "gguf-py" / "gguf").exists():
        pytest.skip("llama.cpp checkout with gguf-py not found (set LLAMA_CPP_DIR)")
    sys.path.insert(0, str(p / "gguf-py"))
    import gguf
    return gguf
