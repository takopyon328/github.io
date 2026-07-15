"""共有フィクスチャ。"""

import os
import stat
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fake_mfa_on_path(tmp_path, monkeypatch):
    """偽 mfa コマンド(tests/fake_mfa.py)を PATH の先頭に置く。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    src = Path(__file__).parent / "fake_mfa.py"
    dst = bin_dir / "mfa"
    dst.write_text(
        f"#!{sys.executable}\n" + src.read_text(encoding="utf-8").split("\n", 1)[1],
        encoding="utf-8",
    )
    dst.chmod(dst.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
