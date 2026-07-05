"""偽 mfa コマンドを使った CLI の end-to-end テスト。"""

import os
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import pytest

from pitchan.cli import main

TEXT = "私は山梨大学で、音声を研究しています。"


@pytest.fixture
def fake_mfa_on_path(tmp_path, monkeypatch):
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


def _make_wav(path: Path, duration: float, f0_hz: float = 150.0, sr: int = 16000):
    x = np.zeros(int(sr * duration))
    period = int(sr / f0_hz)
    x[::period] = 0.5
    sf.write(path, x, sr)


def test_e2e_batch(tmp_path, fake_mfa_on_path):
    from pitchan.textproc import analyze_text

    aps = analyze_text(TEXT)
    n_words = sum(len(ap.words) for ap in aps)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_wav(data_dir / "sample.wav", duration=0.5 + n_words * 0.3 + 0.5)
    (data_dir / "sample.txt").write_text(TEXT, encoding="utf-8")

    out_dir = tmp_path / "out"
    rc = main(
        ["batch", "--dir", str(data_dir), "--out", str(out_dir),
         "--jobs", "1", "--xjtobi"]
    )
    assert rc == 0

    summary = pd.read_csv(out_dir / "sample_ap_summary.csv")
    assert len(summary) == len(aps)
    assert "bpm_auto" in summary.columns
    assert (out_dir / "sample_xjtobi.TextGrid").exists()
    # 全フレームが 150 Hz なので半音値はほぼ 0(基準 = 幾何平均)
    assert summary["f0_mean_st"].abs().max() < 1.0

    frames = pd.read_csv(out_dir / "sample_frames.csv")
    assert (frames["f0_hz"] > 0).any()
    contours = pd.read_csv(out_dir / "sample_ap_contours.csv")
    assert len(contours) == len(aps)
    assert (out_dir / "sample.json").exists()
    assert (out_dir / "sample.TextGrid").exists()


def test_collect_pairs_excludes_output_dir(tmp_path):
    from pitchan.cli import _collect_pairs

    data = tmp_path / "data"
    data.mkdir()
    (data / "a.wav").write_bytes(b"")
    (data / "a.txt").write_text("x", encoding="utf-8")
    # 出力フォルダが入力フォルダ内にあり、過去の実行のコピーが残っている状況
    leftover = data / "results" / "work" / "corpus" / "spk"
    leftover.mkdir(parents=True)
    (leftover / "a.wav").write_bytes(b"")  # コピー(.lab はあるが .txt はない)

    pairs = _collect_pairs(data, "spk", exclude=data / "results")
    assert [p.name for p in pairs] == ["a"]  # 元の 1 件のみ、コピーは無視


def test_e2e_missing_mfa(tmp_path, monkeypatch):
    if shutil.which("mfa"):
        pytest.skip("実物の mfa が存在する環境ではスキップ")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_wav(data_dir / "a.wav", 2.0)
    (data_dir / "a.txt").write_text(TEXT, encoding="utf-8")
    rc = main(["batch", "--dir", str(data_dir), "--out", str(tmp_path / "o")])
    assert rc == 1  # mfa 不在の明確なエラーで終了


def test_e2e_xjtobi_measure(tmp_path):
    """手修正済み TextGrid からのラベル駆動計測(MFA 不要)。"""
    from pitchan import segment, tobi
    from pitchan.textproc import analyze_text

    aps = analyze_text(TEXT)
    t = 0.5
    intervals = []
    for ap in aps:
        for w in ap.words:
            intervals.append((t, t + 0.3, w.pron))
            t += 0.3
    segment.assign_times(aps, intervals)
    dur = t + 0.5

    wav = tmp_path / "rec.wav"
    _make_wav(wav, duration=dur)

    phones = [(w.t_start, w.t_end, "a") for ap in aps for w in ap.words]
    times = np.arange(0, dur, 0.005)
    f0 = np.full_like(times, 0.0)
    tg_path = tmp_path / "rec_xjtobi.TextGrid"
    tobi.write_xjtobi_textgrid(tg_path, aps, phones, dur, times, f0)

    out = tmp_path / "out"
    rc = main([
        "xjtobi-measure", "--wav", str(wav), "--textgrid", str(tg_path),
        "--out", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out / "rec_xjtobi_measures.csv")
    assert len(df) == len(aps)
    assert set(["ap_kana", "nucleus_mora", "bi", "bpm",
                "peak_excl_bpm_st"]) <= set(df.columns)
    # 150 Hz 一定の合成音 → 半音値ほぼ 0
    assert df["f0_max_st"].abs().max() < 1.0
