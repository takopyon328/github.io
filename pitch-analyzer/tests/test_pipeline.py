"""MFA を除く全工程の結合テスト。

合成音声(F0 が既知の正弦波ボコーダ的信号)と疑似アラインメントを使い、
セグメンテーション・正規化・出力生成を検証する。
"""

import numpy as np
import pytest

from pitchan import normalize, outputs, segment
from pitchan.textproc import analyze_text


@pytest.fixture
def aligned_aps():
    aps = analyze_text("私は山梨大学で、音声を研究しています。")
    # 1 語 = 0.5 秒の疑似アラインメント
    t = 0.5  # 先頭に 0.5 秒の無音
    intervals = []
    for ap in aps:
        for w in ap.words:
            intervals.append((t, t + 0.5, w.pron))
            t += 0.5
    segment.assign_times(aps, intervals)
    return aps, t


def test_assign_times(aligned_aps):
    aps, _ = aligned_aps
    assert aps[0].t_start == 0.5
    assert aps[0].t_end == 1.5  # 私+は = 2 語
    assert all(ap.t_start is not None for ap in aps)
    # 0.5 秒/語 の疑似データでは 1 モーラ 100ms 程度 → low_confidence なし
    assert not any(ap.low_confidence for ap in aps)


def test_assign_times_word_count_mismatch():
    aps = analyze_text("私は学生です。")
    with pytest.raises(ValueError):
        segment.assign_times(aps, [(0.0, 0.5, "ワタシ")])


def test_outputs_build(aligned_aps, tmp_path):
    aps, total = aligned_aps
    times = np.arange(0, total + 0.5, 0.005)
    # 120 Hz 基調で時間とともに上昇する既知の F0(20% を無声に)
    f0 = 120.0 + 30.0 * times / times[-1]
    f0[::5] = 0.0
    ref, mu, sigma = normalize.speaker_reference([f0])
    st = normalize.to_semitone(f0, ref)
    z = normalize.to_log_z(f0, mu, sigma)

    frames = outputs.build_frames_df("test", times, f0, st, z, aps)
    assert (frames["ap_index"] >= 0).any()
    assert (frames.loc[frames["time_sec"] < 0.5, "ap_index"] == -1).all()

    summary = outputs.build_ap_summary_df("test", times, st, aps)
    assert len(summary) == len(aps)
    assert summary["f0_mean_st"].notna().all()
    # F0 は単調上昇なので後の AP ほど平均が高い
    assert summary["f0_mean_st"].is_monotonic_increasing

    contours = outputs.build_contours_df("test", times, st, aps, n_points=30)
    assert len(contours) == len(aps)
    assert contours.filter(like="p").notna().all().all()

    outputs.write_json(tmp_path / "test.json", "test", aps, {}, contours)
    assert (tmp_path / "test.json").exists()
    outputs.write_textgrid(tmp_path / "test.TextGrid", aps, [], times[-1])
    assert (tmp_path / "test.TextGrid").exists()


def test_textgrid_roundtrip(aligned_aps, tmp_path):
    """書き出した TextGrid を align.read_word_intervals で読み戻せる。"""
    from pitchan import align

    aps, total = aligned_aps
    outputs.write_textgrid(tmp_path / "rt.TextGrid", aps, [], total)
    tokens = [w.pron for ap in aps for w in ap.words]
    words, _ = align.read_word_intervals(tmp_path / "rt.TextGrid", tokens)
    assert [w[2] for w in words] == tokens


def test_flag_low_confidence_f0(aligned_aps):
    aps, total = aligned_aps
    times = np.arange(0, total + 0.5, 0.005)
    f0 = np.full_like(times, 120.0)

    # AP0: 有声率を下げる(70% を無声化)
    sel0 = (times >= aps[0].t_start) & (times <= aps[0].t_end)
    idx0 = np.where(sel0)[0]
    f0[idx0[: int(len(idx0) * 0.7)]] = 0.0
    # AP1: オクターブ跳躍を入れる(隣接フレームで 2 倍)
    sel1 = (times >= aps[1].t_start) & (times <= aps[1].t_end)
    idx1 = np.where(sel1)[0]
    f0[idx1[len(idx1) // 2]] = 240.0

    segment.flag_low_confidence_f0(aps, times, f0)
    assert aps[0].low_confidence      # 有声率
    assert aps[1].low_confidence      # 跳躍
    assert not aps[2].low_confidence  # 正常な句は影響なし
