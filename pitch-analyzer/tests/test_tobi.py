"""簡易版 X-JToBI 層生成のテスト。"""

import numpy as np
import pytest

from pitchan import segment, tobi
from pitchan.textproc import analyze_text


@pytest.fixture
def aps_with_times():
    """疑似アラインメント付きの AP 列。AP2 の後に 0.5 秒のポーズを入れる。"""
    aps = analyze_text("私は山梨大学で、音声を研究しています。")
    t = 0.5
    intervals = []
    for ap in aps:
        for w in ap.words:
            intervals.append((t, t + 0.4, w.pron))
            t += 0.4
        if ap.index == 1:  # 読点位置に実ポーズ
            t += 0.5
    segment.assign_times(aps, intervals)
    return aps


def test_nucleus_marked_words(aps_with_times):
    aps = aps_with_times
    # ワタシ+ワ 4型 → 核は 2 語目の「ワ」
    assert tobi.nucleus_marked_words(aps[0]) == ["ワタシ", "ワ'"]
    # ヤマナシダイガク+デ 5型 → 1 語目の 5 モーラ目「ダ」の後
    assert tobi.nucleus_marked_words(aps[1]) == ["ヤマナシダ'イガク", "デ"]
    # オンセー+オ 1型 → 頭高
    assert tobi.nucleus_marked_words(aps[2]) == ["オ'ンセー", "オ"]


def test_nucleus_unaccented():
    aps = analyze_text("飴です。")  # アメ+デス
    ap = aps[0]
    ap.accent_type = 0  # 平板とみなす
    assert tobi.nucleus_marked_words(ap) == [w.pron for w in ap.words]


def test_bi_points(aps_with_times):
    aps = aps_with_times
    points = dict(tobi.bi_points(aps))
    # AP 内部の語境界は 1
    assert points[aps[0].words[0].t_end] == "1"
    # ポーズなしで隣接する AP 境界は 2
    assert points[aps[0].t_end] == "2"
    # 0.5 秒のポーズを挟む境界は 3
    assert points[aps[1].t_end] == "3"
    # 発話末は 3
    assert points[aps[-1].t_end] == "3"


def _times_grid(t0, t1):
    return np.arange(t0, t1, 0.005)


def test_classify_bpm_rise(aps_with_times):
    ap = aps_with_times[-1]
    phones = [(ap.t_start, ap.t_end - 0.15, "x"), (ap.t_end - 0.15, ap.t_end, "a")]
    times = _times_grid(0, ap.t_end + 0.1)
    f0 = np.zeros_like(times)
    sel = (times >= ap.t_end - 0.15) & (times <= ap.t_end)
    f0[sel] = np.linspace(0, 4, sel.sum())  # 末尾で 4 半音上昇
    assert tobi.classify_bpm(times, f0, phones, ap) == "H%"


def test_classify_bpm_rise_fall(aps_with_times):
    ap = aps_with_times[-1]
    phones = [(ap.t_start, ap.t_end - 0.15, "x"), (ap.t_end - 0.15, ap.t_end, "a")]
    times = _times_grid(0, ap.t_end + 0.1)
    f0 = np.zeros_like(times)
    sel = (times >= ap.t_end - 0.15) & (times <= ap.t_end)
    n = sel.sum()
    shape = np.concatenate([np.linspace(0, 4, n // 2), np.linspace(4, 0, n - n // 2)])
    f0[sel] = shape  # 上昇後下降
    assert tobi.classify_bpm(times, f0, phones, ap) == "HL%"


def test_classify_bpm_flat(aps_with_times):
    ap = aps_with_times[-1]
    phones = [(ap.t_start, ap.t_end, "a")]
    times = _times_grid(0, ap.t_end + 0.1)
    f0 = np.full_like(times, 1.0)  # 平坦
    assert tobi.classify_bpm(times, f0, phones, ap) == ""


def test_tone_points(aps_with_times):
    aps = aps_with_times
    bpm = {ap.index: "" for ap in aps}
    bpm[aps[-1].index] = "LH%"
    pts = tobi.tone_points(aps, bpm)
    labels = [p[1] for p in pts]
    # 発話頭と実ポーズ後(AP2)に %L
    assert labels.count("%L") == 2
    # 有核句には H*+L(4句とも有核)
    assert labels.count("H*+L") == 4
    # 頭高(1型)のオンセーオには H- が付かない
    n_hminus = labels.count("H-")
    assert n_hminus == sum(1 for ap in aps if ap.accent_type != 1)
    # 句末は L%、BPM 付きは連結表記
    assert labels.count("L%") == 3
    assert "L%LH%" in labels
    # 時刻は単調増加
    times_ = [p[0] for p in pts]
    assert all(t2 > t1 for t1, t2 in zip(times_, times_[1:]))


def test_peak_excl_bpm(aps_with_times):
    ap = aps_with_times[-1]
    phones = [(ap.t_start, ap.t_end - 0.15, "x"), (ap.t_end - 0.15, ap.t_end, "a")]
    times = _times_grid(0, ap.t_end + 0.1)
    f0 = np.zeros_like(times)
    body = (times >= ap.t_start) & (times < ap.t_end - 0.15)
    tail = (times >= ap.t_end - 0.15) & (times <= ap.t_end)
    f0[body] = 2.0  # 句本体は 2 半音
    f0[tail] = np.linspace(0, 6, tail.sum())  # BPM 区間で 6 半音まで上昇
    # BPM ありなら末尾の上昇(6半音)を除いて 2 半音がピークになる
    st, t = tobi.peak_excl_bpm(times, f0, phones, ap, "H%")
    assert st == pytest.approx(2.0)
    assert t < ap.t_end - 0.15
    # BPM なしなら全体の最大(末尾の 6 半音)
    st2, _ = tobi.peak_excl_bpm(times, f0, phones, ap, "")
    assert st2 == pytest.approx(6.0, abs=0.1)


def test_write_xjtobi_textgrid(aps_with_times, tmp_path):
    from praatio import textgrid as ptg

    aps = aps_with_times
    dur = aps[-1].t_end + 0.5
    phones = [
        (w.t_start, w.t_end, "p") for ap in aps for w in ap.words
    ]
    times = _times_grid(0, dur)
    f0 = np.full_like(times, 1.0)
    path = tmp_path / "x.TextGrid"
    tobi.write_xjtobi_textgrid(path, aps, phones, dur, times, f0)

    tg = ptg.openTextgrid(str(path), includeEmptyIntervals=False)
    assert set(tg.tierNames) == {"segments", "tones", "words", "BI"}
    words = [e.label for e in tg.getTier("words").entries]
    assert "ヤマナシダ'イガク" in words
    bi_labels = [e.label for e in tg.getTier("BI").entries]
    assert set(bi_labels) <= {"1", "2", "3"}
    assert "3" in bi_labels
    tone_labels = [e.label for e in tg.getTier("tones").entries]
    assert "%L" in tone_labels
    assert "H*+L" in tone_labels
    assert any(lab.startswith("L%") for lab in tone_labels)
