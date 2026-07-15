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
    # この文では各 AP が 1 文節なので、AP 内部の境界(BI=1)は付かない
    # (私+は は 1 文節。語境界は BI の対象にしない)
    assert "1" not in points.values()
    # ポーズなしで隣接する AP 境界は 2
    assert points[aps[0].t_end] == "2"
    # 0.5 秒のポーズを挟む境界は 3
    assert points[aps[1].t_end] == "3"
    # 発話末は 3
    assert points[aps[-1].t_end] == "3"


def test_bi_points_bunsetsu_internal_boundary():
    """AP 内に複数文節がある場合、文節境界に BI=1 が付く。"""
    from pitchan.textproc import AccentPhrase, Word

    w1 = Word("良い", "ヨイ", 2, "形容詞", "自立", t_start=0.5, t_end=0.9)
    w2 = Word("天気", "テンキ", 3, "名詞", "一般", t_start=0.9, t_end=1.4)
    w3 = Word("です", "デス", 2, "助動詞", "*", t_start=1.4, t_end=1.7)
    ap = AccentPhrase(index=0, words=[w1, w2, w3], mora_count=7)
    ap.t_start, ap.t_end = 0.5, 1.7
    points = dict(tobi.bi_points([ap]))
    assert points[0.9] == "1"      # ヨイ | テンキデス の文節境界
    assert 1.4 not in points       # 天気+です は同一文節(語境界は対象外)
    assert points[1.7] == "3"      # 発話末


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
    assert set(tg.tierNames) == {"segments", "tones", "words", "words_pred", "BI"}
    pred_labels = [e.label for e in tg.getTier("words_pred").entries]
    assert all("/" in lab for lab in pred_labels)  # 辞書型 /N が付いている
    words = [e.label for e in tg.getTier("words").entries]
    # 単位は文節: 山梨大学+で が 1 区間になり、核記号が入る
    assert "ヤマナシダ'イガクデ" in words
    assert "ワタシワ'" in words  # 私+は も 1 文節
    bi_labels = [e.label for e in tg.getTier("BI").entries]
    assert set(bi_labels) <= {"1", "2", "3"}
    assert "3" in bi_labels
    tone_labels = [e.label for e in tg.getTier("tones").entries]
    assert "%L" in tone_labels
    assert "H*+L" in tone_labels
    assert any(lab.startswith("L%") for lab in tone_labels)


def test_classify_bpm_lh(aps_with_times):
    """LH%(上昇調2)= 最終モーラの後半まで低く保たれてから上昇する。"""
    ap = aps_with_times[-1]
    phones = [(ap.t_start, ap.t_end - 0.2, "x"), (ap.t_end - 0.2, ap.t_end, "a")]
    times = _times_grid(0, ap.t_end + 0.1)
    f0 = np.zeros_like(times)
    sel = (times >= ap.t_end - 0.2) & (times <= ap.t_end)
    n = sel.sum()
    k = int(n * 0.65)
    f0[sel] = np.concatenate(
        [np.zeros(k), np.linspace(0, 5, n - k)]
    )  # 低平坦 65% → 遅れて 5 半音上昇
    assert tobi.classify_bpm(times, f0, phones, ap) == "LH%"


def test_classify_bpm_early_rise_is_h(aps_with_times):
    """先行するアクセント下降が食い込んだだけの早い上昇は H% と判定する。"""
    ap = aps_with_times[-1]
    phones = [(ap.t_start, ap.t_end - 0.2, "x"), (ap.t_end - 0.2, ap.t_end, "a")]
    times = _times_grid(0, ap.t_end + 0.1)
    f0 = np.zeros_like(times)
    sel = (times >= ap.t_end - 0.2) & (times <= ap.t_end)
    n = sel.sum()
    k = n // 4
    f0[sel] = np.concatenate(
        [np.linspace(3, 0, k), np.linspace(0, 5, n - k)]
    )  # 前半 25% で核由来の下降 → すぐ上昇
    assert tobi.classify_bpm(times, f0, phones, ap) == "H%"


def test_classify_bpm_hlh(aps_with_times):
    ap = aps_with_times[-1]
    phones = [(ap.t_start, ap.t_end - 0.3, "x"), (ap.t_end - 0.3, ap.t_end, "a")]
    times = _times_grid(0, ap.t_end + 0.1)
    f0 = np.zeros_like(times)
    sel = (times >= ap.t_end - 0.3) & (times <= ap.t_end)
    n = sel.sum()
    k = n // 3
    f0[sel] = np.concatenate(
        [np.linspace(0, 4, k), np.linspace(4, 1, k), np.linspace(1, 5, n - 2 * k)]
    )  # 上昇→下降→上昇
    assert tobi.classify_bpm(times, f0, phones, ap) == "HLH%"


def test_bpm_region_uses_final_vowel(aps_with_times):
    """最終モーラ=「子音+母音」のとき、母音の開始が区間の起点になる。"""
    ap = aps_with_times[0]
    phones = [
        (ap.t_start, ap.t_end - 0.3, "w"),
        (ap.t_end - 0.3, ap.t_end - 0.1, "k"),   # 最終モーラの子音
        (ap.t_end - 0.1, ap.t_end, "a"),          # 最終モーラの母音
    ]
    region = tobi._bpm_region(phones, ap)
    assert region == (ap.t_end - 0.1, ap.t_end)


def test_parse_roundtrip(aps_with_times, tmp_path):
    """write → parse で AP 構造・核位置・BI が復元できる。"""
    aps = aps_with_times
    dur = aps[-1].t_end + 0.5
    phones = [(w.t_start, w.t_end, "a") for ap in aps for w in ap.words]
    times = _times_grid(0, dur)
    f0 = np.full_like(times, 1.0)  # 平坦 → BPM なし
    path = tmp_path / "rt.TextGrid"
    tobi.write_xjtobi_textgrid(path, aps, phones, dur, times, f0)

    laps, segments = tobi.parse_xjtobi_textgrid(path)
    assert len(laps) == len(aps)
    for lap, ap in zip(laps, aps):
        assert lap.kana == ap.kana
        assert lap.nucleus_mora == (ap.accent_type or 0)
        assert lap.t_start == pytest.approx(ap.t_start)
        assert lap.t_end == pytest.approx(ap.t_end)
        assert lap.bpm == ""
    assert len(segments) == len(phones)
    # ポーズを挟む AP1 の右端は BI=3、隣接する AP0 は BI=2
    assert laps[0].bi == "2"
    assert laps[1].bi == "3"


def test_parse_reflects_manual_edits(aps_with_times, tmp_path):
    """手修正(核の移動・削除、BPM の追加)が解析に反映される。"""
    from praatio import textgrid as ptg

    aps = aps_with_times
    dur = aps[-1].t_end + 0.5
    tg = ptg.Textgrid()
    word_entries = []
    for ap in aps:
        for w, lab in zip(ap.words, tobi.nucleus_marked_words(ap)):
            word_entries.append((w.t_start, w.t_end, lab))
    # 手修正の模擬: AP0 の核を削除(平板化)、AP1 の核を 5→3 モーラ目に移動
    word_entries[1] = (word_entries[1][0], word_entries[1][1], "ワ")
    word_entries[2] = (word_entries[2][0], word_entries[2][1], "ヤマナ'シダイガク")
    tg.addTier(ptg.IntervalTier("words", word_entries, 0, dur))
    tg.addTier(ptg.PointTier("BI", tobi.bi_points(aps), 0, dur))
    # 手修正の模擬: 最終 AP に BPM を追加
    tg.addTier(ptg.PointTier("tones", [(aps[-1].t_end, "L%HLH%")], 0, dur))
    path = tmp_path / "edited.TextGrid"
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)

    laps, _ = tobi.parse_xjtobi_textgrid(path)
    assert laps[0].nucleus_mora == 0        # 核削除 → 平板
    assert laps[1].nucleus_mora == 3        # 核移動
    assert laps[-1].bpm == "HLH%"           # BPM 追加


def test_measure_labeled_aps(aps_with_times, tmp_path):
    aps = aps_with_times
    dur = aps[-1].t_end + 0.5
    phones = [(w.t_start, w.t_end, "a") for ap in aps for w in ap.words]
    times = _times_grid(0, dur)
    f0 = np.full_like(times, 2.0)
    path = tmp_path / "m.TextGrid"
    tobi.write_xjtobi_textgrid(path, aps, phones, dur, times, f0)
    laps, segments = tobi.parse_xjtobi_textgrid(path)

    rows = tobi.measure_labeled_aps(times, f0, laps, segments, "m")
    assert len(rows) == len(aps)
    assert all(r["f0_max_st"] == pytest.approx(2.0) for r in rows)
    assert all(r["accented"] == 1 for r in rows)  # 4 句とも有核


def test_bpm_region_long_vowel_second_half(aps_with_times):
    """長母音(ː)末の句では、音素の後半のみが最終モーラ相当になる。"""
    ap = aps_with_times[0]
    phones = [
        (ap.t_start, ap.t_end - 0.4, "s"),
        (ap.t_end - 0.4, ap.t_end, "oː"),  # 長母音 400ms = 2 モーラ
    ]
    region = tobi._bpm_region(phones, ap)
    assert region == pytest.approx((ap.t_end - 0.2, ap.t_end))


def test_bpm_region_survives_boundary_edit(aps_with_times):
    """語末が手修正で 30ms 詰められても最終母音を取り落とさない(重なり判定+クリップ)。"""
    ap = aps_with_times[0]
    orig_end = ap.t_end
    phones = [
        (ap.t_start, orig_end - 0.1, "t"),
        (orig_end - 0.1, orig_end, "a"),  # segments は元の境界のまま
    ]
    ap.words[-1].t_end = orig_end - 0.03  # words 層だけ 30ms 詰めた
    ap.t_end = orig_end - 0.03
    region = tobi._bpm_region(phones, ap)
    assert region == pytest.approx((orig_end - 0.1, orig_end - 0.03))


def test_parse_tolerates_drifted_bi(aps_with_times, tmp_path):
    """BI ポイントが語末から 40ms ずれていても句の分割が保たれる。"""
    from praatio import textgrid as ptg

    aps = aps_with_times
    dur = aps[-1].t_end + 0.5
    tg = ptg.Textgrid()
    word_entries = [
        (w.t_start, w.t_end, lab)
        for ap in aps
        for w, lab in zip(ap.words, tobi.nucleus_marked_words(ap))
    ]
    tg.addTier(ptg.IntervalTier("words", word_entries, 0, dur))
    drifted = [
        (t + (0.04 if lab in ("2", "3") else 0.0), lab)
        for t, lab in tobi.bi_points(aps)
    ]
    tg.addTier(ptg.PointTier("BI", drifted, 0, dur))
    path = tmp_path / "drift.TextGrid"
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)

    laps, _ = tobi.parse_xjtobi_textgrid(path)
    assert len(laps) == len(aps)
    assert [lap.kana for lap in laps] == [ap.kana for ap in aps]


def test_peak_excl_bpm_nan_without_segments(aps_with_times):
    """BPM 付きなのに分節音情報がない場合は誤値でなく NaN を返す。"""
    ap = aps_with_times[-1]
    times = _times_grid(0, ap.t_end + 0.1)
    f0 = np.full_like(times, 2.0)
    st, t = tobi.peak_excl_bpm(times, f0, [], ap, "H%")
    assert np.isnan(st) and np.isnan(t)
    # BPM なしなら segments がなくても全区間で計測できる
    st2, _ = tobi.peak_excl_bpm(times, f0, [], ap, "")
    assert st2 == pytest.approx(2.0)


def test_word_accent_rows_roundtrip(aps_with_times, tmp_path):
    """未修正なら realized == predicted で全文節 match になる。"""
    from pitchan.textproc import bunsetsu_groups

    aps = aps_with_times
    dur = aps[-1].t_end + 0.5
    phones = [(w.t_start, w.t_end, "a") for ap in aps for w in ap.words]
    times = _times_grid(0, dur)
    f0 = np.full_like(times, 1.0)
    path = tmp_path / "wa.TextGrid"
    tobi.write_xjtobi_textgrid(path, aps, phones, dur, times, f0)

    laps, _ = tobi.parse_xjtobi_textgrid(path)
    rows = tobi.word_accent_rows(laps, "wa")
    assert len(rows) == sum(len(bunsetsu_groups(ap)) for ap in aps)
    assert all(r["accent_match"] == "match" for r in rows)
    assert all(r["realized_accent"] == r["predicted_accent"] for r in rows)
    # 山梨大学で(文節)の辞書型が取得できている(単独でも有核)
    yama = next(r for r in rows if r["bunsetsu_kana"] == "ヤマナシダイガクデ")
    assert yama["lexical_accent"] is not None and yama["lexical_accent"] > 0
    assert yama["realized_accent"] == 5


def test_word_accent_rows_detects_edits(aps_with_times, tmp_path):
    """核の削除・移動の手修正が deleted / shifted として検出される。"""
    from praatio import textgrid as ptg

    aps = aps_with_times
    dur = aps[-1].t_end + 0.5
    phones = [(w.t_start, w.t_end, "a") for ap in aps for w in ap.words]
    times = _times_grid(0, dur)
    f0 = np.full_like(times, 1.0)
    path = tmp_path / "we.TextGrid"
    tobi.write_xjtobi_textgrid(path, aps, phones, dur, times, f0)

    # words 層だけ手修正した状態を作る(words_pred はそのまま)
    tg = ptg.openTextgrid(str(path), includeEmptyIntervals=False)
    entries = [(e.start, e.end, e.label) for e in tg.getTier("words").entries]
    edited = []
    for s, e, lab in entries:
        if lab == "ワタシワ'":
            lab = "ワタシワ"                    # 核の削除(平板化)
        elif lab == "ヤマナシダ'イガクデ":
            lab = "ヤマナ'シダイガクデ"          # 核の移動 5→3
        edited.append((s, e, lab))
    tg.replaceTier("words", ptg.IntervalTier("words", edited, 0, tg.maxTimestamp))
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)

    laps, _ = tobi.parse_xjtobi_textgrid(path)
    rows = {r["bunsetsu_kana"]: r for r in tobi.word_accent_rows(laps, "we")}
    assert rows["ワタシワ"]["accent_match"] == "deleted"
    assert rows["ヤマナシダイガクデ"]["accent_match"] == "shifted"
    assert rows["ヤマナシダイガクデ"]["realized_accent"] == 3
    # 核の追加(予測 0 → 実現あり)は inserted
    assert tobi._accent_match(realized=2, predicted=0) == "inserted"


def test_write_accent_textgrid(aps_with_times, tmp_path):
    from praatio import textgrid as ptg

    aps = aps_with_times
    dur = aps[-1].t_end + 0.5
    phones = [(w.t_start, w.t_end, "a") for ap in aps for w in ap.words]
    times = _times_grid(0, dur)
    f0 = np.full_like(times, 1.0)
    src = tmp_path / "a.TextGrid"
    tobi.write_xjtobi_textgrid(src, aps, phones, dur, times, f0)
    laps, _ = tobi.parse_xjtobi_textgrid(src)

    out = tmp_path / "a_accent.TextGrid"
    tobi.write_accent_textgrid(out, laps)
    tg = ptg.openTextgrid(str(out), includeEmptyIntervals=False)
    assert set(tg.tierNames) == {"words", "accent_est"}
    acc = [e.label for e in tg.getTier("accent_est").entries]
    assert "5" in acc  # ヤマナシダ'イガク
    assert all(a.isdigit() for a in acc)
