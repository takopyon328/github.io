"""文単位分割(split モジュール)のテスト。"""

import numpy as np
import pytest

from pitchan import split
from pitchan.textproc import analyze_text_file

TEXT2 = "私は山梨大学で音声を研究しています。今日は良い天気です。"


@pytest.fixture
def aps2(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(TEXT2, encoding="utf-8")
    return analyze_text_file(str(p))


def test_sentence_index(aps2):
    idx = sorted({ap.sentence_index for ap in aps2})
    assert idx == [0, 1]
    # 2 文目の先頭はポーズ直後
    first_of_s1 = next(ap for ap in aps2 if ap.sentence_index == 1)
    assert first_of_s1.follows_pause


def _speech(sr, dur, f0=150.0, amp=0.5):
    x = np.zeros(int(sr * dur))
    x[:: int(sr / f0)] = amp
    return x


def test_detect_silences():
    sr = 16000
    x = np.concatenate([
        _speech(sr, 2.0), np.zeros(int(sr * 0.6)), _speech(sr, 1.5),
    ])
    sil = split.detect_silences(x, sr)
    assert len(sil) == 1
    s0, s1 = sil[0]
    assert 1.8 < s0 < 2.2 and 2.4 < s1 < 2.8
    # 短い無音(0.1 秒)は検出しない
    x2 = np.concatenate([
        _speech(sr, 1.0), np.zeros(int(sr * 0.1)), _speech(sr, 1.0),
    ])
    assert split.detect_silences(x2, sr) == []


def test_plan_utterances_split(aps2):
    # 文のモーラ比(约 24:11)に応じた位置に無音がある → 2 発話に分割
    moras = [
        sum(a.mora_count for a in aps2 if a.sentence_index == s) for s in (0, 1)
    ]
    total = 10.0
    boundary = total * moras[0] / sum(moras)
    utts = split.plan_utterances(aps2, [(boundary - 0.2, boundary + 0.2)], total)
    assert len(utts) == 2
    assert utts[0].t_end == pytest.approx(boundary)
    assert {a.sentence_index for a in utts[0].aps} == {0}
    assert {a.sentence_index for a in utts[1].aps} == {1}


def test_plan_utterances_merge_when_no_silence(aps2):
    # 対応する無音がなければ 1 発話にまとめる
    utts = split.plan_utterances(aps2, [], 10.0)
    assert len(utts) == 1
    assert len(utts[0].aps) == len(aps2)
    # 見当違いの位置(境界予測から 3 秒超ずれ)の無音too → 分割しない
    utts2 = split.plan_utterances(aps2, [(0.1, 0.5)], 30.0)
    assert len(utts2) == 1


def test_plan_utterances_speech_time_robust_to_long_leading_silence(aps2):
    """冒頭に長い無音があっても、発話時間座標での対応付けは狂わない。

    実時間の比例配分(旧方式)では冒頭無音が予測位置を大きく歪めるケース。
    """
    # モーラ比 27:10、発話部分は 0.15 s/モーラ、冒頭に 15 秒の無音
    silences = [(0.0, 15.0), (19.05, 19.55)]
    total = 21.05
    utts = split.plan_utterances(aps2, silences, total)
    assert len(utts) == 2
    assert utts[0].t_end == pytest.approx(19.3, abs=0.01)  # 無音の中央


def test_plan_utterances_ignores_short_comma_pause(aps2):
    """短い読点ポーズ(<0.4 秒)は文境界の候補にしない。"""
    comma = (3.3, 3.65)      # 0.35 秒(読点相当)
    boundary = (4.2, 4.7)    # 0.5 秒(文末相当)
    utts = split.plan_utterances(aps2, [comma, boundary], 7.5)
    assert len(utts) == 2
    assert utts[0].t_end == pytest.approx(4.45, abs=0.01)


def test_plan_utterances_falls_back_when_plan_implausible(aps2):
    """分割結果のモーラあたり時間が不自然なら 1 発話へフォールバックする。"""
    # 総発話時間 1.5 秒に 37 モーラ(40ms/モーラ)は不自然に速い
    utts = split.plan_utterances(aps2, [(0.95, 1.45)], 2.0)
    assert len(utts) == 1
    assert len(utts[0].aps) == len(aps2)


def test_collect_split_alignment_rejects_implausible_alignment(tmp_path):
    """誤ったチャンクへの強制整列(不自然な語長)は成功として通さない。"""
    from praatio import textgrid as ptg

    from pitchan.cli import Pair, _collect_split_alignment

    txt = tmp_path / "t.txt"
    txt.write_text(TEXT2, encoding="utf-8")
    aps = analyze_text_file(str(txt))
    utt = split.Utterance(0, 0.0, 10.0, list(aps))
    pair = Pair("spk", "t", tmp_path / "t.wav", txt)

    aligned = tmp_path / "aligned" / "spk"
    aligned.mkdir(parents=True)
    # 全語 0.02 秒 → モーラあたり約 9 ms(< 30ms)の不自然な整列結果
    tokens = [w.pron for ap in aps for w in ap.words]
    entries = [(0.5 + 0.02 * k, 0.5 + 0.02 * (k + 1), tok)
               for k, tok in enumerate(tokens)]
    tg = ptg.Textgrid()
    tg.addTier(ptg.IntervalTier("words", entries, 0, 10.0))
    tg.addTier(ptg.IntervalTier("phones", entries, 0, 10.0))
    tg.save(str(aligned / "t_u000.TextGrid"),
            format="long_textgrid", includeBlankSpaces=True)

    phones = _collect_split_alignment(pair, [utt], tmp_path / "aligned")
    assert phones == []                       # 音素は採用されない
    assert all(ap.t_start is None for ap in aps)   # 時刻は付与されない
    assert all(ap.low_confidence for ap in aps)    # 要確認として通知
