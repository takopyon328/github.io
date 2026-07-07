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
