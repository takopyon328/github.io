"""refine(局所再アラインメント)のテスト。

e2e は偽 mfa(1 語 0.3 秒、チャンク先頭から 0.5 秒開始)を使う。
元の単語区間を「0.5 秒から 0.3 秒刻みで連続」に作っておくと、
候補の移動量は一様に (0.5 - margin_sec) 秒になり、margin の選び方だけで
AUTO_ACCEPT / REVIEW / KEEP_ORIGINAL を決定的に作り分けられる。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from pitchan import align, outputs, refine, segment
from pitchan.cli import main
from pitchan.refine import RefinementStatus
from pitchan.textproc import analyze_text_file

TEXT = "私は山梨大学で、音声を研究しています。"
WORD_DUR = 0.3
FIRST_START = 0.5


# ---------------------------------------------------------------------------
# run_align の fine-tune 引数(テスト 1-3)
# ---------------------------------------------------------------------------


def _capture_run_align(tmp_path, monkeypatch, **kwargs):
    captured = {}
    monkeypatch.setattr(
        align, "_run_mfa_command", lambda cmd: captured.setdefault("cmd", cmd)
    )
    align.run_align(
        tmp_path / "corpus", tmp_path / "dict.txt", tmp_path / "aligned", **kwargs
    )
    return captured["cmd"]


def test_run_align_no_fine_tune_by_default(tmp_path, monkeypatch):
    cmd = _capture_run_align(tmp_path, monkeypatch)
    assert "--fine_tune" not in cmd
    assert "--fine_tune_boundary_tolerance" not in cmd


def test_run_align_fine_tune_flag(tmp_path, monkeypatch):
    cmd = _capture_run_align(tmp_path, monkeypatch, fine_tune=True)
    assert "--fine_tune" in cmd
    assert "--fine_tune_boundary_tolerance" not in cmd
    # 位置引数(コーパス等)より後に付いている(現行方針)
    assert cmd.index("--fine_tune") > cmd.index(str(tmp_path / "corpus"))


def test_run_align_boundary_tolerance(tmp_path, monkeypatch):
    cmd = _capture_run_align(
        tmp_path, monkeypatch, fine_tune=True, fine_tune_boundary_tolerance=0.01
    )
    i = cmd.index("--fine_tune_boundary_tolerance")
    assert cmd[i + 1] == "0.01"


# ---------------------------------------------------------------------------
# block 生成(テスト 4-5)
# ---------------------------------------------------------------------------


def _uniform_words(n):
    return [
        (FIRST_START + WORD_DUR * k, FIRST_START + WORD_DUR * (k + 1), f"W{k}")
        for k in range(n)
    ]


def test_build_blocks_covers_each_word_once():
    words = _uniform_words(12)
    blocks = refine.build_blocks(words, 5, 2, 0.3, 10.0)
    covered = [i for b in blocks for i in range(b.core_start, b.core_end)]
    assert covered == list(range(12))  # ちょうど 1 回ずつ、順序どおり
    # context は core を含み、前後 2 語まで
    for b in blocks:
        assert b.ctx_start == max(0, b.core_start - 2)
        assert b.ctx_end == min(12, b.core_end + 2)


def test_build_blocks_within_audio_range():
    words = _uniform_words(7)
    wav_dur = words[-1][1] + 0.1
    blocks = refine.build_blocks(words, 5, 2, 1.0, wav_dur)
    for b in blocks:
        assert 0.0 <= b.slice_t0 < b.slice_t1 <= wav_dur


# ---------------------------------------------------------------------------
# 採否判定(テスト 6, 7, 9)と統合(8, 12)
# ---------------------------------------------------------------------------


def _make_block_and_cands(orig_words, shift_sec):
    block = refine.build_blocks(orig_words, len(orig_words), 0, 0.3, 100.0)[0]
    ctx_abs = [(s + shift_sec, e + shift_sec, lab) for s, e, lab in orig_words]
    return block, ctx_abs


def test_evaluate_auto_accept():
    words = _uniform_words(5)
    block, ctx = _make_block_and_cands(words, 0.02)  # 20ms
    r = refine.evaluate_block(block, ctx, [], words, 100.0, 80.0, 250.0)
    assert r.status == RefinementStatus.AUTO_ACCEPT
    assert r.max_shift_ms == pytest.approx(20.0, abs=0.2)


def test_evaluate_review():
    words = _uniform_words(5)
    block, ctx = _make_block_and_cands(words, 0.15)  # 150ms
    r = refine.evaluate_block(block, ctx, [], words, 100.0, 80.0, 250.0)
    assert r.status == RefinementStatus.REVIEW
    assert r.reason == "large_shift"


def test_evaluate_exceeds_hard_max():
    words = _uniform_words(5)
    block, ctx = _make_block_and_cands(words, 0.3)  # 300ms
    r = refine.evaluate_block(block, ctx, [], words, 100.0, 80.0, 250.0)
    assert r.status == RefinementStatus.KEEP_ORIGINAL
    assert r.reason == "shift_exceeds_hard_max"


def test_evaluate_rejects_disordered_candidates():
    words = _uniform_words(3)
    block = refine.build_blocks(words, 3, 0, 0.3, 100.0)[0]
    ctx = [words[0], (words[1][1], words[1][0] + 0.0, "x"), words[2]]  # 区間長 0
    r = refine.evaluate_block(block, ctx, [], words, 100.0, 80.0, 250.0)
    assert r.status == RefinementStatus.KEEP_ORIGINAL
    assert r.reason == "invalid_duration"


def test_integrate_review_kept_by_default_and_applied_with_flag():
    words = _uniform_words(5)
    block, ctx = _make_block_and_cands(words, 0.15)
    r = refine.evaluate_block(block, ctx, [], words, 100.0, 80.0, 250.0)

    applied, _, _ = refine.integrate_blocks([r], words, [], apply_review=False)
    assert applied[0][0] == pytest.approx(words[0][0])  # 元の境界を維持

    r2 = refine.evaluate_block(block, ctx, [], words, 100.0, 80.0, 250.0)
    applied2, _, _ = refine.integrate_blocks([r2], words, [], apply_review=True)
    assert applied2[0][0] == pytest.approx(words[0][0] + 0.15)  # 候補を適用
    assert r2.status == RefinementStatus.REVIEW  # status は REVIEW のまま


def test_integrate_reverts_failed_block_only():
    """局所 MFA 失敗 block は元境界に戻り、他 block の適用は維持される。

    block0 の候補は左へ 20ms(隣接 block1 の元境界とは重複しない方向)。
    """
    words = _uniform_words(10)
    blocks = refine.build_blocks(words, 5, 0, 0.3, 100.0)
    ctx0 = [(s - 0.02, e - 0.02, lab) for s, e, lab in words[0:5]]
    r0 = refine.evaluate_block(blocks[0], ctx0, [], words, 100.0, 80.0, 250.0)
    r1 = refine.keep_original_result(blocks[1], words, "mfa_failed")
    applied, phones, _ = refine.integrate_blocks(
        [r0, r1], words, [], apply_review=False
    )
    assert applied[0][0] == pytest.approx(words[0][0] - 0.02)   # block0 適用
    assert applied[5][0] == pytest.approx(words[5][0])          # block1 は元のまま
    assert r0.status == RefinementStatus.AUTO_ACCEPT
    assert r1.status == RefinementStatus.KEEP_ORIGINAL
    assert phones is None  # 入力に phones がなく失敗 block があるため


def test_integrate_reverts_on_joint_overlap():
    """隣接 block と重複する候補は機械的に按分せず、block ごと元へ戻す。"""
    words = _uniform_words(10)
    blocks = refine.build_blocks(words, 5, 0, 0.3, 100.0)
    # block0 の候補を右へ 60ms ずらして block1 先頭(元位置)と重複させる
    ctx0 = [(s + 0.06, e + 0.06, lab) for s, e, lab in words[0:5]]
    r0 = refine.evaluate_block(blocks[0], ctx0, [], words, 100.0, 80.0, 250.0)
    assert r0.status == RefinementStatus.AUTO_ACCEPT
    r1 = refine.keep_original_result(blocks[1], words, "mfa_failed")
    applied, _, warnings = refine.integrate_blocks(
        [r0, r1], words, [], apply_review=False
    )
    assert r0.status == RefinementStatus.KEEP_ORIGINAL
    assert "integration_overlap" in r0.reason
    assert applied[4][1] == pytest.approx(words[4][1])  # 元の境界
    assert warnings


# ---------------------------------------------------------------------------
# 入力検証(テスト 10, 11)
# ---------------------------------------------------------------------------


def _write_input_textgrid(tmp_path, name="in.TextGrid"):
    txt = tmp_path / "s.txt"
    txt.write_text(TEXT, encoding="utf-8")
    aps = analyze_text_file(str(txt))
    t = FIRST_START
    intervals = []
    for ap in aps:
        for w in ap.words:
            intervals.append((t, t + WORD_DUR, w.pron))
            t += WORD_DUR
    segment.assign_times(aps, intervals)
    dur = t + 0.2
    tg = tmp_path / name
    outputs.write_textgrid(tg, aps, [], dur)
    return txt, tg, aps, dur


def test_load_input_textgrid_label_mismatch(tmp_path):
    txt, tg, aps, dur = _write_input_textgrid(tmp_path)
    other = tmp_path / "other.txt"
    other.write_text("今日は良い天気です。", encoding="utf-8")
    other_words = [w for ap in analyze_text_file(str(other)) for w in ap.words]
    with pytest.raises(align.AlignmentError, match="語数|一致しません"):
        refine.load_input_textgrid(tg, other_words, dur)


_OVERLAP_TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2
        intervals: size = 2
        intervals [1]:
            xmin = 0.5
            xmax = 0.9
            text = "ワタシ"
        intervals [2]:
            xmin = 0.8
            xmax = 1.2
            text = "ワ"
'''


def test_load_input_textgrid_rejects_overlap(tmp_path):
    """単語区間が重複する TextGrid は明確な AlignmentError で拒否される。"""
    txt = tmp_path / "s.txt"
    txt.write_text(TEXT, encoding="utf-8")
    words = [w for ap in analyze_text_file(str(txt)) for w in ap.words]
    path = tmp_path / "overlap.TextGrid"
    path.write_text(_OVERLAP_TG, encoding="utf-8")
    with pytest.raises(align.AlignmentError, match="重複|読み込めません"):
        refine.load_input_textgrid(path, words, 2.0)


def test_load_input_textgrid_requires_words_tier(tmp_path):
    from praatio import textgrid as ptg

    txt = tmp_path / "s.txt"
    txt.write_text(TEXT, encoding="utf-8")
    aps = analyze_text_file(str(txt))
    words = [w for ap in aps for w in ap.words]
    tg = ptg.Textgrid()
    tg.addTier(ptg.IntervalTier("misc", [(0.0, 1.0, "x")], 0, 2.0))
    path = tmp_path / "nowords.TextGrid"
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)
    with pytest.raises(align.AlignmentError, match="words tier"):
        refine.load_input_textgrid(path, words, 2.0)


def test_find_tier_prefers_exact_match():
    names = ["words_original", "words", "phones_original"]
    assert refine._find_tier(names, "words") == "words"
    # 完全一致がない場合のみ末尾一致(words_original は words に一致しない)
    assert refine._find_tier(["sample - words"], "words") == "sample - words"
    assert refine._find_tier(["words_original"], "words") is None


# ---------------------------------------------------------------------------
# e2e(テスト 13-15, 17)
# ---------------------------------------------------------------------------


def _prepare_refine_inputs(tmp_path):
    txt, tg, aps, dur = _write_input_textgrid(tmp_path)
    wav = tmp_path / "in.wav"
    sr = 16000
    x = np.zeros(int(sr * dur))
    x[:: int(sr / 150)] = 0.5
    sf.write(wav, x, sr)
    return txt, tg, wav, aps, dur


def _run_refine_cli(tmp_path, fake_mfa_on_path, *extra):
    txt, tg, wav, aps, dur = _prepare_refine_inputs(tmp_path)
    out = tmp_path / "out"
    rc = main([
        "refine", "--wav", str(wav), "--text", str(txt),
        "--textgrid", str(tg), "--out", str(out), *extra,
    ])
    assert rc == 0
    return txt, tg, wav, aps, out


def test_e2e_refine_auto_accept(tmp_path, fake_mfa_on_path):
    """margin=0.5 → 偽 mfa の出力が元境界と一致し、全 block AUTO_ACCEPT。"""
    from praatio import textgrid as ptg

    tg_before = None
    txt, tg, wav, aps, out = _run_refine_cli(
        tmp_path, fake_mfa_on_path, "--margin-sec", "0.5"
    )
    stem = tg.stem
    refined = out / f"{stem}_refined.TextGrid"
    assert refined.exists()

    diff = pd.read_csv(out / f"{stem}_alignment_diff.csv")
    n_words = sum(len(ap.words) for ap in aps)
    assert len(diff) == n_words
    assert (diff["status"] == "AUTO_ACCEPT").all()
    assert diff["start_shift_ms"].abs().max() < 1.0  # 実質移動なし

    out_tg = ptg.openTextgrid(str(refined), includeEmptyIntervals=False)
    names = set(out_tg.tierNames)
    assert {"accent_phrases", "words", "phones",
            "accent_phrases_original", "words_original",
            "alignment_review"} <= names
    # 入力に phones tier はないが、全 block 成功なので phones が生成される
    assert "phones" in names and "phones_original" not in names

    import json
    summary = json.loads((out / f"{stem}_refine_summary.json").read_text("utf-8"))
    assert summary["n_auto_accept"] == summary["n_blocks"]
    assert summary["fine_tune"] is True


def test_e2e_refine_review_default_keeps_original(tmp_path, fake_mfa_on_path):
    """margin=0.3 → 一様 +200ms の候補は REVIEW となり、既定では元境界を維持。"""
    txt, tg, wav, aps, out = _run_refine_cli(tmp_path, fake_mfa_on_path)
    stem = tg.stem
    diff = pd.read_csv(out / f"{stem}_alignment_diff.csv")
    assert (diff["status"] == "REVIEW").all()
    assert (~diff["applied"]).all()
    # 候補は保存され、適用時刻は元のまま
    assert diff["candidate_start"].notna().all()
    assert (diff["applied_start"] == diff["old_start"]).all()
    cand_shift = (diff["candidate_start"] - diff["old_start"]) * 1000
    assert cand_shift.max() == pytest.approx(200.0, abs=2.0)
    # phones: 入力になく、適用もされないため出力されない(警告あり)
    from praatio import textgrid as ptg
    out_tg = ptg.openTextgrid(
        str(out / f"{stem}_refined.TextGrid"), includeEmptyIntervals=False
    )
    assert "phones" not in out_tg.tierNames
    review = [e.label for e in out_tg.getTier("alignment_review").entries]
    assert review and all(lab.startswith("REVIEW|block=") for lab in review)
    assert all("max_shift_ms=" in lab and "reason=" in lab for lab in review)


def test_e2e_refine_apply_review(tmp_path, fake_mfa_on_path):
    txt, tg, wav, aps, out = _run_refine_cli(
        tmp_path, fake_mfa_on_path, "--apply-review"
    )
    diff = pd.read_csv(out / f"{tg.stem}_alignment_diff.csv")
    assert (diff["status"] == "REVIEW").all()
    assert diff["applied"].all()
    assert (diff["applied_start"] == diff["candidate_start"]).all()


def test_e2e_refine_hard_max(tmp_path, fake_mfa_on_path):
    txt, tg, wav, aps, out = _run_refine_cli(
        tmp_path, fake_mfa_on_path, "--hard-max-shift-ms", "100"
    )
    diff = pd.read_csv(out / f"{tg.stem}_alignment_diff.csv")
    assert (diff["status"] == "KEEP_ORIGINAL").all()
    assert (diff["reason"] == "shift_exceeds_hard_max").all()
    assert (diff["applied_start"] == diff["old_start"]).all()


def test_e2e_refine_does_not_overwrite_input(tmp_path, fake_mfa_on_path):
    txt, tg, wav, aps, dur = _prepare_refine_inputs(tmp_path)
    before = tg.read_bytes()
    out = tmp_path / "out"
    rc = main([
        "refine", "--wav", str(wav), "--text", str(txt),
        "--textgrid", str(tg), "--out", str(out), "--margin-sec", "0.5",
    ])
    assert rc == 0
    assert tg.read_bytes() == before  # 入力は変更されない


def test_e2e_refined_textgrid_readable(tmp_path, fake_mfa_on_path):
    """出力 TextGrid を read_word_intervals で読み戻せる(テスト 15)。"""
    txt, tg, wav, aps, out = _run_refine_cli(
        tmp_path, fake_mfa_on_path, "--margin-sec", "0.5"
    )
    tokens = [w.pron for ap in aps for w in ap.words]
    words, phones = align.read_word_intervals(
        out / f"{tg.stem}_refined.TextGrid", tokens
    )
    assert [w[2] for w in words] == tokens
    assert phones  # phones tier も読める
