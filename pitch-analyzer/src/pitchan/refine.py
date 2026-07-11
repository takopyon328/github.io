"""既存 TextGrid を初期値とする局所再アラインメント(pitchan refine)。

pitchan が生成した TextGrid の単語時刻を「音声の切り出し位置を決める初期値」
として使い、単語列を core block(既定 5 語)+前後 context(既定 2 語)に
分けて局所的に MFA で整列し直す。境界の移動量に応じて block 単位で
AUTO_ACCEPT / REVIEW / KEEP_ORIGINAL を判定し、入力を上書きせずに
修正版 TextGrid・差分 CSV・要約 JSON を出力する。

前提: テキストと実際の発話内容が一致した朗読音声。読み飛ばし・言い直し・
語の挿入や置換の自動処理は対象外。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf
from praatio import textgrid as ptg

from . import align, textproc

logger = logging.getLogger(__name__)

_EPS = 1e-4  # 時刻比較の許容誤差 [s]
MAX_TG_WAV_MISMATCH_SEC = 1.0  # TextGrid 最大時刻と WAV 長のずれの許容量


class RefinementStatus(str, Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    REVIEW = "REVIEW"
    KEEP_ORIGINAL = "KEEP_ORIGINAL"


@dataclass
class RefinementBlock:
    """core(採用候補)と context(端部安定用)の単語範囲。index は語の通し番号。"""

    index: int
    core_start: int  # 閉区間
    core_end: int  # 開区間
    ctx_start: int
    ctx_end: int
    slice_t0: float = 0.0  # 音声切り出し範囲(絶対時刻)
    slice_t1: float = 0.0


@dataclass
class WordCandidate:
    word_index: int
    old_start: float
    old_end: float
    cand_start: float | None = None
    cand_end: float | None = None
    applied_start: float = 0.0
    applied_end: float = 0.0


@dataclass
class BlockResult:
    block: RefinementBlock
    status: RefinementStatus
    reason: str = ""
    max_shift_ms: float = 0.0
    candidates: list[WordCandidate] = field(default_factory=list)
    cand_phones: list[tuple[float, float, str]] = field(default_factory=list)
    applied: bool = False


# ---------------------------------------------------------------------------
# 入力 TextGrid の検証
# ---------------------------------------------------------------------------


def _find_tier(names: list[str], target: str) -> str | None:
    """tier 名を完全一致優先で探す。なければ末尾一致(MFA 出力互換)。"""
    for n in names:
        if n == target:
            return n
    for n in names:
        if n.endswith(target):
            return n
    return None


def load_input_textgrid(
    path: Path, expected: list[textproc.Word], wav_dur: float
) -> tuple[
    list[tuple[float, float, str]],
    list[tuple[float, float, str]],
    list[tuple[float, float, str]],
]:
    """入力 TextGrid を検証して (words, phones, accent_phrases) を返す。

    words tier は必須。phones / accent_phrases は任意(なければ空リスト)。
    words のラベル列はテキスト解析の読み列と完全一致していなければならない。
    """
    try:
        tg = ptg.openTextgrid(str(path), includeEmptyIntervals=False)
    except Exception as e:
        raise align.AlignmentError(
            f"{path}: TextGrid を読み込めません(区間の重複・時刻の逆転・"
            f"ファイル破損の可能性): {e}"
        ) from e
    names = list(tg.tierNames)
    words_name = _find_tier(names, "words")
    if words_name is None:
        raise align.AlignmentError(
            f"{path}: words tier が見つかりません(層: {names})。"
            "pitchan が生成した TextGrid を指定してください"
        )
    words = [
        (e.start, e.end, e.label.strip())
        for e in tg.getTier(words_name).entries
        if e.label.strip()
    ]
    expected_prons = [w.pron for w in expected]
    labels = [w[2] for w in words]
    if len(labels) != len(expected_prons):
        raise align.AlignmentError(
            f"{path}: words tier の語数 ({len(labels)}) がテキストの語数 "
            f"({len(expected_prons)}) と一致しません。テキストと TextGrid の"
            "対応(同じ pitchan 実行の出力か)を確認してください"
        )
    if labels != expected_prons:
        k = next(i for i, (a, b) in enumerate(zip(labels, expected_prons)) if a != b)
        raise align.AlignmentError(
            f"{path}: words tier のラベルがテキストの読みと一致しません "
            f"(最初の不一致 位置 {k}: TextGrid={labels[k:k+3]} / "
            f"テキスト={expected_prons[k:k+3]})。テキストが変更されていないか"
            "確認してください"
        )
    prev_end = None
    for i, (s, e, lab) in enumerate(words):
        if e - s <= 0:
            raise align.AlignmentError(
                f"{path}: 語 {i} ({lab}) の区間長が 0 以下です ({s:.4f}-{e:.4f})"
            )
        if prev_end is not None and s < prev_end - _EPS:
            raise align.AlignmentError(
                f"{path}: 語 {i} ({lab}) が直前の語と重複または逆転しています "
                f"({s:.4f} < {prev_end:.4f})"
            )
        prev_end = e
    if abs(tg.maxTimestamp - wav_dur) > MAX_TG_WAV_MISMATCH_SEC:
        raise align.AlignmentError(
            f"{path}: TextGrid の最大時刻 ({tg.maxTimestamp:.2f}s) と WAV 長 "
            f"({wav_dur:.2f}s) が大きく異なります。音声と TextGrid の対応を"
            "確認してください"
        )

    def _optional(target: str) -> list[tuple[float, float, str]]:
        name = _find_tier(names, target)
        if name is None:
            return []
        return [
            (e.start, e.end, e.label)
            for e in tg.getTier(name).entries
            if e.label.strip()
        ]

    return words, _optional("phones"), _optional("accent_phrases")


# ---------------------------------------------------------------------------
# block 生成
# ---------------------------------------------------------------------------


def build_blocks(
    word_intervals: list[tuple[float, float, str]],
    core_words: int,
    context_words: int,
    margin_sec: float,
    wav_dur: float,
) -> list[RefinementBlock]:
    """単語列を重複しない core block に分け、context と音声範囲を付与する。"""
    n = len(word_intervals)
    blocks: list[RefinementBlock] = []
    for bi, cs in enumerate(range(0, n, core_words)):
        ce = min(cs + core_words, n)
        xs = max(0, cs - context_words)
        xe = min(n, ce + context_words)
        t0 = max(0.0, word_intervals[xs][0] - margin_sec)
        t1 = min(wav_dur, word_intervals[xe - 1][1] + margin_sec)
        blocks.append(RefinementBlock(bi, cs, ce, xs, xe, t0, t1))
    return blocks


# ---------------------------------------------------------------------------
# 候補の評価
# ---------------------------------------------------------------------------


def evaluate_block(
    block: RefinementBlock,
    ctx_words_abs: list[tuple[float, float, str]],
    phones_abs: list[tuple[float, float, str]],
    orig_words: list[tuple[float, float, str]],
    wav_dur: float,
    auto_accept_shift_ms: float,
    hard_max_shift_ms: float,
) -> BlockResult:
    """局所アラインメント結果(絶対時刻)から block の候補と採否を評価する。"""
    off = block.core_start - block.ctx_start
    core = ctx_words_abs[off: off + (block.core_end - block.core_start)]
    cands: list[WordCandidate] = []
    max_shift = 0.0
    reason = ""
    prev_end: float | None = None
    for k, (s, e, _lab) in enumerate(core):
        wi = block.core_start + k
        os_, oe_, _ = orig_words[wi]
        wc = WordCandidate(wi, os_, oe_, s, e)
        cands.append(wc)
        if e - s <= 0:
            reason = "invalid_duration"
        elif prev_end is not None and s < prev_end - _EPS:
            reason = "not_ordered"
        elif s < -_EPS or e > wav_dur + _EPS:
            reason = "out_of_range"
        prev_end = e
        max_shift = max(max_shift, abs(s - os_) * 1000, abs(e - oe_) * 1000)
    result = BlockResult(block, RefinementStatus.KEEP_ORIGINAL,
                         max_shift_ms=round(max_shift, 1), candidates=cands)
    if reason:
        result.reason = reason
        return result
    # core の音素: 音素区間の中点が core 候補範囲内にあるもの
    if core:
        c0, c1 = core[0][0], core[-1][1]
        result.cand_phones = [
            (s, e, lab) for s, e, lab in phones_abs if c0 <= (s + e) / 2 <= c1
        ]
    if max_shift <= auto_accept_shift_ms:
        result.status = RefinementStatus.AUTO_ACCEPT
    elif max_shift <= hard_max_shift_ms:
        result.status = RefinementStatus.REVIEW
        result.reason = "large_shift"
    else:
        result.reason = "shift_exceeds_hard_max"
    return result


def keep_original_result(
    block: RefinementBlock,
    orig_words: list[tuple[float, float, str]],
    reason: str,
) -> BlockResult:
    cands = [
        WordCandidate(wi, orig_words[wi][0], orig_words[wi][1])
        for wi in range(block.core_start, block.core_end)
    ]
    return BlockResult(block, RefinementStatus.KEEP_ORIGINAL,
                       reason=reason, candidates=cands)


# ---------------------------------------------------------------------------
# 統合
# ---------------------------------------------------------------------------


def integrate_blocks(
    results: list[BlockResult],
    orig_words: list[tuple[float, float, str]],
    orig_phones: list[tuple[float, float, str]],
    apply_review: bool,
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]] | None,
           list[str]]:
    """block の採否を確定し、全体の単語区間列と音素区間列を構築する。

    隣接 block 間で重複・逆転が生じた場合は関係する block を元に戻す。
    Returns:
        (適用後の単語区間列, 音素区間列または None, 警告リスト)
    """
    warnings: list[str] = []
    for r in results:
        r.applied = r.status == RefinementStatus.AUTO_ACCEPT or (
            apply_review and r.status == RefinementStatus.REVIEW
        )

    def _revert(r: BlockResult, why: str) -> None:
        r.applied = False
        r.status = RefinementStatus.KEEP_ORIGINAL
        r.reason = (r.reason + ";" if r.reason else "") + why
        warnings.append(f"block {r.block.index}: {why} のため元の境界に戻しました")

    # applied に応じた各語の時刻を確定 → 隣接 joint(単語・音素)の検証 →
    # 破綻があれば関係する block を巻き戻して再検証
    for _ in range(len(results) + 1):
        for r in results:
            for wc in r.candidates:
                if r.applied and wc.cand_start is not None:
                    wc.applied_start, wc.applied_end = wc.cand_start, wc.cand_end
                else:
                    wc.applied_start, wc.applied_end = wc.old_start, wc.old_end
        conflict = False
        for prev, nxt in zip(results, results[1:]):
            if not prev.candidates or not nxt.candidates:
                continue
            joint_bad = (
                prev.candidates[-1].applied_end
                > nxt.candidates[0].applied_start + _EPS
            )
            if not joint_bad:
                pp = _block_phones(prev, orig_words, orig_phones)
                np_ = _block_phones(nxt, orig_words, orig_phones)
                if pp and np_ and pp[-1][1] > np_[0][0] + _EPS:
                    joint_bad = True
            if joint_bad:
                for r in (prev, nxt):
                    if r.applied:
                        _revert(r, "integration_overlap")
                conflict = True
        if not conflict:
            break
    else:  # 収束しなかった場合の安全側フォールバック
        for r in results:
            if r.applied:
                _revert(r, "integration_failed")
        for r in results:
            for wc in r.candidates:
                wc.applied_start, wc.applied_end = wc.old_start, wc.old_end
        warnings.append("統合検証に失敗したため全 block を元の境界に戻しました")

    applied_words = [
        (wc.applied_start, wc.applied_end, orig_words[wc.word_index][2])
        for r in results
        for wc in r.candidates
    ]
    # 最終検証(時刻順・正の区間長)
    prev_end = None
    for s, e, lab in applied_words:
        if e - s <= 0 or (prev_end is not None and s < prev_end - _EPS):
            for r in results:
                if r.applied:
                    _revert(r, "final_validation_failed")
            applied_words = [
                (orig_words[i][0], orig_words[i][1], orig_words[i][2])
                for i in range(len(orig_words))
            ]
            break
        prev_end = e

    phones = _build_phones(results, orig_words, orig_phones, warnings)
    return applied_words, phones, warnings


def _block_orig_span(r: BlockResult, orig_words) -> tuple[float, float]:
    return (orig_words[r.block.core_start][0], orig_words[r.block.core_end - 1][1])


def _block_phones(
    r: BlockResult,
    orig_words: list[tuple[float, float, str]],
    orig_phones: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]] | None:
    """block の適用状態に応じた音素区間列を返す。得られない場合は None。"""
    if r.applied:
        return r.cand_phones
    if not orig_phones:
        return None
    s0, s1 = _block_orig_span(r, orig_words)
    return [
        p for p in orig_phones if s0 - _EPS <= (p[0] + p[1]) / 2 <= s1 + _EPS
    ]


def _build_phones(
    results: list[BlockResult],
    orig_words: list[tuple[float, float, str]],
    orig_phones: list[tuple[float, float, str]],
    warnings: list[str],
) -> list[tuple[float, float, str]] | None:
    """適用状態に応じた phones 列を構築する。一貫性がなければ None。"""
    phones: list[tuple[float, float, str]] = []
    for r in results:
        bp = _block_phones(r, orig_words, orig_phones)
        if bp is None:
            warnings.append(
                f"block {r.block.index} が適用されず入力に phones tier もないため、"
                "一貫した phones tier を構築できません(phones tier は出力しません)"
            )
            return None
        phones.extend(bp)
    prev_end = None
    for s, e, lab in phones:
        if e - s <= 0 or (prev_end is not None and s < prev_end - _EPS):
            warnings.append(
                "block 間で音素区間の重複・逆転が解消できなかったため phones tier"
                "を出力しません(words の採否には影響しません)"
            )
            return None
        prev_end = e
    return phones


# ---------------------------------------------------------------------------
# オーケストレーション
# ---------------------------------------------------------------------------


def run_refine(
    wav_path: Path,
    text_path: Path,
    textgrid_path: Path,
    out_dir: Path,
    speaker: str = "spk",
    core_words: int = 5,
    context_words: int = 2,
    margin_sec: float = 0.30,
    auto_accept_shift_ms: float = 80.0,
    hard_max_shift_ms: float = 250.0,
    apply_review: bool = False,
    fine_tune_boundary_tolerance: float | None = None,
    jobs: int = 4,
    beam: int = 100,
    retry_beam: int = 400,
    acoustic_model: str = "japanese_mfa",
    g2p_model: str = "japanese_mfa",
) -> dict:
    """局所再アラインメントを実行し、要約 dict を返す。入力は上書きしない。"""
    options = {k: (str(v) if isinstance(v, Path) else v) for k, v in locals().items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work" / "refine"

    # (1) テキスト解析と入力 TextGrid の検証
    aps = textproc.analyze_text_file(str(text_path))
    if not aps:
        raise align.AlignmentError(f"{text_path}: アクセント句が得られませんでした")
    flat: list[tuple[int, int, textproc.Word]] = [
        (ai, wi, w) for ai, ap in enumerate(aps) for wi, w in enumerate(ap.words)
    ]
    words_flat = [w for _, _, w in flat]
    x, sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    wav_dur = len(x) / sr
    orig_words, orig_phones, orig_aps_tier = load_input_textgrid(
        textgrid_path, words_flat, wav_dur
    )

    # (2) block 生成と局所コーパスの作成
    blocks = build_blocks(orig_words, core_words, context_words, margin_sec, wav_dur)
    logger.info(
        "%s: %d 語 → %d block (core=%d, context=%d, margin=%.2fs)",
        textgrid_path.name, len(orig_words), len(blocks),
        core_words, context_words, margin_sec,
    )
    chunks_dir = work / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    prons = [w[2] for w in orig_words]
    items: list[align.FileItem] = []
    for b in blocks:
        i0, i1 = int(round(b.slice_t0 * sr)), int(round(b.slice_t1 * sr))
        b.slice_t0 = i0 / sr  # 絶対時刻への復元をサンプル境界に揃える
        chunk = chunks_dir / f"{textgrid_path.stem}_b{b.index:04d}.wav"
        sf.write(chunk, x[i0:i1], sr)
        items.append(
            align.FileItem(speaker, chunk.stem, chunk, prons[b.ctx_start: b.ctx_end])
        )

    # (3) 一括 MFA(refine では fine_tune を既定で使用)
    align.check_mfa_available()
    corpus_dir = work / "corpus"
    aligned_dir = work / "aligned"
    align.prepare_corpus(items, corpus_dir)
    logger.info("発音辞書を生成中 (mfa g2p)...")
    dict_path = align.build_g2p_dictionary(items, work, g2p_model)
    logger.info("局所再アラインメント中 (mfa align --fine_tune, %d block)...", len(blocks))
    align.run_align(
        corpus_dir, dict_path, aligned_dir,
        acoustic_model=acoustic_model, beam=beam, retry_beam=retry_beam,
        num_jobs=jobs, fine_tune=True,
        fine_tune_boundary_tolerance=fine_tune_boundary_tolerance,
    )

    # (4) block ごとの評価
    results: list[BlockResult] = []
    for b, item in zip(blocks, items):
        tg_path = aligned_dir / speaker / f"{item.name}.TextGrid"
        try:
            if not tg_path.exists():
                raise align.AlignmentError("TextGrid が生成されていません (mfa_failed)")
            local_words, local_phones = align.read_word_intervals(
                tg_path, item.tokens
            )
        except align.AlignmentError as e:
            reason = "mfa_failed" if "生成されていません" in str(e) else "label_mismatch"
            logger.warning("block %d: %s (%s)", b.index, reason, e)
            results.append(keep_original_result(b, orig_words, reason))
            continue
        off = b.slice_t0
        results.append(
            evaluate_block(
                b,
                [(s + off, e + off, lab) for s, e, lab in local_words],
                [(s + off, e + off, lab) for s, e, lab in local_phones],
                orig_words, wav_dur, auto_accept_shift_ms, hard_max_shift_ms,
            )
        )

    # (5) 統合と出力
    applied_words, phones, warnings = integrate_blocks(
        results, orig_words, orig_phones, apply_review
    )
    stem = textgrid_path.stem
    _write_refined_textgrid(
        out_dir / f"{stem}_refined.TextGrid",
        aps, flat, applied_words, phones, results,
        orig_words, orig_phones, orig_aps_tier, wav_dur,
    )
    _write_diff_csv(
        out_dir / f"{stem}_alignment_diff.csv", stem, flat, results
    )
    analysis = sorted(aligned_dir.rglob("alignment_analysis.csv"))
    summary = _build_summary(
        options, results, warnings, phones is not None,
        str(analysis[0]) if analysis else None,
    )
    (out_dir / f"{stem}_refine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info(
        "完了: %d block 中 AUTO_ACCEPT %d / REVIEW %d / KEEP_ORIGINAL %d "
        "(変更された語 %d/%d)",
        summary["n_blocks"], summary["n_auto_accept"], summary["n_review"],
        summary["n_keep_original"], summary["n_words_changed"], summary["n_words"],
    )
    for w in warnings:
        logger.warning("%s", w)
    return summary


def _write_refined_textgrid(
    path: Path,
    aps: list[textproc.AccentPhrase],
    flat: list[tuple[int, int, textproc.Word]],
    applied_words: list[tuple[float, float, str]],
    phones: list[tuple[float, float, str]] | None,
    results: list[BlockResult],
    orig_words: list[tuple[float, float, str]],
    orig_phones: list[tuple[float, float, str]],
    orig_aps_tier: list[tuple[float, float, str]],
    wav_dur: float,
) -> None:
    duration = max(wav_dur, orig_words[-1][1], applied_words[-1][1])
    # アクセント句の再計算(句頭単語の開始〜句末単語の終了)
    ap_entries = []
    pos = 0
    for ap in aps:
        n = len(ap.words)
        s = applied_words[pos][0]
        e = applied_words[pos + n - 1][1]
        acc = ap.accent_type if ap.accent_type is not None else "?"
        ap_entries.append((s, e, f"{ap.kana}({acc})"))
        pos += n

    tg = ptg.Textgrid()
    tg.addTier(ptg.IntervalTier("accent_phrases", ap_entries, 0, duration))
    tg.addTier(ptg.IntervalTier("words", applied_words, 0, duration))
    if phones is not None:
        tg.addTier(ptg.IntervalTier("phones", phones, 0, duration))
    if orig_aps_tier:
        tg.addTier(
            ptg.IntervalTier("accent_phrases_original", orig_aps_tier, 0, duration)
        )
    tg.addTier(ptg.IntervalTier("words_original", orig_words, 0, duration))
    if orig_phones:
        tg.addTier(ptg.IntervalTier("phones_original", orig_phones, 0, duration))
    review_entries = [
        (
            *_block_orig_span(r, orig_words),
            f"{r.status.value}|block={r.block.index}"
            f"|max_shift_ms={r.max_shift_ms:.1f}|reason={r.reason}",
        )
        for r in results
        if r.status != RefinementStatus.AUTO_ACCEPT
    ]
    tg.addTier(ptg.IntervalTier("alignment_review", review_entries, 0, duration))
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)


def _write_diff_csv(
    path: Path,
    file_stem: str,
    flat: list[tuple[int, int, textproc.Word]],
    results: list[BlockResult],
) -> None:
    import csv

    cols = [
        "file", "block_index", "ap_index", "word_index", "surface", "pron",
        "old_start", "candidate_start", "applied_start", "start_shift_ms",
        "old_end", "candidate_end", "applied_end", "end_shift_ms",
        "max_block_shift_ms", "status", "applied", "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in results:
            for wc in r.candidates:
                ai, wi, w = flat[wc.word_index]
                writer.writerow({
                    "file": file_stem,
                    "block_index": r.block.index,
                    "ap_index": ai,
                    "word_index": wi,
                    "surface": w.surface,
                    "pron": w.pron,
                    "old_start": round(wc.old_start, 4),
                    "candidate_start": (
                        round(wc.cand_start, 4) if wc.cand_start is not None else ""
                    ),
                    "applied_start": round(wc.applied_start, 4),
                    "start_shift_ms": round(
                        (wc.applied_start - wc.old_start) * 1000, 1
                    ),
                    "old_end": round(wc.old_end, 4),
                    "candidate_end": (
                        round(wc.cand_end, 4) if wc.cand_end is not None else ""
                    ),
                    "applied_end": round(wc.applied_end, 4),
                    "end_shift_ms": round((wc.applied_end - wc.old_end) * 1000, 1),
                    "max_block_shift_ms": r.max_shift_ms,
                    "status": r.status.value,
                    "applied": r.applied,
                    "reason": r.reason,
                })


def _build_summary(
    options: dict,
    results: list[BlockResult],
    warnings: list[str],
    phones_written: bool,
    alignment_analysis: str | None,
) -> dict:
    applied_shifts = [
        abs(d) * 1000
        for r in results
        if r.applied
        for wc in r.candidates
        for d in (wc.applied_start - wc.old_start, wc.applied_end - wc.old_end)
    ]
    n_changed = sum(
        1
        for r in results
        for wc in r.candidates
        if abs(wc.applied_start - wc.old_start) > _EPS
        or abs(wc.applied_end - wc.old_end) > _EPS
    )
    counts = {s: sum(1 for r in results if r.status == s) for s in RefinementStatus}
    return {
        "input_wav": options["wav_path"],
        "input_text": options["text_path"],
        "input_textgrid": options["textgrid_path"],
        "options": options,
        "n_words": sum(len(r.candidates) for r in results),
        "n_blocks": len(results),
        "n_auto_accept": counts[RefinementStatus.AUTO_ACCEPT],
        "n_review": counts[RefinementStatus.REVIEW],
        "n_keep_original": counts[RefinementStatus.KEEP_ORIGINAL],
        "n_words_changed": n_changed,
        "applied_shift_ms_median": (
            round(float(np.median(applied_shifts)), 1) if applied_shifts else 0.0
        ),
        "applied_shift_ms_p90": (
            round(float(np.percentile(applied_shifts, 90)), 1)
            if applied_shifts else 0.0
        ),
        "fine_tune": True,
        "fine_tune_boundary_tolerance": options["fine_tune_boundary_tolerance"],
        "phones_tier_written": phones_written,
        "alignment_analysis_csv": alignment_analysis,
        "warnings": warnings,
    }
