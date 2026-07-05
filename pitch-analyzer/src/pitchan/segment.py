"""単語アラインメント結果をアクセント句に割り当てる。"""

from __future__ import annotations

import logging

import numpy as np

from .textproc import AccentPhrase

logger = logging.getLogger(__name__)

# 1 モーラあたりの長さがこの範囲を外れる AP は low_confidence とする
MIN_SEC_PER_MORA = 0.03
MAX_SEC_PER_MORA = 0.5


def assign_times(
    aps: list[AccentPhrase], word_intervals: list[tuple[float, float, str]]
) -> None:
    """各 AP・単語に時刻を割り当てる(word_intervals は全 AP の単語の通し列)。"""
    n_words = sum(len(ap.words) for ap in aps)
    if n_words != len(word_intervals):
        raise ValueError(
            f"単語数が一致しません (AP 側 {n_words}, アラインメント側 {len(word_intervals)})"
        )
    i = 0
    for ap in aps:
        for w in ap.words:
            start, end, label = word_intervals[i]
            if label != w.pron:
                logger.warning(
                    "AP %d: 単語ラベル不一致 %r vs %r", ap.index, label, w.pron
                )
            w.t_start, w.t_end = start, end
            i += 1
        ap.t_start = ap.words[0].t_start
        ap.t_end = ap.words[-1].t_end
        _check_confidence(ap)


def _check_confidence(ap: AccentPhrase) -> None:
    if ap.duration is None or ap.mora_count == 0:
        ap.low_confidence = True
        return
    per_mora = ap.duration / ap.mora_count
    if not (MIN_SEC_PER_MORA <= per_mora <= MAX_SEC_PER_MORA):
        ap.low_confidence = True
        logger.warning(
            "AP %d (%s): モーラあたり %.0f ms と極端なため low_confidence",
            ap.index, ap.kana, per_mora * 1000,
        )


def flag_low_confidence_f0(
    aps: list[AccentPhrase],
    times,
    f0_hz,
    min_voiced_ratio: float = 0.5,
    octave_jump_st: float = 8.0,
    max_unvoiced_run_sec: float = 0.4,
) -> None:
    """F0 系列に基づく low_confidence の追加判定。

    次のいずれかに該当する AP を low_confidence とする(既存の判定に追加)。
    (a) 有声率 < min_voiced_ratio
    (b) 無声を挟まない隣接有声フレーム間の跳躍 |12*log2(f2/f1)| > octave_jump_st
        (倍/半ピッチ誤りの残存の疑い)
    (c) 区間内の連続無声 > max_unvoiced_run_sec
    """
    times = np.asarray(times, dtype=float)
    f0 = np.asarray(f0_hz, dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.005
    for ap in aps:
        if ap.t_start is None or ap.t_end is None:
            continue
        sel = (times >= ap.t_start) & (times <= ap.t_end)
        f = f0[sel]
        if len(f) == 0:
            continue
        voiced = f > 0
        reasons = []
        vr = float(voiced.mean())
        if vr < min_voiced_ratio:
            reasons.append(f"有声率 {vr:.2f}")
        idx = np.where(voiced)[0]
        if len(idx) >= 2:
            adj = np.diff(idx) == 1
            if adj.any():
                v = f[idx]
                jumps = np.abs(12.0 * np.log2(v[1:] / v[:-1]))[adj]
                if float(jumps.max()) > octave_jump_st:
                    reasons.append(f"跳躍 {float(jumps.max()):.1f} st")
        run = mx = 0
        for is_v in voiced:
            run = 0 if is_v else run + 1
            mx = max(mx, run)
        if mx * dt > max_unvoiced_run_sec:
            reasons.append(f"連続無声 {mx * dt:.2f} s")
        if reasons:
            if not ap.low_confidence:
                logger.warning(
                    "AP %d (%s): low_confidence (%s)",
                    ap.index, ap.kana, ", ".join(reasons),
                )
            ap.low_confidence = True
