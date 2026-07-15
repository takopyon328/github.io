"""F0 抽出(WORLD harvest + stonemask)。"""

from __future__ import annotations

import logging

import numpy as np
import pyworld
import soundfile as sf

logger = logging.getLogger(__name__)

# 探索範囲外の値を無声化する際のガード帯(倍率)。境界近傍の真の値
# (句末の下降や強調のピーク)を誤って消さないための余裕。
RANGE_GUARD = 1.25


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """WAV を読み込み、モノラル float64 で返す。"""
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    return data.mean(axis=1), sr


def extract_f0(
    x: np.ndarray,
    sr: int,
    f0_floor: float = 60.0,
    f0_ceil: float = 500.0,
    frame_shift_ms: float = 5.0,
    median_filter: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """F0 系列を抽出する。

    Returns:
        (times, f0): 秒単位の時刻と Hz 単位の F0。無声フレームは 0。
    """
    f0, t = pyworld.harvest(
        x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_shift_ms
    )
    f0 = pyworld.stonemask(x, f0, t, sr)
    # harvest は探索上下限の外側の値も返すことがある。倍・半ピッチ等の明確な
    # 外れ値のみ無声化し、境界近傍の真の値(句末の下降・強調のピーク)は残す
    lo, hi = f0_floor / RANGE_GUARD, f0_ceil * RANGE_GUARD
    out_of_range = (f0 > 0) & ((f0 < lo) | (f0 > hi))
    n_voiced = int((f0 > 0).sum())
    if n_voiced and out_of_range.sum() / n_voiced > 0.02:
        logger.warning(
            "有声フレームの %.1f%% が探索範囲外で無声化されました。"
            "f0_floor/f0_ceil が狭すぎる可能性があります (floor=%.0f, ceil=%.0f)",
            100 * out_of_range.sum() / n_voiced, f0_floor, f0_ceil,
        )
    f0[out_of_range] = 0.0
    if median_filter:
        f0 = _median5_voiced(f0)
    return t, f0


def _median5_voiced(f0: np.ndarray) -> np.ndarray:
    """有声区間のみに 5 点メディアンフィルタを適用する(倍/半ピッチ誤りの緩和)。"""
    out = f0.copy()
    voiced = f0 > 0
    n = len(f0)
    for i in np.where(voiced)[0]:
        lo, hi = max(0, i - 2), min(n, i + 3)
        win = f0[lo:hi]
        win = win[win > 0]
        out[i] = np.median(win)
    return out


def interpolate_unvoiced_in_spans(
    times: np.ndarray, f0: np.ndarray, spans: list[tuple[float, float]]
) -> np.ndarray:
    """指定区間(アクセント句)内の無声フレームを線形補間した F0 を返す。

    区間の外側は変更しない。区間内に有声フレームが 2 未満なら何もしない。
    """
    out = f0.copy()
    for t0, t1 in spans:
        idx = np.where((times >= t0) & (times <= t1))[0]
        if len(idx) == 0:
            continue
        seg = f0[idx]
        voiced = seg > 0
        if voiced.sum() < 2:
            continue
        seg_t = times[idx]
        out[idx] = np.where(
            voiced, seg, np.interp(seg_t, seg_t[voiced], seg[voiced])
        )
    return out


def range_from_f0(
    f0: np.ndarray,
    q_low: float = 0.75,
    q_high: float = 1.5,
    floor_min: float = 50.0,
    ceil_max: float = 600.0,
    default: tuple[float, float] = (60.0, 500.0),
) -> tuple[float, float]:
    """第 1 パスの F0 分布から抽出レンジを推定する。

    floor = q_low * Q25, ceil = q_high * Q75(有声フレームの四分位)。
    有声フレームが 20 未満なら default を返す。
    """
    voiced = f0[f0 > 0]
    if len(voiced) < 20:
        return default
    q25, q75 = np.percentile(voiced, [25, 75])
    floor = max(floor_min, q_low * float(q25))
    ceil = min(ceil_max, max(q_high * float(q75), floor * 2.0))
    return float(floor), float(ceil)


def estimate_speaker_range(
    x: np.ndarray, sr: int, frame_shift_ms: float = 5.0
) -> tuple[float, float]:
    """広域(60–600 Hz)の第 1 パスからレンジを推定する(単一ファイル用)。"""
    f0, t = pyworld.harvest(
        x, sr, f0_floor=60.0, f0_ceil=600.0, frame_period=frame_shift_ms
    )
    f0 = pyworld.stonemask(x, f0, t, sr)
    return range_from_f0(f0)
