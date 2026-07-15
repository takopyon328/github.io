"""F0 の正規化(半音変換・log z スコア・時間正規化輪郭)。"""

from __future__ import annotations

import numpy as np


def speaker_reference(f0_arrays: list[np.ndarray]) -> tuple[float, float, float]:
    """話者の全有声フレームから (基準F0[Hz], log平均, log標準偏差) を計算する。

    基準 F0 は幾何平均。
    """
    voiced = np.concatenate([f0[f0 > 0] for f0 in f0_arrays])
    if len(voiced) == 0:
        raise ValueError("有声フレームがありません")
    logf = np.log(voiced)
    mu = float(logf.mean())
    sigma = float(logf.std())
    return float(np.exp(mu)), mu, sigma


def to_semitone(f0: np.ndarray, ref_hz: float) -> np.ndarray:
    """半音変換。無声フレームは NaN。"""
    st = np.full_like(f0, np.nan, dtype=float)
    voiced = f0 > 0
    st[voiced] = 12.0 * np.log2(f0[voiced] / ref_hz)
    return st


def to_log_z(f0: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """log 領域 z スコア。無声フレームは NaN。"""
    z = np.full_like(f0, np.nan, dtype=float)
    voiced = f0 > 0
    if sigma > 0:
        z[voiced] = (np.log(f0[voiced]) - mu) / sigma
    return z


def time_normalized_contour(
    times: np.ndarray,
    values: np.ndarray,
    t_start: float,
    t_end: float,
    n_points: int = 30,
    min_voiced_frames: int = 4,
) -> np.ndarray:
    """区間 [t_start, t_end] の値(NaN=無声)を等間隔 n_points にリサンプルする。

    有声フレームが少なすぎる場合は全 NaN を返す。
    """
    idx = np.where((times >= t_start) & (times <= t_end))[0]
    out = np.full(n_points, np.nan)
    if len(idx) == 0:
        return out
    seg_t = times[idx]
    seg_v = values[idx]
    ok = ~np.isnan(seg_v)
    if ok.sum() < min_voiced_frames or t_end <= t_start:
        return out
    rel = (seg_t[ok] - t_start) / (t_end - t_start)
    grid = np.linspace(0.0, 1.0, n_points)
    return np.interp(grid, rel, seg_v[ok])
