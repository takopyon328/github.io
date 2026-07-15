"""分析結果の出力(CSV / JSON / TextGrid / PNG)。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from praatio import textgrid as ptg

from .textproc import AccentPhrase, bunsetsu_groups

logger = logging.getLogger(__name__)


def build_frames_df(
    name: str,
    times: np.ndarray,
    f0_hz: np.ndarray,
    f0_st: np.ndarray,
    f0_z: np.ndarray,
    aps: list[AccentPhrase],
) -> pd.DataFrame:
    ap_index = np.full(len(times), -1, dtype=int)
    word_index = np.full(len(times), -1, dtype=int)
    ap_text = np.full(len(times), "", dtype=object)
    for ap in aps:
        if ap.t_start is None:
            continue
        sel = (times >= ap.t_start) & (times <= ap.t_end)
        ap_index[sel] = ap.index
        ap_text[sel] = ap.kana
        for wi, w in enumerate(ap.words):
            wsel = (times >= w.t_start) & (times <= w.t_end)
            word_index[wsel] = wi
    return pd.DataFrame(
        {
            "file": name,
            "time_sec": np.round(times, 4),
            "f0_hz": np.round(f0_hz, 2),
            "f0_st": np.round(f0_st, 3),
            "f0_z": np.round(f0_z, 3),
            "voiced": (f0_hz > 0).astype(int),
            "ap_index": ap_index,
            "word_index": word_index,
            "ap_text": ap_text,
        }
    )


def build_ap_summary_df(
    name: str,
    times: np.ndarray,
    f0_st: np.ndarray,
    aps: list[AccentPhrase],
) -> pd.DataFrame:
    rows = []
    for ap in aps:
        row: dict = {
            "file": name,
            "ap_index": ap.index,
            "ap_surface": ap.surface,
            "ap_kana": ap.kana,
            "accent_type": ap.accent_type,
            "mora_count": ap.mora_count,
            "follows_pause": int(ap.follows_pause),
            "t_start": ap.t_start,
            "t_end": ap.t_end,
            "duration_sec": ap.duration,
            "low_confidence": int(ap.low_confidence),
        }
        if ap.t_start is not None:
            sel = (times >= ap.t_start) & (times <= ap.t_end)
            st = f0_st[sel]
            voiced = ~np.isnan(st)
            row["voiced_ratio"] = round(float(voiced.mean()), 3) if len(st) else np.nan
            if voiced.any():
                st_v = st[voiced]
                t_v = times[sel][voiced]
                peak = float(t_v[np.argmax(st_v)])
                row.update(
                    f0_mean_st=round(float(st_v.mean()), 3),
                    f0_max_st=round(float(st_v.max()), 3),
                    f0_min_st=round(float(st_v.min()), 3),
                    f0_range_st=round(float(st_v.max() - st_v.min()), 3),
                    peak_time_ratio=round((peak - ap.t_start) / ap.duration, 3)
                    if ap.duration
                    else np.nan,
                )
        rows.append(row)
    return pd.DataFrame(rows)


def build_contours_df(
    name: str,
    times: np.ndarray,
    f0_st: np.ndarray,
    aps: list[AccentPhrase],
    n_points: int,
) -> pd.DataFrame:
    from .normalize import time_normalized_contour

    rows = []
    for ap in aps:
        contour = (
            time_normalized_contour(times, f0_st, ap.t_start, ap.t_end, n_points)
            if ap.t_start is not None
            else np.full(n_points, np.nan)
        )
        row = {"file": name, "ap_index": ap.index, "ap_kana": ap.kana}
        row.update(
            {f"p{i+1:02d}": round(float(v), 3) if not np.isnan(v) else np.nan
             for i, v in enumerate(contour)}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(
    path: Path,
    name: str,
    aps: list[AccentPhrase],
    params: dict,
    contours: pd.DataFrame,
) -> None:
    contour_cols = [c for c in contours.columns if c.startswith("p")]
    data = {
        "file": name,
        "params": params,
        "accent_phrases": [
            {
                "index": ap.index,
                "surface": ap.surface,
                "kana": ap.kana,
                "accent_type": ap.accent_type,
                "mora_count": ap.mora_count,
                "follows_pause": ap.follows_pause,
                "t_start": ap.t_start,
                "t_end": ap.t_end,
                "low_confidence": ap.low_confidence,
                "words": [
                    {
                        "surface": w.surface,
                        "pron": w.pron,
                        "pos": w.pos,
                        "t_start": w.t_start,
                        "t_end": w.t_end,
                    }
                    for w in ap.words
                ],
                "contour_st": [
                    None if pd.isna(v) else v
                    for v in contours.iloc[ap.index][contour_cols]
                ],
            }
            for ap in aps
        ],
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def write_textgrid(
    path: Path,
    aps: list[AccentPhrase],
    phones: list[tuple[float, float, str]],
    duration: float,
) -> None:
    tg = ptg.Textgrid()
    word_entries = [
        (w.t_start, w.t_end, w.pron)
        for ap in aps
        for w in ap.words
        if w.t_start is not None
    ]
    ap_entries = [
        (
            ap.t_start,
            ap.t_end,
            f"{ap.kana}({ap.accent_type if ap.accent_type is not None else '?'})",
        )
        for ap in aps
        if ap.t_start is not None
    ]
    bunsetsu_entries = [
        (g[0].t_start, g[-1].t_end, "".join(w.pron for w in g))
        for ap in aps
        if ap.t_start is not None
        for g in bunsetsu_groups(ap)
        if g[0].t_start is not None
    ]
    tg.addTier(ptg.IntervalTier("accent_phrases", ap_entries, 0, duration))
    tg.addTier(ptg.IntervalTier("bunsetsu", bunsetsu_entries, 0, duration))
    tg.addTier(ptg.IntervalTier("words", word_entries, 0, duration))
    if phones:
        tg.addTier(ptg.IntervalTier("phones", phones, 0, duration))
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)


def _setup_matplotlib():
    """matplotlib を読み込み、日本語フォントを設定する。なければ None。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        available = {fm.name for fm in font_manager.fontManager.ttflist}
        for name in (
            "Noto Sans CJK JP", "IPAexGothic", "Yu Gothic",
            "Meiryo", "MS Gothic", "Hiragino Sans",
        ):
            if name in available:
                matplotlib.rcParams["font.family"] = name
                break
        return plt
    except ImportError:
        logger.warning("matplotlib がないため PNG 出力をスキップします")
        return None


def plot_f0(
    path: Path,
    times: np.ndarray,
    f0_st: np.ndarray,
    aps: list[AccentPhrase],
) -> None:
    plt = _setup_matplotlib()
    if plt is None:
        return

    total = times[-1] if len(times) else 1.0
    n_panels = max(1, int(np.ceil(total / 30.0)))
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(16, 2.2 * n_panels), squeeze=False
    )
    for p in range(n_panels):
        ax = axes[p][0]
        t0, t1 = p * 30.0, (p + 1) * 30.0
        sel = (times >= t0) & (times < t1)
        ax.plot(times[sel], f0_st[sel], ".", markersize=2)
        for ap in aps:
            if ap.t_start is None or ap.t_end < t0 or ap.t_start >= t1:
                continue
            ax.axvspan(ap.t_start, ap.t_end, alpha=0.08, color="C1")
            ax.text(
                (ap.t_start + ap.t_end) / 2, ax.get_ylim()[1],
                ap.kana, rotation=45, fontsize=6, ha="left", va="bottom",
            )
        ax.set_xlim(t0, t1)
        ax.set_ylabel("F0 [st]")
    axes[-1][0].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


MIN_AP_PLOT_SPAN_ST = 8.0  # 句ごと自動スケール時の縦軸の最小スパン(半音)


def plot_ap_pngs(
    dir_path: Path,
    times: np.ndarray,
    f0_st: np.ndarray,
    aps: list[AccentPhrase],
    shared_ylim: bool = False,
) -> None:
    """アクセント句ごとに 1 枚の PNG を出力する。

    既定では縦軸を句ごとに自動スケールし(最小スパン 8 半音)、句内の
    上昇・下降が見やすいようにする。shared_ylim=True でファイル内共通の
    スケールに切り替えられる(句どうしの高さの比較用)。
    """
    plt = _setup_matplotlib()
    if plt is None:
        return
    dir_path.mkdir(parents=True, exist_ok=True)

    voiced = f0_st[~np.isnan(f0_st)]
    if len(voiced):
        glo, ghi = np.percentile(voiced, [1, 99])
        pad = max(1.0, 0.1 * (ghi - glo))
        glo, ghi = glo - pad, ghi + pad
    else:
        glo, ghi = -12, 12

    for ap in aps:
        if ap.t_start is None:
            continue
        sel = (times >= ap.t_start) & (times <= ap.t_end)
        if shared_ylim:
            ylo, yhi = glo, ghi
        else:
            v = f0_st[sel]
            v = v[~np.isnan(v)]
            if len(v):
                mid = (float(v.min()) + float(v.max())) / 2
                span = max(float(v.max()) - float(v.min()) + 2.0,
                           MIN_AP_PLOT_SPAN_ST)
                ylo, yhi = mid - span / 2, mid + span / 2
            else:
                ylo, yhi = glo, ghi
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.plot(times[sel] - ap.t_start, f0_st[sel], ".", markersize=4, color="C0")
        for w in ap.words:
            if w.t_start is None:
                continue
            if w.t_start > ap.t_start:
                ax.axvline(w.t_start - ap.t_start, color="gray",
                           linestyle="--", linewidth=0.7)
            ax.text(
                (w.t_start + w.t_end) / 2 - ap.t_start, ylo + 0.02 * (yhi - ylo),
                w.pron, ha="center", va="bottom", fontsize=8, color="gray",
            )
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0, ap.t_end - ap.t_start)
        acc = f"{ap.accent_type}型" if ap.accent_type is not None else "?型"
        flag = " [low_confidence]" if ap.low_confidence else ""
        ax.set_title(
            f"[{ap.index:03d}] {ap.surface} ({ap.kana}) "
            f"{acc}・{ap.mora_count}モーラ・{ap.t_start:.2f}s〜{flag}",
            fontsize=9,
        )
        ax.set_xlabel("句内時間 [s]")
        ax.set_ylabel("F0 [半音]")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        kana = ap.kana[:12]
        fig.savefig(dir_path / f"ap{ap.index:04d}_{kana}.png", dpi=110)
        plt.close(fig)
