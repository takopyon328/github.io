"""簡易版X-JToBI(五十嵐 2015)準拠の層の自動下書き生成。

出力する 4 層:
- segments: 音素区間(MFA の phones をそのまま流用)
- tones:    BPM(句末境界音調)の自動判定ラベル(H% / LH% / HL%)。ポイント層
- words:    カナ単語+アクセント核記号(')。核位置はテキスト予測(規範)
- BI:       1=語境界, 2=アクセント句境界, 3=イントネーション句境界。ポイント層

注意: words 層の核記号と BI=2 はテキストからの予測(東京方言の規範)であり、
実際の発話の記述は Praat 上での手修正によって行う(下書きとしての出力)。
BI=3 は実ポーズの有無による近似、BPM は F0 形状の規則判定でありドラフト品質。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from praatio import textgrid as ptg

from .textproc import AccentPhrase, split_moras

logger = logging.getLogger(__name__)

MIN_PAUSE_FOR_BI3 = 0.2  # [s] これ以上の実ポーズを イントネーション句境界(BI=3)とみなす
BPM_THRESHOLD_ST = 1.5  # [半音] 境界音調とみなす最小の F0 変化量
MIN_BPM_FRAMES = 4  # BPM 判定に必要な最小有声フレーム数
MIN_BPM_REGION_SEC = 0.06  # 最終音素がこれより短ければ直前の音素まで判定区間を広げる


def nucleus_marked_words(ap: AccentPhrase) -> list[str]:
    """AP 内の各単語のカナに、予測アクセント核の記号 ' を挿入したラベル列を返す。

    例: ヤマナシダイガク+デ(5型)→ ["ヤマナシダ'イガク", "デ"]
    平板(0型)・アクセント型不明の場合は記号を付けない。
    """
    labels = [w.pron for w in ap.words]
    if not ap.accent_type:  # None(不明)または 0(平板)
        return labels
    remaining = ap.accent_type
    for i, w in enumerate(ap.words):
        moras = split_moras(w.pron)
        if remaining <= len(moras):
            labels[i] = "".join(moras[:remaining]) + "'" + "".join(moras[remaining:])
            return labels
        remaining -= len(moras)
    logger.warning(
        "AP %d (%s): 予測核位置 %d がモーラ数を超えています",
        ap.index, ap.kana, ap.accent_type,
    )
    return labels


def bi_points(aps: list[AccentPhrase]) -> list[tuple[float, str]]:
    """BI 層のポイント列 (時刻, ラベル) を返す。

    語境界=1、アクセント句境界=2、実ポーズ(または発話末)を伴う境界=3。
    """
    points: list[tuple[float, str]] = []
    for k, ap in enumerate(aps):
        if ap.t_start is None:
            continue
        for w in ap.words[:-1]:
            if w.t_end is not None:
                points.append((w.t_end, "1"))
        nxt = next(
            (a for a in aps[k + 1:] if a.t_start is not None), None
        )
        if nxt is None or (nxt.t_start - ap.t_end) >= MIN_PAUSE_FOR_BI3:
            points.append((ap.t_end, "3"))
        else:
            points.append((ap.t_end, "2"))
    return points


def _bpm_region(
    phones: list[tuple[float, float, str]], ap: AccentPhrase, eps: float = 0.02
) -> tuple[float, float] | None:
    """BPM 判定区間(AP 末尾の音素、短ければ 1 つ前まで)を返す。"""
    seg = [
        p for p in phones
        if p[0] >= ap.t_start - eps and p[1] <= ap.t_end + eps
    ]
    if not seg:
        return None
    start, end, _ = seg[-1]
    if end - start < MIN_BPM_REGION_SEC and len(seg) >= 2:
        start = seg[-2][0]
    return start, end


def classify_bpm(
    times: np.ndarray,
    f0_st: np.ndarray,
    phones: list[tuple[float, float, str]],
    ap: AccentPhrase,
) -> str:
    """AP 末尾の F0 形状から BPM を規則判定する(ドラフト品質)。

    戻り値: "H%"(上昇) / "LH%"(下降後上昇) / "HL%"(上昇後下降) / ""(なし)
    """
    if ap.t_start is None:
        return ""
    region = _bpm_region(phones, ap)
    if region is None:
        return ""
    idx = np.where((times >= region[0]) & (times <= region[1]))[0]
    v = f0_st[idx]
    v = v[~np.isnan(v)]
    if len(v) < MIN_BPM_FRAMES:
        return ""
    th = BPM_THRESHOLD_ST
    start, end = v[0], v[-1]
    vmax, vmin = v.max(), v.min()
    imax, imin = int(np.argmax(v)), int(np.argmin(v))
    n = len(v)
    # 上昇後下降(imax が中間)
    if vmax - start >= th and vmax - end >= th and 0 < imax < n - 1:
        return "HL%"
    # 下降後上昇(imin が中間で深い谷)
    if (
        end - vmin >= th
        and start - vmin >= 0.5 * th
        and n // 5 <= imin <= 4 * n // 5
    ):
        return "LH%"
    # 単純上昇
    if end - start >= th:
        return "H%"
    return ""


def classify_bpm_all(
    times: np.ndarray,
    f0_st: np.ndarray,
    phones: list[tuple[float, float, str]],
    aps: list[AccentPhrase],
) -> dict[int, str]:
    return {ap.index: classify_bpm(times, f0_st, phones, ap) for ap in aps}


def write_xjtobi_textgrid(
    path: Path,
    aps: list[AccentPhrase],
    phones: list[tuple[float, float, str]],
    duration: float,
    times: np.ndarray,
    f0_st: np.ndarray,
) -> None:
    """簡易版 X-JToBI 準拠 4 層の TextGrid(下書き)を書き出す。"""
    tg = ptg.Textgrid()
    if phones:
        tg.addTier(ptg.IntervalTier("segments", phones, 0, duration))

    tone_points = []
    for ap in aps:
        label = classify_bpm(times, f0_st, phones, ap)
        if label and ap.t_end is not None:
            tone_points.append((ap.t_end, label))
    tg.addTier(ptg.PointTier("tones", tone_points, 0, duration))

    word_entries = []
    for ap in aps:
        if ap.t_start is None:
            continue
        for w, label in zip(ap.words, nucleus_marked_words(ap)):
            word_entries.append((w.t_start, w.t_end, label))
    tg.addTier(ptg.IntervalTier("words", word_entries, 0, duration))

    tg.addTier(ptg.PointTier("BI", bi_points(aps), 0, duration))
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)
