"""簡易版X-JToBI(五十嵐 2015)準拠の層の自動下書き生成。

出力する 4 層:
- segments: 音素区間(MFA の phones をそのまま流用)
- tones:    トーンの下書き。イントネーション句頭の %L、句頭上昇 H-、
            アクセント核 H*+L(いずれも予測に基づく近似位置)と、
            句末の L%(+BPM 自動判定 H% / LH% / HL% を連結、例 "L%LH%")。ポイント層
- words:    カナ単語+アクセント核記号(')。核位置はテキスト予測(規範)
- BI:       1=語境界, 2=アクセント句境界, 3=イントネーション句境界。ポイント層

注意: words 層の核記号・BI=2・tones 層の %L/H-/H*+L はテキストからの予測
(東京方言の規範)であり、位置は近似(簡易版の方針どおり F0 曲線への正確な
同期は行わない)。実際の発話の記述は Praat 上での手修正によって行う。
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
LH_ONSET_RATIO = 0.5  # 上昇開始が最終モーラのこの割合より遅ければ LH%(早ければ H%)
LABEL_MATCH_TOL = 0.15  # [s] 手修正でずれた BI / BPM ポイントを語末・句末に対応付ける許容幅


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


BPM_LABELS = ("H%", "LH%", "HL%", "HLH%")  # X-JToBI の BPM 4 種(五十嵐資料 p.10)

# japanese_mfa(IPA)の母音・撥音など「モーラの核」になりうる音素の判定用
_VOWEL_INITIALS = set("aiueoɯ")
_MORAIC_LABELS = {"ɴ", "n̩", "ɰ̃", "N", "m̩"}


def _is_vowel_like(label: str) -> bool:
    lab = label.strip()
    if not lab:
        return False
    return lab[0] in _VOWEL_INITIALS or "ː" in lab or lab in _MORAIC_LABELS


def _bpm_region(
    phones: list[tuple[float, float, str]], ap
) -> tuple[float, float] | None:
    """BPM 判定区間(最終モーラ相当)を返す。

    資料の手順どおり分節音の情報から最終モーラを特定する: AP 末尾から
    さかのぼって最初の母音的音素(母音・長母音・撥音)の開始を区間の起点とする。
    長母音(ː)で終わる句では音素の後半のみを最終モーラ相当とする。
    音素表記が判別できない場合は「最終音素、短ければ 1 つ前まで」に後退する。

    words 層だけが手修正されて segments と境界がずれていても拾えるよう、
    音素は重なりで選別し(音素長の半分以上または 30ms 以上の重なり)、
    区間は AP の範囲にクリップする。
    """
    seg = []
    for p0, p1, lab in phones:
        ov = min(p1, ap.t_end) - max(p0, ap.t_start)
        if ov > 0 and (ov >= 0.5 * (p1 - p0) or ov >= 0.03):
            seg.append((max(p0, ap.t_start), min(p1, ap.t_end), lab))
    if not seg:
        return None
    end = seg[-1][1]
    for s0, s1, lab in reversed(seg):
        if _is_vowel_like(lab):
            if "ː" in lab:  # 長母音は後半のみが最終モーラ
                s0 = s0 + (s1 - s0) / 2
            return s0, end
    start, s1, _ = seg[-1]
    if s1 - start < MIN_BPM_REGION_SEC and len(seg) >= 2:
        start = seg[-2][0]
    return start, end


def _pivots(v: np.ndarray, delta: float) -> list[float]:
    """F0 系列を有意な転回点(振幅 delta 以上)で縮約した値列を返す。

    戻り値は [始点, 転回点..., 終点]。各隣接ペアの差は delta 以上になる。
    """
    pivots = [float(v[0])]
    ext = float(v[0])
    direction = 0  # 0=未確定, 1=上昇中, -1=下降中
    for x in map(float, v[1:]):
        if direction == 0:
            if x - pivots[0] >= delta:
                direction, ext = 1, x
            elif pivots[0] - x >= delta:
                direction, ext = -1, x
        elif direction == 1:
            if x > ext:
                ext = x
            elif ext - x >= delta:
                pivots.append(ext)
                direction, ext = -1, x
        else:
            if x < ext:
                ext = x
            elif x - ext >= delta:
                pivots.append(ext)
                direction, ext = 1, x
    if direction != 0:
        pivots.append(ext)
    return pivots


def classify_bpm(
    times: np.ndarray,
    f0_st: np.ndarray,
    phones: list[tuple[float, float, str]],
    ap,
) -> str:
    """AP 末尾(最終モーラ区間)の F0 形状から BPM を規則判定する(ドラフト品質)。

    転回点分析により X-JToBI の BPM 4 種すべてを判定する:
    上昇="H%" / 下降後上昇="LH%" / 上昇後下降="HL%" / 上昇下降上昇="HLH%" /
    有意な動きなし="")
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
    if len(v) >= 5:  # 3 点移動平均で倍ピッチ等の微細なジッタを抑える
        v = np.convolve(v, np.ones(3) / 3, mode="valid")
    th = BPM_THRESHOLD_ST
    pattern = "".join("+" if d > 0 else "-" for d in np.diff(_pivots(v, th)))
    if pattern == "+-":
        return "HL%"
    if pattern == "+-+":
        return "HLH%"
    if pattern.endswith("+"):
        # 上昇で終わる形状。X-JToBI の弁別基準に従い、H%(上昇調1)と
        # LH%(上昇調2)は上昇開始の時間的な遅れで分ける:
        # 最小値付近に最後に居た時点(=上昇開始)が最終モーラの前半なら H%、
        # 後半まで低く保たれてから上昇するなら LH%。
        # 先行するアクセント下降が食い込んだだけの早い上昇は H% に落ちる。
        near_min = np.where(v <= v.min() + 0.25 * th)[0]
        onset_ratio = float(near_min[-1]) / max(len(v) - 1, 1)
        return "LH%" if onset_ratio >= LH_ONSET_RATIO else "H%"
    return ""


def _nucleus_time(ap: AccentPhrase) -> float | None:
    """予測アクセント核(核モーラの終端)の近似時刻を返す。

    核を含む単語の中で、モーラ数に比例した線形配分で近似する
    (簡易版の方針どおり正確な同期は求めない)。
    """
    if not ap.accent_type or ap.t_start is None:
        return None
    remaining = ap.accent_type
    for w in ap.words:
        moras = split_moras(w.pron)
        if remaining <= len(moras):
            if w.t_start is None:
                return None
            return w.t_start + (remaining / len(moras)) * (w.t_end - w.t_start)
        remaining -= len(moras)
    return None


def tone_points(
    aps: list[AccentPhrase], bpm: dict[int, str]
) -> list[tuple[float, str]]:
    """tones 層のポイント列を生成する。

    - %L: イントネーション句頭(発話頭・実ポーズ後)の AP 開始位置
    - H-: 句頭上昇の目標(第 1 モーラ終端の近似位置)。頭高(1型)の句には付けない
    - H*+L: 予測アクセント核の近似位置(有核句のみ)
    - L%(+BPM): AP 終端。BPM 判定があれば "L%H%" のように連結
    """
    points: list[tuple[float, str]] = []
    prev: AccentPhrase | None = None
    for ap in aps:
        if ap.t_start is None:
            continue
        dur = ap.t_end - ap.t_start
        if prev is None or (ap.t_start - prev.t_end) >= MIN_PAUSE_FOR_BI3:
            points.append((ap.t_start, "%L"))
        if ap.accent_type != 1 and ap.mora_count >= 2:
            points.append((ap.t_start + dur / ap.mora_count, "H-"))
        if ap.accent_type and ap.accent_type >= 1:
            t_nuc = _nucleus_time(ap)
            if t_nuc is not None:
                points.append((t_nuc, "H*+L"))
        points.append((ap.t_end, "L%" + bpm.get(ap.index, "")))
        prev = ap
    return _strictly_increasing(points)


def _strictly_increasing(
    points: list[tuple[float, str]], min_step: float = 0.001
) -> list[tuple[float, str]]:
    """ポイント層の時刻を単調増加に補正する(同時刻・逆順は 1ms ずつずらす)。"""
    points = sorted(points, key=lambda p: p[0])
    out: list[tuple[float, str]] = []
    for t, label in points:
        if out and t <= out[-1][0]:
            t = out[-1][0] + min_step
        out.append((t, label))
    return out


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

    bpm = classify_bpm_all(times, f0_st, phones, aps)
    tg.addTier(ptg.PointTier("tones", tone_points(aps, bpm), 0, duration))

    word_entries = []
    for ap in aps:
        if ap.t_start is None:
            continue
        for w, label in zip(ap.words, nucleus_marked_words(ap)):
            word_entries.append((w.t_start, w.t_end, label))
    tg.addTier(ptg.IntervalTier("words", word_entries, 0, duration))

    tg.addTier(ptg.PointTier("BI", bi_points(aps), 0, duration))
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)


def peak_excl_bpm(
    times: np.ndarray,
    f0_st: np.ndarray,
    phones: list[tuple[float, float, str]],
    ap: AccentPhrase,
    bpm_label: str,
) -> tuple[float, float]:
    """BPM 区間を除いた AP 内の最大 F0(半音)とその時刻を返す(五十嵐方式)。

    BPM がない句では AP 全体の最大値と同じになる。
    BPM があるのに分節音情報から BPM 区間を特定できない場合は、
    誤った値(BPM の上昇を含む最大値)を返す代わりに (nan, nan) を返す。
    """
    if ap.t_start is None:
        return (float("nan"), float("nan"))
    t_end = ap.t_end
    if bpm_label:
        region = _bpm_region(phones, ap)
        if region is None:
            logger.warning(
                "AP %s: BPM=%s だが分節音情報がなく BPM 区間を特定できないため "
                "peak_excl_bpm は NaN になります",
                getattr(ap, "index", "?"), bpm_label,
            )
            return (float("nan"), float("nan"))
        t_end = region[0]
    idx = np.where((times >= ap.t_start) & (times <= t_end))[0]
    if len(idx) == 0:
        return (float("nan"), float("nan"))
    v = f0_st[idx]
    ok = ~np.isnan(v)
    if not ok.any():
        return (float("nan"), float("nan"))
    sub = np.where(ok)[0]
    imax = sub[int(np.argmax(v[sub]))]
    return float(v[imax]), float(times[idx][imax])


def peak_excl_bpm_all(
    times: np.ndarray,
    f0_st: np.ndarray,
    phones: list[tuple[float, float, str]],
    aps: list[AccentPhrase],
    bpm: dict[int, str],
) -> dict[int, tuple[float, float]]:
    return {
        ap.index: peak_excl_bpm(times, f0_st, phones, ap, bpm.get(ap.index, ""))
        for ap in aps
    }


# ---------------------------------------------------------------------------
# ラベル駆動の計測(五十嵐資料 p.14-23 の手順)
# 手修正済みの簡易版 X-JToBI TextGrid を読み戻し、修正後のラベル
# (核記号・BI・BPM)に基づいて F0 を計測する。
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class LabeledAP:
    """TextGrid のラベルから再構成したアクセント句。"""

    index: int
    words: list[tuple[float, float, str]] = field(default_factory=list)
    bi: str = ""  # この句の右端の BI(2 / 3)
    bpm: str = ""  # tones 層由来の BPM(修正後の値)

    @property
    def t_start(self) -> float:
        return self.words[0][0]

    @property
    def t_end(self) -> float:
        return self.words[-1][1]

    @property
    def kana(self) -> str:
        return "".join(lab.replace("'", "") for _, _, lab in self.words)

    @property
    def nucleus_mora(self) -> int:
        """核記号 ' の位置(AP 内の通しモーラ番号)。無核なら 0。"""
        cum = 0
        for _, _, lab in self.words:
            if "'" in lab:
                return cum + len(split_moras(lab.split("'")[0]))
            cum += len(split_moras(lab.replace("'", "")))
        return 0


def _bpm_from_tone_label(label: str) -> str:
    """tones 層のラベルから BPM 部分を取り出す(例 "L%LH%" → "LH%")。"""
    lab = label.strip()
    if lab.startswith("L%"):
        lab = lab[2:]
    return lab if lab in BPM_LABELS else ""


def parse_xjtobi_textgrid(
    path: Path,
) -> tuple[list[LabeledAP], list[tuple[float, float, str]]]:
    """簡易版 X-JToBI TextGrid(手修正済み可)を解析する。

    Returns:
        (アクセント句リスト, segments 層の音素区間リスト)
    """
    tg = ptg.openTextgrid(str(path), includeEmptyIntervals=False)
    names = tg.tierNames

    def find_tier(target: str) -> str | None:
        for n in names:
            if n == target or n.lower() == target.lower():
                return n
        return None

    words_name = find_tier("words")
    bi_name = find_tier("BI")
    if words_name is None or bi_name is None:
        raise ValueError(
            f"{path}: words 層 / BI 層が見つかりません(層: {list(names)})"
        )
    words = [
        (e.start, e.end, e.label.strip())
        for e in tg.getTier(words_name).entries
        if e.label.strip()
    ]
    bi_points_ = [
        (p.time, p.label.strip()) for p in tg.getTier(bi_name).entries
    ]
    tones_name = find_tier("tones")
    tone_points_ = (
        [(p.time, p.label.strip()) for p in tg.getTier(tones_name).entries]
        if tones_name
        else []
    )
    seg_name = find_tier("segments")
    segments = (
        [
            (e.start, e.end, e.label)
            for e in tg.getTier(seg_name).entries
            if e.label.strip()
        ]
        if seg_name
        else []
    )

    # BI の 2/3 ポイントを最寄りの語末に割り当てる(Praat での境界手修正で
    # ポイントが語末から多少ずれていても句の分割が壊れないようにする)
    word_ends = [w[1] for w in words]
    break_after: dict[int, str] = {}
    for t, lab in bi_points_:
        if lab not in ("2", "3"):
            continue
        j = min(range(len(word_ends)), key=lambda i: abs(word_ends[i] - t))
        dt = abs(word_ends[j] - t)
        if dt > LABEL_MATCH_TOL:
            logger.warning(
                "%s: BI=%s (t=%.3f) がどの語末からも %.0f ms 以上離れているため"
                "無視します", path, lab, t, LABEL_MATCH_TOL * 1000,
            )
            continue
        if break_after.get(j) != "3":  # 同一語末に複数割当なら 3 を優先
            break_after[j] = lab
    if not break_after:
        logger.warning(
            "%s: BI 層に 2/3 のポイントがないため、全体を 1 アクセント句として"
            "扱います", path,
        )

    aps: list[LabeledAP] = []
    cur: list[tuple[float, float, str]] = []
    for i, w in enumerate(words):
        cur.append(w)
        if i in break_after or i == len(words) - 1:
            aps.append(
                LabeledAP(index=len(aps), words=cur, bi=break_after.get(i, "3"))
            )
            cur = []

    # tones 層の BPM を最寄りの句末に割り当てる
    for t, lab in tone_points_:
        bpm = _bpm_from_tone_label(lab)
        if not bpm:
            continue
        ap = min(aps, key=lambda a: abs(a.t_end - t))
        if abs(ap.t_end - t) <= LABEL_MATCH_TOL:
            ap.bpm = bpm
        else:
            logger.warning(
                "%s: BPM=%s (t=%.3f) がどの句末からも %.0f ms 以上離れているため"
                "無視します", path, bpm, t, LABEL_MATCH_TOL * 1000,
            )

    if not segments:
        logger.warning(
            "%s: segments 層がないため、BPM 付き句の peak_excl_bpm は NaN に"
            "なります", path,
        )
    return aps, segments


def measure_labeled_aps(
    times: np.ndarray,
    f0_st: np.ndarray,
    aps: list[LabeledAP],
    segments: list[tuple[float, float, str]],
    file_name: str,
) -> list[dict]:
    """修正後ラベルに基づく AP ごとの F0 計測(資料 p.14-23 の手順)。"""
    rows = []
    for ap in aps:
        sel = np.where((times >= ap.t_start) & (times <= ap.t_end))[0]
        st = f0_st[sel]
        ok = ~np.isnan(st)
        row: dict = {
            "file": file_name,
            "ap_index": ap.index,
            "ap_kana": ap.kana,
            "nucleus_mora": ap.nucleus_mora,
            "accented": int(ap.nucleus_mora > 0),
            "t_start": round(ap.t_start, 4),
            "t_end": round(ap.t_end, 4),
            "duration_sec": round(ap.t_end - ap.t_start, 4),
            "bi": ap.bi,
            "bpm": ap.bpm,
            "voiced_ratio": round(float(ok.mean()), 3) if len(st) else np.nan,
        }
        if ok.any():
            sub = np.where(ok)[0]
            imax = sub[int(np.argmax(st[sub]))]
            row.update(
                f0_mean_st=round(float(st[sub].mean()), 3),
                f0_max_st=round(float(st[imax]), 3),
                f0_max_time=round(float(times[sel][imax]), 4),
            )
        peak_st, peak_t = peak_excl_bpm(times, f0_st, segments, ap, ap.bpm)
        row["peak_excl_bpm_st"] = round(peak_st, 3)
        row["peak_excl_bpm_time"] = round(peak_t, 4)
        rows.append(row)
    return rows
