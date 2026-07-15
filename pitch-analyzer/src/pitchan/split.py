"""音声とテキストを文単位に対応付けて分割する(発話単位アラインメントのため)。

ファイル全体を 1 発話として MFA に渡すと、言い淀み等の不一致 1 箇所で
ファイル全体のアラインメントが失敗する。文末記号で区切った文境界を、
音声の無音区間に対応付けて発話単位に分割することで、失敗・精度劣化の
影響をその文だけに閉じ込める。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .textproc import AccentPhrase

logger = logging.getLogger(__name__)

MIN_PAUSE_SEC = 0.30  # 無音区間として検出する最小長
BOUNDARY_MIN_PAUSE_SEC = 0.40  # 文境界の候補とする無音の最小長(読点ポーズを除外)
BOUNDARY_TOL_SEC = 1.5  # 予測境界と無音候補の対応付けを許す最大のずれ(発話時間座標)
MIN_SEC_PER_MORA_PLAN = 0.05  # 分割計画の妥当性検査(発話部分のモーラあたり時間)
MAX_SEC_PER_MORA_PLAN = 0.40
_FRAME_MS = 10.0


@dataclass
class Utterance:
    """1 発話 = 連続する 1 つ以上の文。"""

    index: int
    t_start: float
    t_end: float
    aps: list[AccentPhrase]

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def detect_silences(
    x: np.ndarray, sr: int, min_pause: float = MIN_PAUSE_SEC
) -> list[tuple[float, float]]:
    """RMS エネルギーに基づいて min_pause 以上の無音区間を検出する。"""
    frame = max(1, int(sr * _FRAME_MS / 1000))
    n = len(x) // frame
    if n == 0:
        return []
    e = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    # しきい値: 発話部のレベル(90 パーセンタイル)の -30dB 相当
    thr = max(float(np.percentile(e, 90)) * 0.03, 1e-8)
    silent = e < thr
    silences: list[tuple[float, float]] = []
    start = None
    dt = frame / sr
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * dt >= min_pause:
                silences.append((start * dt, i * dt))
            start = None
    if start is not None and (n - start) * dt >= min_pause:
        silences.append((start * dt, n * dt))
    return silences


def plan_utterances(
    aps: list[AccentPhrase],
    silences: list[tuple[float, float]],
    total_dur: float,
    tol: float = BOUNDARY_TOL_SEC,
) -> list[Utterance]:
    """文境界を無音候補に対応付け、発話区間のリストを返す。

    対応付けは**発話時間座標**(それまでの無音を除いた累積時間)で行う。
    ポーズの長さ・分布が予測位置を歪めないため、実時間での比例配分より頑健。
    予測は境界を 1 つ確定するごとに残り区間で再計算する(再アンカー)ので、
    読速の変動による累積ずれも自己補正される。

    候補は BOUNDARY_MIN_PAUSE_SEC 以上の無音に限定し(読点ポーズを除外)、
    許容幅 tol 内に候補がない文境界は分割せず前後の文を同一発話にまとめる。
    分割計画がモーラあたり時間として不自然な場合は、誤った切り出しで黙って
    ずれるより安全なファイル全体(1 発話)へフォールバックする。
    """
    sents: dict[int, list[AccentPhrase]] = {}
    for ap in aps:
        sents.setdefault(ap.sentence_index, []).append(ap)
    order = sorted(sents)
    moras = np.array([sum(a.mora_count for a in sents[s]) for s in order], float)
    m = len(order)
    whole = [Utterance(0, 0.0, total_dur, list(aps))]
    if m <= 1 or not silences:
        return whole

    sil = sorted(silences)

    def speech_time(t: float) -> float:
        """時刻 t までの発話時間(無音を除いた累積時間)。"""
        return t - sum(max(0.0, min(e, t) - s) for s, e in sil)

    total_speech = speech_time(total_dur)
    total_moras = float(moras.sum())
    if total_speech <= 0 or total_moras <= 0:
        return whole
    cum = np.cumsum(moras)

    # 文境界の候補: 十分長い無音の中央(ファイル端は除外)。発話時間座標も持つ
    cands = [
        ((s0 + s1) / 2, speech_time((s0 + s1) / 2))
        for s0, s1 in sil
        if (s1 - s0) >= BOUNDARY_MIN_PAUSE_SEC
        and 0.2 < (s0 + s1) / 2 < total_dur - 0.2
    ]

    # 逐次・再アンカー方式の対応付け(発話時間座標)
    assigned: dict[int, float] = {}  # 文境界 index -> 絶対時刻
    prev_st = 0.0
    prev_moras = 0.0
    ci = 0
    for i in range(m - 1):
        k_local = (total_speech - prev_st) / max(total_moras - prev_moras, 1.0)
        expected_st = prev_st + (cum[i] - prev_moras) * k_local
        best_j, best_cost = None, tol
        for j in range(ci, len(cands)):
            cost = abs(cands[j][1] - expected_st)
            if cost <= best_cost:
                best_j, best_cost = j, cost
        if best_j is None:
            continue  # 対応する無音なし → 前後の文を同一発話に
        assigned[i] = cands[best_j][0]
        prev_st = cands[best_j][1]
        prev_moras = float(cum[i])
        ci = best_j + 1
    n_skipped = (m - 1) - len(assigned)
    if n_skipped:
        logger.info(
            "%d 個の文境界に対応する無音が見つからず、前後の文を同一発話に"
            "まとめました", n_skipped,
        )

    # 発話の構築
    utts: list[Utterance] = []
    cur: list[AccentPhrase] = []
    t0 = 0.0
    for si in range(m):
        cur.extend(sents[order[si]])
        if si in assigned or si == m - 1:
            t1 = assigned.get(si, total_dur)
            utts.append(Utterance(len(utts), t0, t1, cur))
            t0 = t1
            cur = []

    # 分割計画の妥当性検査: 各発話の発話部分のモーラあたり時間が現実的な範囲か。
    # 不自然な場合は誤った境界に強制整列するより 1 発話(従来方式)へ戻す
    if len(utts) > 1:
        for u in utts:
            u_moras = sum(ap.mora_count for ap in u.aps)
            u_speech = speech_time(u.t_end) - speech_time(u.t_start)
            per_mora = u_speech / max(u_moras, 1)
            if not (MIN_SEC_PER_MORA_PLAN <= per_mora <= MAX_SEC_PER_MORA_PLAN):
                logger.warning(
                    "発話 %d のモーラあたり時間が %.0f ms と不自然なため、"
                    "文単位分割を中止してファイル全体を 1 発話として扱います"
                    "(文境界と無音の対応付けに失敗した可能性)",
                    u.index, per_mora * 1000,
                )
                return whole
    return utts
