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

MIN_PAUSE_SEC = 0.30  # これ以上の無音を文境界の候補とする
BOUNDARY_TOL_SEC = 3.0  # 予測境界と無音候補の対応付けを許す最大のずれ
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

    予測境界時刻(累積モーラ数に比例)と無音候補の中央時刻を、順序を保った
    動的計画法で対応付ける。対応する無音がない文境界は分割せず前後の文を
    同一発話にまとめる(ポーズなしで読み継がれた文)。
    """
    # 文ごとの AP と モーラ数
    sents: dict[int, list[AccentPhrase]] = {}
    for ap in aps:
        sents.setdefault(ap.sentence_index, []).append(ap)
    order = sorted(sents)
    moras = np.array([sum(a.mora_count for a in sents[s]) for s in order], float)
    m = len(order)
    if m <= 1 or not silences:
        return [Utterance(0, 0.0, total_dur, list(aps))]

    # 文境界(m-1 個)の予測時刻
    k = total_dur / max(moras.sum(), 1.0)
    expected = np.cumsum(moras)[:-1] * k
    # 無音候補の中央時刻(ファイル端に近すぎるものは除外)
    cands = [
        (s0 + s1) / 2 for s0, s1 in silences if 0.2 < (s0 + s1) / 2 < total_dur - 0.2
    ]

    # DP による順序保存の対応付け。dp[i][j] = 境界 i 個・候補 j 個まで見た最小コスト。
    # 境界を割り当てない(分割しない)ペナルティ = tol。
    B, n = m - 1, len(cands)
    INF = float("inf")
    SKIP_B, SKIP_C, MATCH = 0, 1, 2
    dp = np.full((B + 1, n + 1), INF)
    move = np.full((B + 1, n + 1), -1, int)
    dp[0, :] = 0.0
    move[0, 1:] = SKIP_C
    for i in range(1, B + 1):
        for j in range(n + 1):
            best, mv = dp[i - 1, j] + tol, SKIP_B  # 境界 i をスキップ
            if j >= 1 and dp[i, j - 1] < best:  # 候補 j を使わない
                best, mv = dp[i, j - 1], SKIP_C
            if j >= 1:
                cost = abs(cands[j - 1] - expected[i - 1])
                if cost <= tol and dp[i - 1, j - 1] + cost < best:
                    best, mv = dp[i - 1, j - 1] + cost, MATCH
            dp[i, j], move[i, j] = best, mv

    assigned: dict[int, float] = {}  # 境界 index (0..B-1) -> 時刻
    i, j = B, n
    while i > 0 or j > 0:
        mv = move[i, j]
        if mv == MATCH:
            assigned[i - 1] = cands[j - 1]
            i, j = i - 1, j - 1
        elif mv == SKIP_B:
            i -= 1
        else:
            j -= 1
    n_skipped = B - len(assigned)
    if n_skipped:
        logger.info(
            "%d 個の文境界に対応する無音が見つからず、前後の文を同一発話に"
            "まとめました", n_skipped,
        )

    # 発話の構築
    utts: list[Utterance] = []
    cur: list[AccentPhrase] = []
    t0 = 0.0
    for si, s in enumerate(order):
        cur.extend(sents[s])
        if si in assigned or si == m - 1:
            t1 = assigned.get(si, total_dur)
            utts.append(Utterance(len(utts), t0, t1, cur))
            t0 = t1
            cur = []
    return utts
