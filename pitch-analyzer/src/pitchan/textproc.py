"""テキストからアクセント句構造を推定する(pyopenjtalk ベース)。

NJD(run_frontend)の chain_flag で単語をアクセント句にまとめ、
アクセント型・モーラ数はフルコンテキストラベルの F: フィールドから取る。
単語の acc 値は連結後のアクセント句のアクセント型と一致しないため使わない。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import pyopenjtalk

logger = logging.getLogger(__name__)

_SMALL_KANA = set("ァィゥェォャュョヮ")
# NJD の発音に含まれる無声化マーク・記号類(カタカナと長音以外は除去)
_KATAKANA_RE = re.compile(r"[ァ-ヴー]+")


@dataclass
class Word:
    surface: str
    pron: str  # カタカナ読み(無声化マーク等を除去済み、ヲ→オ正規化済み)
    mora_count: int
    pos: str
    pos_group1: str = ""  # 品詞細分類(自立 / 非自立 / サ変接続 / 接尾 など)
    t_start: float | None = None
    t_end: float | None = None


@dataclass
class AccentPhrase:
    index: int
    words: list[Word] = field(default_factory=list)
    accent_type: int | None = None
    mora_count: int = 0
    follows_pause: bool = False  # 直前がポーズまたは文頭
    sentence_index: int = 0  # 所属する文の通し番号(文末記号・改行で区切る)
    t_start: float | None = None
    t_end: float | None = None
    low_confidence: bool = False

    @property
    def surface(self) -> str:
        return "".join(w.surface for w in self.words)

    @property
    def kana(self) -> str:
        return "".join(w.pron for w in self.words)

    @property
    def duration(self) -> float | None:
        if self.t_start is None or self.t_end is None:
            return None
        return self.t_end - self.t_start


def split_moras(kana: str) -> list[str]:
    """カタカナ文字列をモーラに分割する。拗音・小書き文字は前のモーラに付ける。"""
    moras: list[str] = []
    for ch in kana:
        if ch in _SMALL_KANA and moras:
            moras[-1] += ch
        else:
            moras.append(ch)
    return moras


def clean_pron(pron: str) -> str:
    """NJD の発音から無声化マーク等を除去し、カタカナと長音のみ残す。"""
    pron = pron.replace("ヲ", "オ")
    return "".join(_KATAKANA_RE.findall(pron))


def _ap_info_from_labels(text: str) -> list[tuple[int, int, bool]]:
    """フルコンテキストラベルから AP ごとの (モーラ数, アクセント型, ポーズ直後) を得る。

    F: フィールドは AP 内で一定かつ位置情報を含むため AP の同一性判定キーに使える。
    """
    infos: list[tuple[int, int, bool]] = []
    prev_key: str | None = None
    after_pause = True
    for lb in pyopenjtalk.extract_fullcontext(text):
        ph = re.search(r"\-(.+?)\+", lb).group(1)
        if ph in ("sil", "pau"):
            after_pause = True
            prev_key = None
            continue
        m = re.search(r"/F:(\d+)_(\d+)#[^/]*", lb)
        if m is None:
            continue
        key = m.group(0)
        if key != prev_key:
            infos.append((int(m.group(1)), int(m.group(2)), after_pause))
            prev_key = key
            after_pause = False
    return infos


def analyze_text(text: str) -> list[AccentPhrase]:
    """テキストをアクセント句のリストに変換する。"""
    text = unicodedata.normalize("NFKC", text).strip()
    if not text:
        return []

    feats = pyopenjtalk.run_frontend(text)

    aps: list[AccentPhrase] = []
    after_pause = True
    for f in feats:
        if f["mora_size"] == 0 or f["pos"] == "記号":
            after_pause = True
            continue
        pron = clean_pron(f["pron"])
        if not pron:
            logger.warning("読みが空の語をスキップ: %r", f["string"])
            after_pause = True
            continue
        word = Word(
            surface=f["string"],
            pron=pron,
            mora_count=f["mora_size"],
            pos=f["pos"],
            pos_group1=f.get("pos_group1", "") or "",
        )
        if f["chain_flag"] == 1 and aps and not after_pause:
            aps[-1].words.append(word)
        else:
            aps.append(
                AccentPhrase(index=len(aps), words=[word], follows_pause=after_pause)
            )
            after_pause = False

    for ap in aps:
        ap.mora_count = sum(w.mora_count for w in ap.words)

    # フルコンテキストラベルからアクセント型を補完(AP 数が一致する場合のみ)
    try:
        infos = _ap_info_from_labels(text)
    except Exception:
        logger.exception("フルコンテキストラベルの解析に失敗。アクセント型は未設定になります")
        infos = []
    if len(infos) == len(aps):
        for ap, (mora_count, accent_type, _) in zip(aps, infos):
            ap.accent_type = accent_type
            if mora_count != ap.mora_count:
                logger.warning(
                    "AP %d (%s): モーラ数不一致 NJD=%d labels=%d",
                    ap.index, ap.kana, ap.mora_count, mora_count,
                )
    else:
        logger.warning(
            "AP 数が不一致 (NJD=%d, labels=%d)。アクセント型は未設定になります",
            len(aps), len(infos),
        )
    return aps


def _read_text_file(path: str) -> str:
    """UTF-8(BOM可)を優先し、だめなら Shift_JIS(CP932)として読む。"""
    raw = open(path, "rb").read()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        text = raw.decode("cp932")
        logger.info("%s: Shift_JIS として読み込みました", path)
        return text
    except UnicodeDecodeError as e:
        raise ValueError(
            f"{path}: 文字コードを判別できません (UTF-8 でも Shift_JIS でもありません)。"
            "UTF-8 で保存し直してください。"
        ) from e


_SENTENCE_END_RE = re.compile(r"(?<=[。!?!?])")


_ATTACH_POS = ("助詞", "助動詞")
_PREFIX_POS = ("接頭詞", "接頭辞")


def bunsetsu_groups(ap: AccentPhrase) -> list[list[Word]]:
    """AP 内の単語列を文節(自立語+付属語)にまとめる。

    直前の文節に付けるもの:
    - 付属語(助詞・助動詞)、接尾語(〜さん・〜的 など)
    - 非自立の動詞・形容詞(補助動詞・補助形容詞: 〜て「いる」・〜て「ほしい」)
    - 接頭辞の直後の語、サ変名詞に続く動詞(研究+し)

    非自立でも**名詞**(形式名詞: 日・こと・とき 等)は自立語として
    新しい文節を始める(「天気の|日に」)。自らアクセントを担いうる単位のため。
    """
    groups: list[list[Word]] = []
    for w in ap.words:
        attach = False
        if groups:
            prev = groups[-1][-1]
            if w.pos in _ATTACH_POS:
                attach = True
            elif w.pos_group1 == "接尾":
                attach = True
            elif w.pos_group1 == "非自立" and w.pos != "名詞":
                attach = True
            elif prev.pos in _PREFIX_POS:
                attach = True
            elif (
                w.pos == "動詞"
                and prev.pos == "名詞"
                and prev.pos_group1 == "サ変接続"
            ):
                attach = True
        if attach:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


@lru_cache(maxsize=4096)
def lexical_accent(surface: str) -> int | None:
    """語を単独で解析したときのアクセント型(辞書型)を返す。

    複数のアクセント句に分かれる語や解析不能な語は None。
    """
    try:
        aps = analyze_text(surface)
    except Exception:
        return None
    if len(aps) != 1:
        return None
    return aps[0].accent_type


def analyze_text_file(path: str) -> list[AccentPhrase]:
    """テキストファイル全体を解析する。改行・空行はポーズとして扱われる。

    文末記号(。!?)と改行で文に区切り、各 AP に sentence_index を付与する。
    """
    lines = [ln.strip() for ln in _read_text_file(path).splitlines()]
    aps: list[AccentPhrase] = []
    sentence_index = 0
    for line in lines:
        if not line:
            continue
        for sent in _SENTENCE_END_RE.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sent_aps = analyze_text(sent)
            if not sent_aps:
                continue
            for ap in sent_aps:
                ap.index = len(aps)
                ap.sentence_index = sentence_index
                if not aps:
                    ap.follows_pause = True
                aps.append(ap)
            sentence_index += 1
    return aps
