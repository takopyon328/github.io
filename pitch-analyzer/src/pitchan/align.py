"""Montreal Forced Aligner(MFA)による単語レベル強制アラインメント。

転記トークンには NJD のカタカナ読みを使い、発音辞書は mfa g2p で
コーパス専用に自動生成する。これにより OOV と音素セット変換の問題を回避する。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from praatio import textgrid as ptg

logger = logging.getLogger(__name__)

# MFA が words tier に出力する非単語ラベル
_NON_WORD_LABELS = {"", "sil", "sp", "spn", "<eps>", "<unk>"}


class AlignmentError(RuntimeError):
    pass


@dataclass
class FileItem:
    """1 音声ファイル分の入力。tokens は AP を構成する単語のカタカナ読み列。"""

    speaker: str
    name: str
    wav_path: Path
    tokens: list[str]


def check_mfa_available() -> None:
    if shutil.which("mfa") is None:
        raise AlignmentError(
            "mfa コマンドが見つかりません。conda 環境に Montreal Forced Aligner を"
            "インストールし、モデルをダウンロードしてください(README 参照)。"
        )


def prepare_corpus(items: list[FileItem], corpus_dir: Path) -> None:
    """MFA 用コーパスディレクトリ(speaker/name.wav + name.lab)を作る。"""
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    for item in items:
        spk_dir = corpus_dir / item.speaker
        spk_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.wav_path, spk_dir / f"{item.name}.wav")
        (spk_dir / f"{item.name}.lab").write_text(
            " ".join(item.tokens) + "\n", encoding="utf-8"
        )


def build_g2p_dictionary(
    items: list[FileItem], work_dir: Path, g2p_model: str = "japanese_mfa"
) -> Path:
    """コーパス中の全単語タイプの発音辞書を mfa g2p で生成する。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    types = sorted({tok for item in items for tok in item.tokens})
    wordlist = work_dir / "wordlist.txt"
    wordlist.write_text("\n".join(types) + "\n", encoding="utf-8")
    dict_path = work_dir / "corpus_dict.txt"
    _run_mfa_command(
        ["mfa", "g2p", str(wordlist), g2p_model, str(dict_path), "--clean"]
    )
    if not dict_path.exists():
        raise AlignmentError("mfa g2p が辞書を生成しませんでした")
    _check_dictionary_coverage(dict_path, types)
    return dict_path


def _check_dictionary_coverage(dict_path: Path, types: list[str]) -> None:
    covered = set()
    for line in dict_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            covered.add(line.split("\t")[0].split(" ")[0])
    missing = [t for t in types if t not in covered]
    if missing:
        logger.warning(
            "G2P で発音を生成できなかった語が %d 件あります(先頭 10 件): %s",
            len(missing), missing[:10],
        )


def run_align(
    corpus_dir: Path,
    dict_path: Path,
    out_dir: Path,
    acoustic_model: str = "japanese_mfa",
    beam: int = 100,
    retry_beam: int = 400,
    num_jobs: int = 4,
    fine_tune: bool = False,
    fine_tune_boundary_tolerance: float | None = None,
) -> None:
    """mfa align を実行する。out_dir に speaker/name.TextGrid が生成される。

    fine_tune=True で MFA の境界微調整(--fine_tune)を有効にする。
    fine_tune_boundary_tolerance は指定時のみ渡す。
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # MFA(click)は --beam 等の追加設定オプションを位置引数の後に置く必要がある
    # (前に置くと値がコーパスパスとして解釈される)
    cmd = [
        "mfa", "align",
        str(corpus_dir), str(dict_path), acoustic_model, str(out_dir),
        "--clean", "--overwrite",
        "--beam", str(beam), "--retry_beam", str(retry_beam),
        "--num_jobs", str(num_jobs),
    ]
    if fine_tune:
        cmd.append("--fine_tune")
    if fine_tune_boundary_tolerance is not None:
        cmd += ["--fine_tune_boundary_tolerance", str(fine_tune_boundary_tolerance)]
    _run_mfa_command(cmd)


def _run_mfa_command(cmd: list[str]) -> None:
    logger.info("実行: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AlignmentError(
            f"MFA コマンドが失敗しました ({proc.returncode}):\n"
            f"{' '.join(cmd)}\n--- stderr ---\n{proc.stderr[-3000:]}"
        )


def read_word_intervals(
    tg_path: Path, expected_tokens: list[str]
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """MFA 出力 TextGrid から単語区間列と音素区間列を読み取る。

    単語区間列は expected_tokens と 1:1 対応していることを検証する。
    """
    tg = ptg.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    tier_names = tg.tierNames
    word_tier = next((n for n in tier_names if n.endswith("words")), None)
    phone_tier = next((n for n in tier_names if n.endswith("phones")), None)
    if word_tier is None:
        raise AlignmentError(f"{tg_path}: words tier が見つかりません ({tier_names})")

    words = [
        (e.start, e.end, e.label)
        for e in tg.getTier(word_tier).entries
        if e.label.strip() not in _NON_WORD_LABELS
    ]
    phones = (
        [
            (e.start, e.end, e.label)
            for e in tg.getTier(phone_tier).entries
            if e.label.strip()
        ]
        if phone_tier
        else []
    )

    labels = [w[2] for w in words]
    if labels != expected_tokens:
        n_show = next(
            (i for i, (a, b) in enumerate(zip(labels, expected_tokens)) if a != b),
            min(len(labels), len(expected_tokens)),
        )
        raise AlignmentError(
            f"{tg_path}: アラインメント結果の単語列がテキストと一致しません "
            f"(結果 {len(labels)} 語 / 期待 {len(expected_tokens)} 語, "
            f"最初の不一致位置 {n_show}: "
            f"{labels[n_show:n_show+3]} vs {expected_tokens[n_show:n_show+3]})"
        )
    return words, phones
