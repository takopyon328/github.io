#!/usr/bin/env python3
"""テスト用の偽 mfa コマンド。

g2p: 単語リストの各語にダミー発音を与えた辞書を書く。
align: コーパスの .lab を読み、1 語 0.3 秒の TextGrid を生成する。
"""

import sys
from pathlib import Path


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]
    # フラグの値(--beam 100 など)を位置引数から除外する
    # (--fine_tune は値なしフラグなので valued に含めない)
    valued = {"--beam", "--retry_beam", "--num_jobs", "--fine_tune_boundary_tolerance"}
    skip_next = False
    positional = []
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in valued:
            skip_next = True
            continue
        if a.startswith("--"):
            continue
        positional.append(a)

    cmd = positional[0]
    if cmd == "g2p":
        wordlist, _model, out = positional[1], positional[2], positional[3]
        words = Path(wordlist).read_text(encoding="utf-8").split()
        Path(out).write_text(
            "".join(f"{w}\tp a\n" for w in words), encoding="utf-8"
        )
        return 0
    if cmd == "align":
        corpus, _dict, _model, out = (
            positional[1], positional[2], positional[3], positional[4]
        )
        from praatio import textgrid as ptg

        for lab in Path(corpus).rglob("*.lab"):
            tokens = lab.read_text(encoding="utf-8").split()
            entries = []
            t = 0.5
            for tok in tokens:
                entries.append((t, t + 0.3, tok))
                t += 0.3
            total = t + 0.5
            tg = ptg.Textgrid()
            tg.addTier(ptg.IntervalTier("words", entries, 0, total))
            tg.addTier(ptg.IntervalTier("phones", entries, 0, total))
            spk = lab.parent.name
            out_dir = Path(out) / spk
            out_dir.mkdir(parents=True, exist_ok=True)
            tg.save(
                str(out_dir / f"{lab.stem}.TextGrid"),
                format="long_textgrid",
                includeBlankSpaces=True,
            )
        return 0
    print(f"fake mfa: unknown command {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
