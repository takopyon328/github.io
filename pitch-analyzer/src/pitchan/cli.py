"""pitchan コマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import align, f0 as f0mod, normalize, outputs, segment, textproc, tobi

logger = logging.getLogger("pitchan")


@dataclass
class Pair:
    speaker: str
    name: str
    wav: Path
    text: Path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        if args.command == "analyze":
            pairs = [Pair(args.speaker, args.wav.stem, args.wav, args.text)]
        else:
            pairs = _collect_pairs(args.dir, args.speaker, args.out)
            if not pairs:
                logger.error("%s に .wav/.txt ペアが見つかりません", args.dir)
                return 1
        run_pipeline(pairs, args)
        return 0
    except (align.AlignmentError, ValueError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitchan",
        description="日本語朗読音声のアクセント句単位ピッチ(F0)分析",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="単一の wav/text ペアを分析する")
    p_an.add_argument("--wav", type=Path, required=True)
    p_an.add_argument("--text", type=Path, required=True)

    p_ba = sub.add_parser("batch", help="ディレクトリ内の wav/txt ペアを一括分析する")
    p_ba.add_argument(
        "--dir", type=Path, required=True,
        help="同名の .wav/.txt を置いたディレクトリ。話者別サブディレクトリ可",
    )

    for p in (p_an, p_ba):
        p.add_argument("--out", type=Path, required=True, help="出力ディレクトリ")
        p.add_argument("--speaker", default="spk", help="話者 ID(既定 spk)")
        p.add_argument("--f0-floor", type=float, default=60.0)
        p.add_argument("--f0-ceil", type=float, default=500.0)
        p.add_argument("--frame-shift", type=float, default=5.0, help="ms")
        p.add_argument(
            "--ref", default="speaker",
            help="半音変換の基準: speaker / file / value:<Hz>(既定 speaker)",
        )
        p.add_argument("--norm-points", type=int, default=30)
        p.add_argument("--interpolate", action="store_true",
                       help="AP 内の無声区間を線形補間する")
        p.add_argument("--median-filter", action="store_true",
                       help="F0 に 5 点メディアンフィルタを適用する")
        p.add_argument("--plot", action="store_true", help="PNG 可視化を出力する")
        p.add_argument("--plot-ap", action="store_true",
                       help="アクセント句ごとの PNG を <name>_ap_plots/ に出力する")
        p.add_argument("--xjtobi", action="store_true",
                       help="簡易版 X-JToBI 準拠の TextGrid(下書き)を出力する")
        p.add_argument("--bom", action="store_true",
                       help="CSV を BOM 付き UTF-8 で出力する(Excel 用)")
        p.add_argument("--jobs", type=int, default=4,
                       help="並列数(F0 抽出・MFA)")
        p.add_argument("--beam", type=int, default=100)
        p.add_argument("--retry-beam", type=int, default=400)
        p.add_argument("--acoustic-model", default="japanese_mfa")
        p.add_argument("--g2p-model", default="japanese_mfa")
    return parser


def _collect_pairs(
    root: Path, default_speaker: str, exclude: Path | None = None
) -> list[Pair]:
    pairs = []
    # 出力フォルダが入力フォルダの内側にある場合、過去の実行でコピーされた
    # 音声(work/corpus 等)を拾わないよう除外する。
    exclude_resolved = exclude.resolve() if exclude is not None else None
    for wav in sorted(root.rglob("*.wav")):
        if exclude_resolved is not None and exclude_resolved in wav.resolve().parents:
            continue
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            logger.warning("%s: 対応する .txt がないためスキップ", wav.name)
            continue
        rel = wav.relative_to(root)
        speaker = rel.parts[0] if len(rel.parts) > 1 else default_speaker
        pairs.append(Pair(speaker, wav.stem, wav, txt))
    return pairs


def _extract_f0_worker(task: tuple) -> tuple[np.ndarray, np.ndarray, float]:
    wav_path, floor, ceil, shift, median = task
    x, sr = f0mod.load_wav(str(wav_path))
    t, f0 = f0mod.extract_f0(
        x, sr, f0_floor=floor, f0_ceil=ceil,
        frame_shift_ms=shift, median_filter=median,
    )
    return t, f0, len(x) / sr


def run_pipeline(pairs: list[Pair], args) -> None:
    out_dir: Path = args.out
    work_dir = out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) 言語処理
    logger.info("言語処理: %d ファイル", len(pairs))
    all_aps: dict[str, list[textproc.AccentPhrase]] = {}
    items: list[align.FileItem] = []
    for pr in pairs:
        aps = textproc.analyze_text_file(str(pr.text))
        if not aps:
            raise ValueError(f"{pr.text}: アクセント句が得られませんでした")
        all_aps[pr.name] = aps
        tokens = [w.pron for ap in aps for w in ap.words]
        items.append(align.FileItem(pr.speaker, pr.name, pr.wav, tokens))
        logger.info("  %s: %d アクセント句 / %d 語", pr.name, len(aps), len(tokens))

    # (2) 強制アラインメント(コーパス一括)
    align.check_mfa_available()
    corpus_dir = work_dir / "corpus"
    aligned_dir = work_dir / "aligned"
    align.prepare_corpus(items, corpus_dir)
    logger.info("発音辞書を生成中 (mfa g2p)...")
    dict_path = align.build_g2p_dictionary(items, work_dir, args.g2p_model)
    logger.info("アラインメント中 (mfa align)... 長尺ファイルでは時間がかかります")
    align.run_align(
        corpus_dir, dict_path, aligned_dir,
        acoustic_model=args.acoustic_model,
        beam=args.beam, retry_beam=args.retry_beam, num_jobs=args.jobs,
    )
    all_phones: dict[str, list] = {}
    for item in items:
        tg_path = aligned_dir / item.speaker / f"{item.name}.TextGrid"
        if not tg_path.exists():
            raise align.AlignmentError(
                f"{tg_path} が生成されていません(アラインメント失敗)"
            )
        words, phones = align.read_word_intervals(tg_path, item.tokens)
        segment.assign_times(all_aps[item.name], words)
        all_phones[item.name] = phones

    # (3) F0 抽出
    logger.info("F0 抽出中 (WORLD harvest, jobs=%d)...", args.jobs)
    tasks = [
        (pr.wav, args.f0_floor, args.f0_ceil, args.frame_shift, args.median_filter)
        for pr in pairs
    ]
    if args.jobs > 1 and len(pairs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_extract_f0_worker, tasks))
    else:
        results = [_extract_f0_worker(t) for t in tasks]
    f0_data = {pr.name: res for pr, res in zip(pairs, results)}

    # (4) 正規化基準
    refs = _compute_refs(pairs, f0_data, args.ref)

    # (5) 出力
    for pr in pairs:
        times, f0_raw, dur = f0_data[pr.name]
        aps = all_aps[pr.name]
        f0_hz = f0_raw
        if args.interpolate:
            spans = [(ap.t_start, ap.t_end) for ap in aps if ap.t_start is not None]
            f0_hz = f0mod.interpolate_unvoiced_in_spans(times, f0_raw, spans)
        ref_hz, mu, sigma = refs[pr.name]
        f0_st = normalize.to_semitone(f0_hz, ref_hz)
        f0_z = normalize.to_log_z(f0_hz, mu, sigma)

        frames = outputs.build_frames_df(pr.name, times, f0_hz, f0_st, f0_z, aps)
        summary = outputs.build_ap_summary_df(pr.name, times, f0_st, aps)
        bpm = tobi.classify_bpm_all(times, f0_st, all_phones[pr.name], aps)
        summary["bpm_auto"] = summary["ap_index"].map(bpm)
        peaks = tobi.peak_excl_bpm_all(times, f0_st, all_phones[pr.name], aps, bpm)
        summary["peak_excl_bpm_st"] = summary["ap_index"].map(
            lambda i: round(peaks[i][0], 3)
        )
        summary["peak_excl_bpm_time"] = summary["ap_index"].map(
            lambda i: round(peaks[i][1], 4)
        )
        contours = outputs.build_contours_df(
            pr.name, times, f0_st, aps, args.norm_points
        )
        enc = {"index": False, "encoding": "utf-8-sig" if args.bom else "utf-8"}
        frames.to_csv(out_dir / f"{pr.name}_frames.csv", **enc)
        summary.to_csv(out_dir / f"{pr.name}_ap_summary.csv", **enc)
        contours.to_csv(out_dir / f"{pr.name}_ap_contours.csv", **enc)
        params = {
            "speaker": pr.speaker, "ref_hz": round(ref_hz, 2),
            "f0_floor": args.f0_floor, "f0_ceil": args.f0_ceil,
            "frame_shift_ms": args.frame_shift,
            "interpolate": args.interpolate, "norm_points": args.norm_points,
        }
        outputs.write_json(out_dir / f"{pr.name}.json", pr.name, aps, params, contours)
        outputs.write_textgrid(
            out_dir / f"{pr.name}.TextGrid", aps, all_phones[pr.name], dur
        )
        if args.xjtobi:
            tobi.write_xjtobi_textgrid(
                out_dir / f"{pr.name}_xjtobi.TextGrid",
                aps, all_phones[pr.name], dur, times, f0_st,
            )
        if args.plot:
            outputs.plot_f0(out_dir / f"{pr.name}_f0.png", times, f0_st, aps)
        if args.plot_ap:
            outputs.plot_ap_pngs(out_dir / f"{pr.name}_ap_plots", times, f0_st, aps)
        n_low = sum(ap.low_confidence for ap in aps)
        logger.info(
            "  %s: %d AP 出力 (low_confidence %d 件, ref=%.1f Hz)",
            pr.name, len(aps), n_low, ref_hz,
        )
    logger.info("完了: %s", out_dir)


def _compute_refs(pairs, f0_data, ref_opt: str) -> dict[str, tuple[float, float, float]]:
    """ファイルごとの (基準F0[Hz], logμ, logσ) を計算する(キーはファイル名)。"""
    refs: dict[str, tuple[float, float, float]] = {}
    if ref_opt == "file":
        for pr in pairs:
            refs[pr.name] = normalize.speaker_reference([f0_data[pr.name][1]])
        return refs

    by_spk: dict[str, tuple[float, float, float]] = {}
    for spk in sorted({pr.speaker for pr in pairs}):
        arrays = [f0_data[pr.name][1] for pr in pairs if pr.speaker == spk]
        by_spk[spk] = normalize.speaker_reference(arrays)
    if ref_opt.startswith("value:"):
        ref_hz = float(ref_opt.split(":", 1)[1])
        by_spk = {s: (ref_hz, mu, sigma) for s, (_, mu, sigma) in by_spk.items()}
    elif ref_opt != "speaker":
        raise ValueError(f"--ref の値が不正です: {ref_opt}")
    for pr in pairs:
        refs[pr.name] = by_spk[pr.speaker]
    return refs


if __name__ == "__main__":
    sys.exit(main())
