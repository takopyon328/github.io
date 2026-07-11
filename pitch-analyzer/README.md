# pitchan — 日本語朗読音声のアクセント句単位ピッチ分析

朗読音声(WAV)と朗読テキストから、アクセント句ごとに分割した正規化 F0 を出力する
コマンドラインツールです。設計の詳細は [SPEC.md](SPEC.md) を参照してください。

## インストール

Montreal Forced Aligner(MFA)を使うため conda 環境を推奨します。

```bash
# 1. conda 環境の作成と MFA の導入
conda create -n pitchan python=3.11 -y
conda activate pitchan
conda install -c conda-forge montreal-forced-aligner -y

# 2. MFA の日本語モデルをダウンロード(初回のみ)
mfa model download acoustic japanese_mfa
mfa model download g2p japanese_mfa

# 3. 本パッケージのインストール(このディレクトリで)
pip install -e ".[plot]"
```

## 使い方

### 一括処理(推奨)

同名の `.wav` / `.txt` ペアを 1 つのディレクトリに置きます:

```
data/
  chapter01.wav
  chapter01.txt
  chapter02.wav
  chapter02.txt
  ...
```

```bash
pitchan batch --dir data/ --out results/ --plot --bom
```

- 既定では**ディレクトリ全体を 1 話者**として扱い、半音変換の基準
  (有声フレームの幾何平均)を全ファイルから計算します(`--ref speaker`)。
- 複数話者を扱う場合は `data/<話者ID>/xxx.wav` のように話者別サブディレクトリに
  分けてください。正規化は話者ごとに行われます。

### 単一ファイル

```bash
pitchan analyze --wav recording.wav --text script.txt --out results/
```

### 主なオプション

| オプション | 既定 | 説明 |
|-----------|------|------|
| `--f0-floor` / `--f0-ceil` | 60 / 500 | F0 探索範囲 [Hz]。男性話者なら `--f0-ceil 350` 程度に下げると倍ピッチ誤りが減ります |
| `--frame-shift` | 5 | フレームシフト [ms] |
| `--ref` | speaker | 半音変換の基準: `speaker` / `file` / `value:<Hz>` |
| `--norm-points` | 30 | 時間正規化輪郭の点数 |
| `--interpolate` | off | アクセント句内の無声区間を線形補間 |
| `--median-filter` | off | F0 の 5 点メディアンフィルタ(倍/半ピッチ誤り緩和) |
| `--adaptive-range` | off | 話者ごとに 2 パスで F0 探索範囲を自動推定(第 1 パス 60–600 Hz の有声フレーム四分位から floor=0.75×Q25 / ceil=1.5×Q75)。`--f0-floor/--f0-ceil` の手動指定より優先。実際に使われた値は各 `.json` の `f0_floor/f0_ceil` に記録 |
| `--split-sentences` | off | **推奨**: 文末記号(。!?・改行)と無音検出でファイルを文単位に分割してアラインメントする。言い淀み等の不一致があってもその文だけの失敗で済み(該当句は `low_confidence`・時刻なしで出力)、長尺ファイルの境界精度も上がる |
| `--plot` | off | ファイル全体の F0 曲線+句境界の PNG を出力 |
| `--plot-ap` | off | アクセント句ごとの PNG を `<name>_ap_plots/` に出力(表記・読み・アクセント型・単語境界つき)。縦軸は既定で**句ごとの自動スケール**(最小スパン 8 半音)。句どうしの高さ比較には `--plot-ap-shared-ylim` でファイル内共通スケールに切替 |
| `--xjtobi` | off | 簡易版 X-JToBI(五十嵐 2015)準拠の TextGrid 下書きを `<name>_xjtobi.TextGrid` に出力 |
| `--bom` | off | CSV を BOM 付き UTF-8 で出力(Excel で開く場合) |
| `--jobs` | 4 | 並列数(F0 抽出と MFA に適用) |
| `--fine-tune-alignment` | off | MFA の境界微調整(`--fine_tune`)を有効化。音素・単語境界を 10ms 刻みの通常出力より細かく(1ms 刻みで)推定し直す。**1ms 刻みの出力は 1ms の正確さを保証しない**点に注意 |
| `--fine-tune-boundary-tolerance` | なし | MFA `--fine_tune_boundary_tolerance` に渡す値。指定すると `--fine-tune-alignment` も有効として扱う |

## 出力ファイル

`results/` に音声ファイルごとに生成されます:

| ファイル | 内容 |
|---------|------|
| `<name>_frames.csv` | フレーム単位(5ms)の F0: 生値 [Hz]、半音値 `f0_st`、z スコア `f0_z`、所属アクセント句 |
| `<name>_ap_summary.csv` | アクセント句単位の要約: 表記・カナ・アクセント型・モーラ数・時刻・平均/最大/最小/レンジ(半音)・ピーク位置比・有声率・信頼度フラグ |
| `<name>_ap_contours.csv` | 各句の F0 を等間隔 30 点にリサンプルした時間正規化輪郭(半音値)。句の形状比較用 |
| `<name>.json` | 上記を統合した構造化データ(単語タイミング含む) |
| `<name>.TextGrid` | Praat で開いて境界を確認・手修正するための TextGrid |
| `<name>_f0.png` | (`--plot` 時)F0 曲線+句境界の図 |

`results/work/` に MFA の中間ファイル(コーパス・生成辞書・アラインメント結果)が
残るので、アラインメントの検証に使えます。

## 局所再アラインメント(`refine`)

pitchan が生成した TextGrid を**初期値**として、対応する WAV と朗読テキストから
単語境界を局所的に整列し直し、修正版 TextGrid を生成します。**入力 TextGrid は
上書きしません**。

```bash
pitchan refine --wav recording.wav --text script.txt \
    --textgrid results/recording.TextGrid --out results/refine/
```

仕組み: 単語列を core block(既定 5 語)に分け、前後 2 語の context と 0.30 秒の
margin を付けて音声を切り出し、全 block を 1 つの一時コーパスにまとめて MFA
(`--fine_tune` 既定 ON)で一括再アラインメントします。採否は block 単位:

| status | 条件 | 動作 |
|---|---|---|
| `AUTO_ACCEPT` | 候補が有効かつ最大移動量 ≤ `--auto-accept-shift-ms`(既定 80) | 候補を自動採用 |
| `REVIEW` | 有効だが移動量が 80〜`--hard-max-shift-ms`(既定 250) | 既定では元の境界を維持し候補を CSV に保存。`--apply-review` 指定時のみ適用(status は REVIEW のまま) |
| `KEEP_ORIGINAL` | MFA 失敗・ラベル不一致・時刻の逆転/重複/範囲外・hard max 超過・統合検証失敗 | 元の境界を維持 |

出力(`<stem>_` は入力 TextGrid 名):

- `<stem>_refined.TextGrid` — tier: `accent_phrases` / `words` / `phones`(構築可能な場合)
  + `*_original`(入力の保存)+ `alignment_review`(AUTO_ACCEPT 以外の block の
  status・移動量・理由。例 `REVIEW|block=12|max_shift_ms=137.4|reason=large_shift`)
- `<stem>_alignment_diff.csv` — 1 語 1 行。修正前後の時刻・候補・採否・理由
- `<stem>_refine_summary.json` — 実行オプション・件数・移動量の中央値/90 パーセンタイル・警告
- `work/refine/` — MFA の中間ファイル

注意:

- **テキストと発話内容が一致している朗読音声が前提**です。読み飛ばし・言い直し・
  語の挿入や置換の自動処理(音声認識による補正)は対象外で、その場合は明確な
  エラーまたは KEEP_ORIGINAL になります。
- 閾値(80ms / 250ms / margin 0.30s など)は**暫定値**です。手修正済みデータと
  突き合わせて較正した上で本採用してください。
- MFA の fine-tune は境界を 1ms 刻みで出力しますが、**1ms の正確さを保証する
  ものではありません**。

## 簡易版 X-JToBI 出力(`--xjtobi`)

簡易版 X-JToBI(五十嵐 2015)準拠の 4 層 TextGrid を**下書き**として出力します:

| 層 | 内容 | 由来 |
|----|------|------|
| `segments` | 音素区間 | MFA アラインメント |
| `tones` | `%L`(イントネーション句頭)・`H-`(句頭上昇。頭高句を除く)・`H*+L`(予測核)・`L%`(句末。BPM 判定があれば `L%H%` のように連結)(ポイント層) | %L/H-/H*+L はテキスト予測の近似位置、BPM は F0 形状の規則判定(ドラフト) |
| `words` | カナ単語+アクセント核記号 `'`(例: ヤマナシダ'イガク)。**手修正はこの層に対して行う** | テキスト予測(東京方言の規範) |
| `words_pred` | words の予測の凍結コピー+語の辞書型(例: `ヤマナシダ'イガク/5`)。**編集しない**こと(実現 vs 予測の対照の基準になる) | テキスト予測+語単独の解析 |
| `BI` | 1=語境界 / 2=アクセント句境界 / 3=イントネーション句境界(ポイント層) | 1・2 はテキスト予測、3 は実ポーズ(0.2 秒以上)で検出 |

BPM の自動判定は X-JToBI の 4 種すべてに対応しています:
`H%`(上昇調1)/ `LH%`(上昇調2)/ `HL%`(上昇下降調)/ `HLH%`(上昇下降上昇調)。
H% と LH% は X-JToBI の弁別基準に従い**上昇開始の時間的な遅れ**で分けます
(上昇開始が最終モーラの前半なら H%、後半まで低く保たれてから上昇するなら LH%)。
判定区間は資料の手順どおり分節音層から特定した**最終モーラ**です(末尾の母音的
音素の開始〜句末。長母音 `ː` 末の句では音素の後半のみ)。

### 手修正後のラベル駆動計測(`xjtobi-measure`)

`<name>_xjtobi.TextGrid` を Praat で手修正(核記号の移動・削除、BI の変更、BPM の追加・修正)
した後、**修正後のラベルに基づいて** F0 を計測し直せます(五十嵐方式の分析手順):

```bash
pitchan xjtobi-measure --wav rec.wav --textgrid rec_xjtobi_fixed.TextGrid --out results/ --bom
```

`<name>_xjtobi_measures.csv` が生成されます:
`ap_kana / nucleus_mora(核記号の位置。0=平板)/ accented / t_start / t_end / bi / bpm /`
`f0_mean_st / f0_max_st / f0_max_time / peak_excl_bpm_st / peak_excl_bpm_time / voiced_ratio`

- 句の区切りは **BI 層の 2・3 の位置**から、核は **words 層の `'` の位置**から、
  BPM は **tones 層のラベル**から読み取ります。手修正がそのまま計測に反映されます。
- 半音変換の基準は既定でそのファイル単体(`--ref file`)。batch の結果と比較する場合は、
  該当ファイルの `.json` にある `ref_hz` を `--ref value:<Hz>` で指定してください。

あわせて**単語レベルのアクセント対照**も出力されます:

- `<name>_xjtobi_words.csv` — 1 語 1 行:
  `realized_accent`(実現型: 修正後 words 層の `'` の語内位置。0=核なし)/
  `predicted_accent`(予測型: words_pred 層)/
  `lexical_accent`(辞書型: 語単独のアクセント)/
  `accent_match`(`match`=一致 / `shifted`=核位置ずれ / `deleted`=核脱落 / `inserted`=核過剰)
- `<name>_accent.TextGrid` — 実現アクセント型を語区間ラベルにした `accent_est` 層
  (Praat での目視確認用)

「辞書型 → 文脈予測型 → 実現型」の 3 点比較により、単語アクセント知識の問題か、
句を形成する際の問題かの切り分けに使えます。実現型の実体はアノテータが words 層に
置いた核記号であり、音響からの自動推定ではない点に注意してください。

- 核記号と BI=2 は**規範(予測)**です。実際の発話の記述としては Praat 上で手修正してください。
  修正前(規範)と修正後(実現)の差分が、話者の韻律的逸脱のデータになります。
- BPM の自動判定はドラフト品質です。`ap_summary.csv` の `bpm_auto` 列にも同じ判定が
  入るので、まず数ファイルで聴覚判断との一致を確認してから利用してください。
- `tones` 層の `%L`/`H-`/`H*+L` の位置はモーラ数からの線形近似です(簡易版の方針
  どおり F0 曲線への正確な同期は行いません)。
- `ap_summary.csv` に五十嵐方式の計測列が入ります:
  `peak_excl_bpm_st` / `peak_excl_bpm_time` = **BPM 区間(句末最終音素)を除いた**
  句内最大 F0(半音)とその時刻。核ピークの計測は有核句(`accent_type >= 1`)に
  絞って利用してください。BPM がない句では通常の最大値と一致します。

## 正規化の定義

- **半音値**: `f0_st = 12 × log2(F0 / F0_ref)`。`F0_ref` は話者の全有声フレームの
  幾何平均(既定)。話者間で声の高さの違いを除いた比較ができます。
- **z スコア**: log F0 の話者単位 z スコア。レンジの個人差も除きたい場合に使用。
- **時間正規化輪郭**: 各アクセント句の半音値を句内相対時間 0–1 上の等間隔 30 点に
  線形補間でリサンプルしたもの。無声フレームは補間に使いません
  (句内の有声フレームが 4 未満の句は全欠損になります)。

## 注意・既知の限界

- **読み・アクセント型は OpenJTalk の推定**です。固有名詞・数字などで読みを誤ることが
  あります。`ap_summary` の `accent_type` を使う際は精度に注意してください
  (句境界の誤りは TextGrid を Praat で確認できます)。
- **テキストと音声が一致していること**が前提です。読み飛ばし・言い直しが多いと
  アラインメントが破綻します。15 分程度の長尺でも動作しますが、不一致が多い場合は
  段落単位で音声・テキストを分割すると頑健になります。
- `low_confidence=1` は次のいずれかに該当する句に付きます。集計前に除外を検討してください。
  - 1 モーラあたりの長さが極端(30ms 未満 / 500ms 超)= アラインメントの疑い
  - 有声率 50% 未満 / 隣接有声フレーム間の 8 半音超の跳躍(倍・半ピッチ誤りの疑い)/
    0.4 秒超の連続無声 = F0 抽出の疑い

## 開発

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

テストは MFA を偽コマンドでモックするため、MFA なしで全件実行できます。
