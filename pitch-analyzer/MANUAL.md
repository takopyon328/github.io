# pitchan 利用マニュアル

日本語朗読音声のアクセント句単位ピッチ(F0)分析ツール

対象読者: 音声・韻律研究者(プログラミング経験は前提としません)
技術仕様は [SPEC.md](SPEC.md)、簡潔なリファレンスは [README.md](README.md) を参照。

---

## 1. このツールでできること

朗読音声(WAV)と朗読テキストから、次を自動生成します。

1. **アクセント句単位の分割** — テキストからアクセント句を推定し、強制アラインメント
   (MFA)で音声上の時刻を確定
2. **正規化 F0** — 話者の声の高さの違いを除いた半音値・z スコア・時間正規化輪郭
3. **簡易版 X-JToBI(五十嵐 2015)の下書き** — Praat で手修正し、修正後のラベルに
   基づく再計測まで一巡できる
4. **境界の精密化(refine)** — 生成済み TextGrid の単語境界を局所的に整列し直す

処理の流れ:

```
テキスト ─┐
          ├→ アクセント句推定 → MFA アラインメント → F0 抽出 → 正規化 → 出力
音声 WAV ─┘                                                      (CSV/JSON/TextGrid/PNG)
                                     ↓
                     (任意) X-JToBI 下書き → Praat で手修正 → xjtobi-measure で再計測
                     (任意) refine で境界を精密化
```

### 重要な前提

- **テキストと実際の発話内容が一致していること**。読み飛ばし・言い直しがある場合は
  テキスト側を発話に合わせて修正してください(→ §10)
- アクセント句の区切り・アクセント型は **OpenJTalk によるテキストからの予測(東京方言の
  規範)** です。実際の発話の記述は Praat での手修正で行います(→ §7)

---

## 2. セットアップ(初回のみ)

Windows での手順。Montreal Forced Aligner(MFA)を使うため conda 環境が必要です。

1. **Miniconda をインストール** — https://www.anaconda.com/download/success
   から Miniconda を入れる。以後の操作はスタートメニューの
   「Anaconda Prompt (Miniconda3)」で行う

2. **環境の作成と MFA の導入**(すべて conda-forge から一括で入れること。
   標準チャンネルと混ぜると DLL エラーが起きる → §10)

   ```
   conda create -n pitchan -c conda-forge --override-channels python=3.11 montreal-forced-aligner -y
   conda activate pitchan
   mfa version
   ```

3. **日本語モデルのダウンロード**

   ```
   mfa model download acoustic japanese_mfa
   mfa model download g2p japanese_mfa
   ```

4. **pitchan 本体の取得とインストール**

   ```
   git clone -b claude/japanese-pitch-analysis-mft9xp https://github.com/takopyon328/takopyon328.github.io.git
   cd takopyon328.github.io\pitch-analyzer
   pip install -e ".[plot]"
   pitchan --help
   ```

### 毎回の起動手順

1. Anaconda Prompt を開く
2. `conda activate pitchan`(行頭が `(pitchan)` になる)
3. `pitchan ...` を実行

### 更新手順(ツールが修正されたとき)

```
cd takopyon328.github.io\pitch-analyzer
git pull
```

`pip install -e` 方式なので `git pull` だけで反映されます。

---

## 3. データの準備

### ファイルの置き方

**拡張子だけが違う同名の** `.wav` と `.txt` をペアで置きます。

```
onsei\
    frjpnu005tex_mr_yomiage.wav
    frjpnu005tex_mr_yomiage.txt
    frjpnu013tex_bg_yomiage.wav
    frjpnu013tex_bg_yomiage.txt
    ...
```

- **1 話者 = 1 ファイルの場合**(現在の 7 名構成): 上のようにフラットに置き、
  実行時に `--ref file` を付ける(各話者が自分の声の高さ基準で正規化される)
- **1 話者 = 複数ファイルの場合**: 話者ごとのサブフォルダに分け、`--ref speaker`
  (既定)で実行する。フォルダ名が話者 ID になる

  ```
  onsei\
      話者A\ part1.wav part1.txt part2.wav part2.txt
      話者B\ ...
  ```

  ※ フラットに複数話者を置いて `--ref speaker` にすると全員が混ざった基準で
  正規化されてしまうので注意。話者をまたいで同じファイル名も不可(出力が衝突)

### テキストの書き方

- 漢字かな交じりの普通のテキストで OK(読みはツールが推定)
- 文字コードは UTF-8 または Shift_JIS(自動判別)
- **実際に読まれた内容と一字一句合わせる**(最重要)。言い直しがあれば
  「やま、山梨大学」のように発話どおりに書く
- 1 文ごとに改行を推奨。**文の途中で改行しない**(改行はポーズとして扱われる)
- 数字・英字・特殊な読みの固有名詞は読まれたとおりに開く
  (「2020年」→「二千二十年」、「AI」→「エーアイ」)
- 注記記号(括弧・※・スラッシュ等)は入れない。句読点(、。)は入れてよい
- タイトル・話者ラベルなど音声に存在しない文字列は削除

---

## 4. 基本の分析(batch)

7 名分をまとめて分析する標準的なコマンド:

```
pitchan batch --dir C:\Users\nunom\onsei --out C:\Users\nunom\onsei\results ^
    --ref file --split-sentences --adaptive-range --plot-ap --xjtobi --bom
```

(`^` は Anaconda Prompt での行継続。1 行で書いても同じ)

推奨オプションの意味:

| オプション | 推奨理由 |
|---|---|
| `--ref file` | 1 話者 1 ファイル構成での話者別正規化(§3) |
| `--split-sentences` | 文単位でアラインメント。言い淀み等の不一致があっても失敗がその文に閉じ、長尺でも安定 |
| `--adaptive-range` | 話者ごとに F0 探索範囲を自動推定(男女混在データで倍・半ピッチ誤りが減る) |
| `--plot-ap` | 句ごとの F0 図(§5) |
| `--xjtobi` | 簡易版 X-JToBI 下書き(§7) |
| `--bom` | CSV を Excel で文字化けなく開ける形式に |

単一ファイルなら `pitchan analyze --wav X.wav --text X.txt --out results\ ...`。

### ログの読み方

- `N 文 → M 発話に分割` — 文単位分割の結果
- `N/M 発話が失敗` — その文だけスキップされた(該当句は時刻なし・low_confidence で出力)
- `low_confidence N 件` — 要確認の句の数
- `有声フレームの X% が探索範囲外で無声化` — F0 レンジが狭すぎる可能性の警告

---

## 5. 出力ファイル

`results\` に音声ファイルごとに生成されます。

| ファイル | 内容 |
|---|---|
| `<名前>_ap_summary.csv` | **アクセント句単位の要約(主に使うファイル)** |
| `<名前>_frames.csv` | 5ms ごとの F0(生値 Hz・半音値・z スコア・所属句) |
| `<名前>_ap_contours.csv` | 各句の F0 を等間隔 30 点にした時間正規化輪郭(形状比較用) |
| `<名前>.json` | 全情報の構造化データ(正規化パラメータの記録つき) |
| `<名前>.TextGrid` | Praat 用(accent_phrases / words / phones 層)。境界の目視確認に |
| `<名前>_xjtobi.TextGrid` | 簡易版 X-JToBI 下書き(§7) |
| `<名前>_ap_plots\` | 句ごとの F0 図。ファイル名は `ap0001_ヤマナシダイガクデ.png` |
| `<名前>_f0.png` | ファイル全体の F0 図(`--plot` 指定時) |
| `results\work\` | MFA の中間ファイル(通常は見なくてよい) |

### ap_summary.csv の主な列

| 列 | 意味 | 由来 |
|---|---|---|
| `ap_surface` / `ap_kana` | 句の表記・読み | テキスト予測 |
| `accent_type` | アクセント型(0=平板、1=頭高…) | テキスト予測(規範) |
| `follows_pause` | ポーズ直後の句か | テキスト(句読点・改行) |
| `t_start` / `t_end` / `duration_sec` | 句の時刻 | 音響(アラインメント) |
| `f0_mean_st` / `f0_max_st` / `f0_min_st` / `f0_range_st` | 半音値の統計 | 音響 |
| `peak_time_ratio` | F0 ピークの句内相対位置(0〜1) | 音響 |
| `peak_excl_bpm_st` / `_time` | 句末境界音調(BPM)区間を除いた最大 F0(核ピーク計測用。有核句で使う) | 音響 |
| `bpm_auto` | 句末境界音調の自動判定(H% / LH% / HL% / HLH%)※ドラフト品質 | 音響 |
| `voiced_ratio` | 有声フレーム率 | 音響 |
| `low_confidence` | 1 なら要確認(アラインメント・F0 抽出に疑い)。**集計前に除外を検討** | 判定 |

**「テキスト予測」の列は全話者・全読みで同一**(同じテキストなら句数・型も同一)。
だから `ap_index` や `ap_kana` をキーに話者間で同じ句を直接比較できます。

---

## 6. 正規化の定義

- **半音値** `f0_st = 12 × log2(F0 / 基準F0)`。基準は話者の全有声フレームの幾何平均。
  話者間で声の高さの違いを除いた比較ができる。使った基準値は `.json` の `ref_hz`
- **z スコア** log F0 の話者単位 z スコア(レンジの個人差も除く場合)
- **時間正規化輪郭** 各句の半音値を句内相対時間 0–1 の等間隔 30 点にリサンプル

注意: `--adaptive-range` の有無や対象ファイルの集合が変わると基準値も変わります。
**比較する一連のデータは同じ設定で一括生成**してください。

---

## 7. 簡易版 X-JToBI ワークフロー

「自動下書き → Praat で手修正 → 修正を反映した計測」の一巡:

**(1) 下書き生成** — `--xjtobi` 付きで batch/analyze を実行 →
`<名前>_xjtobi.TextGrid` が生成される。層構成:

| 層 | 内容 | 手修正 |
|---|---|---|
| `segments` | 音素区間 | 通常不要 |
| `tones` | %L / H- / H*+L(予測の近似位置)と句末 L%(+BPM 自動判定を連結。例 `L%LH%`) | BPM の追加・修正 |
| `words` | カナ単語+**アクセント核記号 `'`**(例 `ヤマナシダ'イガク`) | **核の移動・削除・追加はここ** |
| `words_pred` | words の予測の凍結コピー+語の辞書型(例 `…/5`) | **編集禁止**(対照の基準) |
| `BI` | 1=語 / 2=アクセント句 / 3=イントネーション句境界 | 句切りの修正 |

**(2) Praat で手修正** — 音声と一緒に開き、実際の発話に合わせて words 層の核記号・
BI・tones の BPM を直す。境界を多少ドラッグしてもポイントの対応付けは自動で追従
(±150ms)。

**(3) 修正後の計測**

```
pitchan xjtobi-measure --wav X.wav --textgrid X_xjtobi.TextGrid --out results\measures --bom
```

出力:

- `X_xjtobi_measures.csv` — 句単位: 修正後の核位置(`nucleus_mora`)・BI・BPM・
  F0 統計・BPM 区間を除いた核ピーク
- `X_xjtobi_words.csv` — 語単位: **実現型**(修正後の `'` 位置)/ **予測型** /
  **辞書型**(語単独のアクセント)と `accent_match`
  (match=一致 / shifted=核位置ずれ / deleted=核脱落 / inserted=核過剰)
- `X_accent.TextGrid` — 実現アクセント型の層(Praat 確認用)

「辞書型 → 文脈予測型 → 実現型」の 3 点比較により、単語アクセント知識の問題か
句形成の問題かを切り分けられます。

※ batch の結果と同じ正規化で比較したい場合は、該当 `.json` の `ref_hz` を
`--ref value:<Hz>` で指定。

---

## 8. 境界の精密化(refine)

生成済み TextGrid の単語境界を、局所的な再アラインメント(MFA fine-tune)で
整列し直します。**入力 TextGrid は上書きされません**。

```
pitchan refine --wav X.wav --text X.txt --textgrid results\X.TextGrid --out results\refine
```

- 単語列を 5 語ずつの block に分け、境界の移動量で採否を判定:
  - `AUTO_ACCEPT`(移動 ≤80ms)= 自動採用
  - `REVIEW`(80〜250ms)= 既定では**元の境界を維持**し候補を CSV に保存。
    `--apply-review` で適用
  - `KEEP_ORIGINAL` = 失敗・矛盾・250ms 超 → 元のまま
- 出力: `X_refined.TextGrid`(修正版+元の層+`alignment_review` 層)、
  `X_alignment_diff.csv`(1 語 1 行の前後比較)、`X_refine_summary.json`
- **閾値(80/250ms 等)は暫定値**です。手修正済みの区間と突き合わせて較正してから
  本採用してください。fine-tune の 1ms 刻み出力は 1ms の正確さを保証しません

---

## 9. コマンド一覧

| コマンド | 用途 |
|---|---|
| `pitchan batch --dir D --out O [オプション]` | フォルダ内の wav/txt ペアを一括分析 |
| `pitchan analyze --wav W --text T --out O [オプション]` | 単一ペアの分析 |
| `pitchan xjtobi-measure --wav W --textgrid G --out O` | 手修正済み X-JToBI からの再計測 |
| `pitchan refine --wav W --text T --textgrid G --out O` | 境界の局所精密化 |

各コマンドの全オプションは `pitchan <コマンド> --help` で表示されます。
主要オプションの一覧と既定値は README.md の表を参照してください。

---

## 10. トラブルシューティング

**「対応する .txt がないためスキップ」** — wav と txt の名前(拡張子以外)が
不一致。同名にする。

**`NoAlignmentsError`(アラインメント失敗)** — テキストと発話の不一致が原因の
ことが多い。対処の順番: (1) `--split-sentences` を付ける(失敗が文単位に閉じる)、
(2) 言い淀み箇所のテキストを発話どおりに逐語化する、(3) それでもだめなら
`--beam 1000 --retry-beam 4000`。

**一部のファイルだけ TextGrid が生成されない** — そのファイルにテキストとの
不一致がある。ログの `N/M 発話が失敗` と該当句の `low_confidence` を確認。

**文字コードのエラー** — UTF-8 / Shift_JIS は自動判別される。それ以外
(UTF-16 など)は UTF-8 で保存し直す。

**インストール時の DLL エラー(gdk-pixbuf 等)** — conda の標準チャンネルと
conda-forge が混在している。環境を削除して §2 手順 2 のとおり
`-c conda-forge --override-channels` で一括作成し直す。

**conda の Terms of Service エラー** — §2 のとおり conda-forge のみを使えば
発生しない(表示されたコマンドで同意しても可)。

**`pyopenjtalk` のビルドエラー** — 依存は `pyopenjtalk-plus`(ビルド済み配布)に
移行済み。`git pull` して `pip install -e ".[plot]"` をやり直す。

**句ごとの図が平坦に見える** — 縦軸は既定で句ごとの自動スケール。ファイル内共通
スケール(`--plot-ap-shared-ylim`)にしていないか確認。データ自体が平坦かどうかは
`f0_range_st` と Praat のピッチ曲線で確認する。

**「ONNX Runtime is not installed」の警告** — 無害。無視してよい。

---

## 11. 分析のヒント

- **句頭上昇を見る場合**: 頭高型(`accent_type == 1`)は句頭上昇を持たないので除外。
  `follows_pause` で層別。第 1〜2 モーラが無声化した句は上昇の起点が測れない点に注意
- **話者間比較**: 同一テキストなら `ap_index` / `ap_kana` で句を対応付けて横並びに
  できる。輪郭(`ap_contours`)は句の長さ差を除いた形状比較に使う
- **除外基準**: `low_confidence == 1` の句、言い淀みを逐語化した句は集計前に除外を検討
- **BPM(`bpm_auto`)と核ピーク(`peak_excl_bpm_st`)**: まず数ファイルで聴覚判断との
  一致を確認してから使う。核ピークは有核句(`accent_type >= 1`)に絞る

## 12. 既知の制約

- 読み・アクセント型・句区切りの予測は OpenJTalk 依存(誤りは Praat 手修正で補正)
- 学習者音声では MFA(母語話者モデル)の境界精度が落ちる。目視確認を必ず行う
- BPM 自動判定・refine の閾値は暫定値(実データでの較正が前提)
- フル版 X-JToBI の細分記号(BI 2+p、PNLP 等)・CSJ 式分節音ラベルは対象外
- 歌唱・自発対話・複数話者混在の音声は対象外
