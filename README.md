# 音声アノテーションツール - TextGrid生成アプリ

テキストと朗読音声をもとに、音声ファイルに分節音（音素）レベルのアノテーションを付与したTextGridを生成するWebアプリケーションです。

## 機能

- 📝 テキスト入力と音声ファイルのアップロード
- 🎯 音素レベルの自動アライメント
- 📊 単語レベルと音素レベルの2層TextGrid生成
- 🇯🇵 日本語テキストの音素変換対応（pyopenjtalk使用）
- 🌐 使いやすいWebインターフェース
- 🎨 Praatで開ける標準TextGrid形式での出力

## セットアップ

### 必要要件

- Python 3.8以上
- pip（Pythonパッケージマネージャー）
- (オプション) 仮想環境ツール（venv, conda等）

### インストール手順

1. **リポジトリをクローン**
```bash
git clone <repository-url>
cd github.io
```

2. **仮想環境を作成（推奨）**
```bash
python3 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

3. **依存パッケージをインストール**
```bash
pip install -r requirements.txt
```

**注意**: pyopenjtalkのインストールにはC++コンパイラが必要です。

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install build-essential
  ```

- **macOS**:
  ```bash
  xcode-select --install
  ```

- **Windows**:
  Visual Studio Build Toolsをインストールしてください。

## 使い方

### アプリケーションの起動

```bash
python app.py
```

ブラウザで `http://localhost:5000` にアクセスしてください。

### 基本的な使用手順

1. **テキストを入力**
   - 朗読されたテキストをテキストエリアに入力します

2. **音声ファイルをアップロード**
   - 対応形式: WAV, MP3, FLAC, OGG, M4A
   - ファイルサイズ上限: 100MB

3. **TextGridを生成**
   - 「TextGridを生成」ボタンをクリック
   - 処理完了後、TextGridファイルが自動ダウンロードされます

4. **Praatで確認**
   - ダウンロードしたTextGridファイルをPraatで開いて確認できます

## アライメント手法

このツールは2つのアライメント手法を提供しています：

### 1. シンプルアライメント（`simple`）
- 音声の長さを音素数で均等分割
- 処理が高速
- 精度は低め

### 2. エネルギーベースアライメント（`energy`）- デフォルト
- 音声のエネルギー分布を考慮
- 音声区間を検出して音素を配置
- より自然なアライメント結果

## プロジェクト構造

```
.
├── app.py                  # Flaskアプリケーション本体
├── aligner.py             # アライメントモジュール
├── requirements.txt       # 依存パッケージリスト
├── templates/
│   └── index.html        # Webインターフェース
├── static/               # 静的ファイル（CSS, JSなど）
└── README.md             # このファイル
```

## 技術スタック

- **Backend**: Flask（Pythonウェブフレームワーク）
- **音声処理**: librosa, soundfile
- **音素変換**: pyopenjtalk（日本語形態素解析・音素変換）
- **TextGrid生成**: praatio
- **Frontend**: HTML5, CSS3, JavaScript（Vanilla）

## TextGridファイルについて

生成されるTextGridファイルには以下の2つのティア（層）が含まれます：

1. **phonemes** - 音素レベルのアノテーション
   - 各音素の開始時刻と終了時刻
   - 音素記号（pyopenjtalkの音素体系）

2. **words** - 単語レベルのアノテーション
   - 各単語の開始時刻と終了時刻
   - 単語テキスト

## 制限事項と注意点

- 現在のアライメント精度は参考程度です
- より高精度なアライメントには、Montreal Forced Aligner等の専門ツールをご検討ください
- 長時間音声（10分以上）の処理には時間がかかる場合があります
- 背景ノイズが多い音声ではアライメント精度が低下します

## トラブルシューティング

### pyopenjtalkのインストールエラー

```bash
# C++コンパイラがインストールされているか確認
gcc --version  # Linux/macOS
cl.exe         # Windows
```

### 音声ファイルが読み込めない

- ファイル形式が対応しているか確認してください
- ファイルが破損していないか確認してください
- サンプリングレートが極端に高い場合は、事前に16kHzに変換してください

### メモリエラー

- 大きな音声ファイルを処理する場合、メモリ不足になることがあります
- ファイルを分割して処理するか、システムメモリを増やしてください

## 今後の改善予定

- [ ] より高度なアライメントアルゴリズムの実装
- [ ] Wav2Vec2等のニューラルネットワークモデルの統合
- [ ] バッチ処理機能
- [ ] アライメント結果のビジュアルプレビュー
- [ ] 多言語対応（英語、中国語等）

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 参考資料

- [Praat - 音声分析ソフトウェア](https://www.fon.hum.uva.nl/praat/)
- [pyopenjtalk - 日本語音素変換](https://github.com/r9y9/pyopenjtalk)
- [praatio - TextGrid操作ライブラリ](https://github.com/timmahrt/praatIO)
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)

## お問い合わせ

質問や提案がある場合は、GitHubのissueを作成してください。
