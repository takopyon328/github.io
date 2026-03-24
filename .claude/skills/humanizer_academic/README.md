# Humanizer Academic

日本語の学術論文（応用言語学・第二言語習得分野）からAI生成文の痕跡を除去するClaude Codeスキル。投稿先として「日本語教育」「音声研究」等の国内学術誌を想定。

## インストール

### 推奨（Claude Codeスキルディレクトリに直接クローン）

```bash
mkdir -p ~/.claude/skills
git clone https://github.com/matsuikentaro1/humanizer_academic.git ~/.claude/skills/humanizer_academic
```

### 手動インストール（スキルファイルのみ）

```bash
mkdir -p ~/.claude/skills/humanizer_academic
cp SKILL.md ~/.claude/skills/humanizer_academic/
```

## 使い方

Claude Codeでスキルを呼び出す：

```
/humanizer_academic

[論文テキストを貼り付ける]
```

または直接依頼する：

```
このテキストをヒューマナイズしてください：[テキスト]
```

## 概要

[Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)を基に、日本語のSLA論文向けに適応。日本語特有のAI文体パターンを14項目に整理し、Before/After例付きで提示。

## 検出する14パターン

### 内容パターン

| # | パターン | Before | After |
|---|---------|--------|-------|
| 1 | **意義の過剰な強調** | 「極めて重要な役割を果たしており」 | 具体的なデータ・引用で記述 |
| 2 | **曖昧な典拠** | 「多くの研究が示している」 | 具体的な研究者名・文献を引用 |
| 3 | **「〜することで」の浅い分析** | 「明らかにすることで発展に寄与する」 | 具体的な分析手法と目的を記述 |
| 4 | **「多角的」「包括的」** | 「多角的な視点から包括的に検討」 | 実際の対象・手法を具体的に記述 |
| 5 | **定型的な課題と展望** | 「さらなる研究の蓄積が期待される」 | 具体的な限界と検証項目 |

### 語彙・文法パターン

| # | パターン | Before | After |
|---|---------|--------|-------|
| 6 | **AI頻出語彙** | 「さらに」「加えて」「興味深いことに」 | 削除または簡潔な接続に |
| 7 | **コピュラ回避** | 「として位置づけられる」 | 「である」 |
| 8 | **受身・名詞化の過剰使用** | 「統計的分析が実施された」 | 「t検定で比較した」 |
| 9 | **三点セットの強制** | 「音声・文法・語用の三つの側面」 | 実際の対象に限定 |
| 10 | **同義語の不自然な循環** | 「学習者→習得者→対象者」 | 「学習者」で統一 |

### 文体パターン

| # | パターン | Before | After |
|---|---------|--------|-------|
| 11 | **接続表現の過剰使用** | 「一方で」「他方」の多用 | 実際の対比関係がある場合のみ使用 |
| 12 | **ヘッジの連発** | 「〜と言えるだろう」「〜と考えられる」 | データに基づく記述と解釈を区別 |
| 13 | **前置きフレーズ** | 「周知のように」「言うまでもなく」 | 削除 |
| 14 | **定型的な肯定的結論** | 「発展が期待される」「願ってやまない」 | 具体的な結論と今後の検証 |

## 参考

- [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- [WikiProject AI Cleanup](https://en.wikipedia.org/wiki/Wikipedia:WikiProject_AI_Cleanup)

想定投稿先：
- 『日本語教育』（日本語教育学会）
- 『音声研究』（日本音声学会）

## バージョン履歴

- **2.0.0** - 日本語応用言語学（SLA）向けに全面改訂
- **1.0.0** - 英語医学論文向け初版

## ライセンス

MIT

原版：[blader/humanizer](https://github.com/blader/humanizer)
