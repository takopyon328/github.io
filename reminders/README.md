# 🏫 学校おたよりリマインダー

学校から配布される紙のおたよりをスマホで撮影してアップロードすると、AI が提出物・持ち物・イベントを読み取り、**前日の朝7時にスマホへリマインド**してくれるシステムです。サーバー不要で、GitHub の機能（Issues + Actions）だけで動きます。

## 毎日の使い方

1. おたよりをスマホで撮影する
2. このリポジトリの **Issues → New issue → 「📄 おたよりアップロード」** を開く
   （GitHub アプリまたはブラウザ。よく使うのでホーム画面にショートカットを置くと便利です）
3. 「おたよりの写真」欄に写真を添付して **Submit**
4. 数分後、読み取り結果が Issue にコメントされ、登録完了の通知がスマホに届きます
5. 予定の前日 朝7:00 に「明日の予定・提出物」がスマホに通知されます

読み取り結果が間違っていた場合は `reminders/events.json` を直接編集すれば修正できます。

## 初期設定（最初の1回だけ）

### 1. main ブランチに取り込む

このフォルダと `.github/` の変更を main ブランチにマージします（Issue 起動・毎朝の定期実行は main ブランチのワークフローだけが動くため）。

### 2. AI の読み取り設定（どちらか）

| 方法 | 設定 | 備考 |
|---|---|---|
| **Claude API（推奨）** | リポジトリの Settings → Secrets and variables → Actions → New repository secret で `ANTHROPIC_API_KEY` を登録 | 読み取り精度が高い。[console.anthropic.com](https://console.anthropic.com/) で API キーを取得（従量課金・1枚あたり数円程度） |
| GitHub Models（無料） | 設定不要（`ANTHROPIC_API_KEY` が無い場合に自動で使用） | 無料だがレート制限があり、精度もやや落ちます |

### 3. 通知先の設定（いずれか1つ以上）

Secrets は同じく Settings → Secrets and variables → Actions で登録します。

#### 📱 LINE で受け取る（希望されていた方法）

LINE 公式アカウント（Messaging API）の「プッシュ通知」を使います。自分だけの通知用ボットを作るイメージです。

1. [LINE Developers](https://developers.line.biz/ja/) にLINEアカウントでログイン
2. プロバイダーを新規作成（名前は何でもOK。例:「家庭用」）
3. **Messaging API チャネル**を作成（名前例:「学校リマインダー」）
   ※ 2024年以降は LINE Official Account Manager 経由での作成に誘導される場合があります。その場合は公式アカウントを作成後、[LINE Developers コンソール](https://developers.line.biz/console/)で Messaging API を有効化してください
4. チャネルの **Messaging API 設定**タブ → 一番下の「チャネルアクセストークン（長期）」を**発行**してコピー
   → Secret `LINE_CHANNEL_ACCESS_TOKEN` に登録
5. **チャネル基本設定**タブ → 一番下の「あなたのユーザーID」（`U` で始まる文字列）をコピー
   → Secret `LINE_USER_ID` に登録
6. Messaging API 設定タブに表示される QR コードから、**ボットを友だち追加**する

これで、そのボットからのメッセージとして通知が届きます。無料枠（月200通）で十分足ります。

#### 💬 Discord で受け取る（設定が最も簡単・5分）

1. Discord アプリで自分用サーバーを作成（無料）
2. 通知用チャンネルの設定 → 連携サービス → **ウェブフック**を作成 → URL をコピー
3. Secret `DISCORD_WEBHOOK_URL` に登録
4. スマホの Discord アプリでそのチャンネルの通知をオンにする

#### 💼 Slack で受け取る

1. Slack で [Incoming Webhook](https://api.slack.com/messaging/webhooks) を作成
2. Secret `SLACK_WEBHOOK_URL` に登録

複数設定した場合はすべてに送信されます。

### 4. 動作テスト

1. Actions タブ → 「前日リマインド送信」 → **Run workflow** で手動実行 → 通知設定の確認
   （予定が無い日は「スキップ」とログに出るだけで通知は来ません。テストするなら `events.json` に明日の日付の予定を1件書いてから実行してください）
2. 適当なおたより（または予定が書かれた紙）を撮影して Issue を作成 → 読み取り結果のコメントと通知を確認

## 通知時刻の変更

`.github/workflows/daily-reminder.yml` の cron を編集します。UTC 表記なので **日本時間 − 9時間** です。

```yaml
- cron: '0 22 * * *'   # 22:00 UTC = 朝 7:00 JST
- cron: '0 11 * * *'   # 11:00 UTC = 夜 8:00 JST (前日の夜に通知したい場合)
```

※ GitHub Actions の cron は数分〜数十分遅れることがあります。

## ⚠️ プライバシーに関する重要な注意

**このリポジトリ（GitHub Pages 用）は公開リポジトリです。** Issue に添付した写真は**誰でも閲覧できます**。おたよりには学校名・子どもの名前・行事の日時など個人情報が含まれることがあります。

**非公開（プライベート）リポジトリでの運用を強くおすすめします。** このシステムは自己完結しているので、簡単に引っ越せます:

1. GitHub で新しい **Private リポジトリ**を作成（例: `school-reminders`）
2. このリポジトリから以下をコピー:
   - `reminders/` フォルダ一式
   - `.github/workflows/process-notice.yml`
   - `.github/workflows/daily-reminder.yml`
   - `.github/ISSUE_TEMPLATE/otayori.yml`
3. 新リポジトリの Settings → Secrets に上記のシークレットを登録

プライベートリポジトリでも Actions の無料枠（月2,000分）内で十分動作します（1回の処理は1〜2分程度）。

## 仕組み

```
📱 写真を Issue に添付
   └→ GitHub Actions (process-notice.yml)
        ├→ AI (Claude / GitHub Models) が画像から予定を抽出
        ├→ reminders/events.json に保存
        ├→ Issue に読み取り結果をコメントしてクローズ
        └→ 登録完了をスマホに通知

⏰ 毎朝 7:00 JST
   └→ GitHub Actions (daily-reminder.yml)
        ├→ events.json から「明日」と「今日」の予定を取得
        ├→ LINE / Discord / Slack に通知
        └→ 終わった予定を掃除
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| `events.json` | 登録された予定データ（手動編集OK） |
| `scripts/process-notice.mjs` | おたより読み取り・登録 |
| `scripts/send-reminders.mjs` | 前日リマインド送信 |
| `scripts/lib/notify.mjs` | LINE / Discord / Slack への送信 |
| `scripts/lib/util.mjs` | 日付処理など |
