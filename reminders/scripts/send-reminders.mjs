// 前日リマインド送信スクリプト (毎朝 7:00 JST に GitHub Actions から実行)
// 「明日」の予定を中心に、「今日」の予定もあわせて通知する。
// あわせて古くなった予定を events.json から掃除する。
//
// 依存パッケージなし (Node 20+ の標準機能のみで動く)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { sendNotifications } from "./lib/notify.mjs";
import { todayJST, addDays, formatDateJa, formatEventLine } from "./lib/util.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const EVENTS_FILE = path.join(here, "..", "events.json");

async function main() {
  const store = JSON.parse(fs.readFileSync(EVENTS_FILE, "utf8"));
  const today = todayJST();
  const tomorrow = addDays(today, 1);

  const todayEvents = store.events.filter((ev) => ev.date === today);
  const tomorrowEvents = store.events.filter((ev) => ev.date === tomorrow);

  // 3日以上前の予定は削除 (コミットは workflow 側で行う)
  const cutoff = addDays(today, -3);
  const kept = store.events.filter((ev) => ev.date >= cutoff);
  if (kept.length !== store.events.length) {
    fs.writeFileSync(EVENTS_FILE, JSON.stringify({ ...store, events: kept }, null, 2) + "\n");
    console.log(`古い予定を ${store.events.length - kept.length} 件削除しました`);
  }

  if (todayEvents.length === 0 && tomorrowEvents.length === 0) {
    console.log("今日・明日の予定はありません。通知はスキップします。");
    return;
  }

  const sections = [`🏫 学校リマインダー ${formatDateJa(today)}`];
  if (tomorrowEvents.length > 0) {
    sections.push(`\n【明日 ${formatDateJa(tomorrow)}】\n${tomorrowEvents.map((ev) => formatEventLine(ev)).join("\n")}`);
  }
  if (todayEvents.length > 0) {
    sections.push(`\n【今日】\n${todayEvents.map((ev) => formatEventLine(ev)).join("\n")}`);
  }
  const message = sections.join("\n");
  console.log(message);

  const { configured, sent, errors } = await sendNotifications(message);
  if (configured === 0) {
    console.warn(
      "通知先が設定されていません。LINE_CHANNEL_ACCESS_TOKEN + LINE_USER_ID / DISCORD_WEBHOOK_URL / SLACK_WEBHOOK_URL のいずれかをリポジトリの Secrets に設定してください。",
    );
    process.exit(1);
  }
  console.log(`送信済み: ${sent.join(", ") || "なし"}`);
  if (errors.length > 0) {
    console.error(`送信エラー: ${errors.join("; ")}`);
    if (sent.length === 0) process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
