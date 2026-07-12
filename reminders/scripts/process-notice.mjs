// おたより処理スクリプト
// Issue に添付されたおたよりの写真を AI で読み取り、提出物・イベントを
// reminders/events.json に登録して、結果を Issue にコメントする。
//
// 必要な環境変数:
//   GITHUB_TOKEN, REPO (owner/repo), ISSUE_NUMBER
// 任意:
//   ANTHROPIC_API_KEY (あれば Claude で読み取り。なければ GitHub Models にフォールバック)
//   LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID / DISCORD_WEBHOOK_URL / SLACK_WEBHOOK_URL

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { sendNotifications } from "./lib/notify.mjs";
import { todayJST, formatDateJa, CATEGORY_EMOJI } from "./lib/util.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const EVENTS_FILE = path.join(here, "..", "events.json");

const { GITHUB_TOKEN, REPO, ISSUE_NUMBER } = process.env;
if (!GITHUB_TOKEN || !REPO || !ISSUE_NUMBER) {
  console.error("GITHUB_TOKEN / REPO / ISSUE_NUMBER が設定されていません");
  process.exit(1);
}

const CATEGORIES = ["提出物", "持ち物", "イベント", "その他"];

// ---------- GitHub API ----------

async function github(method, apiPath, body) {
  const res = await fetch(`https://api.github.com${apiPath}`, {
    method,
    headers: {
      Authorization: `Bearer ${GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`GitHub API ${method} ${apiPath} -> ${res.status}: ${text.slice(0, 300)}`);
  }
  return res.json();
}

async function comment(text) {
  await github("POST", `/repos/${REPO}/issues/${ISSUE_NUMBER}/comments`, { body: text });
}

// ---------- 画像の取得 ----------

function extractImageUrls(body) {
  const urls = new Set();
  for (const m of body.matchAll(/!\[[^\]]*\]\((https?:\/\/[^\s)]+)\)/g)) urls.add(m[1]);
  for (const m of body.matchAll(/<img[^>]+src="(https?:\/\/[^"]+)"/g)) urls.add(m[1]);
  for (const m of body.matchAll(
    /(?<![("])(https:\/\/github\.com\/user-attachments\/assets\/[\w-]+|https:\/\/(?:private-)?user-images\.githubusercontent\.com\/[^\s)>"]+)/g,
  )) {
    urls.add(m[1]);
  }
  return [...urls];
}

async function downloadImage(url) {
  // 公開リポジトリの添付は認証なしで取得できる。トークンを付けても
  // undici はクロスオリジンのリダイレクト時に Authorization を外すので安全。
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${GITHUB_TOKEN}` },
    redirect: "follow",
  });
  if (!res.ok) throw new Error(`画像の取得に失敗 (${res.status}): ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());

  // 大きな写真でも API 制限に収まるよう縮小して JPEG に統一
  const jpeg = await sharp(buf)
    .rotate() // EXIF の向きを反映
    .resize(2000, 2000, { fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();
  return jpeg.toString("base64");
}

// ---------- AI での読み取り ----------

const EXTRACTION_SCHEMA = {
  type: "object",
  properties: {
    summary: { type: "string", description: "おたより全体の内容の要約(1〜2文)" },
    events: {
      type: "array",
      items: {
        type: "object",
        properties: {
          date: { type: "string", description: "YYYY-MM-DD 形式の日付" },
          title: { type: "string", description: "15文字以内の簡潔な名称" },
          category: { type: "string", enum: CATEGORIES },
          time: { anyOf: [{ type: "string" }, { type: "null" }], description: "時刻 (例: 8:30〜) 不明なら null" },
          notes: { anyOf: [{ type: "string" }, { type: "null" }], description: "補足 (任意) なければ null" },
        },
        required: ["date", "title", "category", "time", "notes"],
        additionalProperties: false,
      },
    },
  },
  required: ["summary", "events"],
  additionalProperties: false,
};

function buildPrompt(memo) {
  return [
    "あなたは学校から配布されたおたより(プリント)を読み取るアシスタントです。",
    `今日は ${todayJST()} (日本時間) です。`,
    "添付画像のおたよりから、保護者が対応・準備すべき「日付が特定できる」項目をすべて抽出してください。",
    "",
    "ルール:",
    "- 提出物(締切のあるもの)・持ち物・行事やイベント・その他の予定を対象とする",
    "- date は YYYY-MM-DD 形式。年の記載がない場合は今日以降で最も近い日付と解釈する(令和などの和暦は西暦に変換)",
    "- 「◯日まで」の締切は、その日を date とし、title の末尾に「〆切」を付ける",
    "- 期間のある行事(例: 7/14〜7/18)は、開始日など重要な日をそれぞれ別の項目にする",
    "- すでに過ぎた日付の項目や、日付が特定できない一般的なお願いは含めない",
    "- title は15文字以内で簡潔に。time は時刻が書かれている場合のみ",
    "- summary にはおたより全体の内容を1〜2文で日本語でまとめる",
    memo ? `\n投稿者からのメモ: ${memo}` : "",
  ].join("\n");
}

async function extractWithClaude(images, memo) {
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic();

  const content = images.map((data) => ({
    type: "image",
    source: { type: "base64", media_type: "image/jpeg", data },
  }));
  content.push({ type: "text", text: buildPrompt(memo) });

  const response = await client.messages.create({
    model: "claude-opus-4-8",
    max_tokens: 16000,
    thinking: { type: "adaptive" },
    output_config: { format: { type: "json_schema", schema: EXTRACTION_SCHEMA } },
    messages: [{ role: "user", content }],
  });

  if (response.stop_reason === "refusal") {
    throw new Error("AI が読み取りを拒否しました (refusal)");
  }
  const text = response.content.find((b) => b.type === "text")?.text;
  if (!text) throw new Error("AI から結果が返りませんでした");
  return JSON.parse(text);
}

async function extractWithGitHubModels(images, memo) {
  const content = [
    {
      type: "text",
      text:
        buildPrompt(memo) +
        "\n\n出力は次の形式の JSON のみ:\n" +
        '{"summary": "…", "events": [{"date": "YYYY-MM-DD", "title": "…", "category": "提出物|持ち物|イベント|その他", "time": "…または null", "notes": "…または null"}]}',
    },
    ...images.map((data) => ({
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${data}` },
    })),
  ];

  const res = await fetch("https://models.github.ai/inference/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GITHUB_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/gpt-4o",
      messages: [{ role: "user", content }],
      response_format: { type: "json_object" },
      max_tokens: 4000,
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`GitHub Models API エラー (${res.status}): ${text.slice(0, 300)}`);
  }
  const data = await res.json();
  const raw = data.choices?.[0]?.message?.content ?? "";
  const jsonText = raw.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
  return JSON.parse(jsonText);
}

function sanitizeEvents(events) {
  const result = [];
  for (const ev of events ?? []) {
    if (!ev || typeof ev !== "object") continue;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(ev.date ?? "")) continue;
    if (!ev.title) continue;
    result.push({
      date: ev.date,
      title: String(ev.title).slice(0, 40),
      category: CATEGORIES.includes(ev.category) ? ev.category : "その他",
      time: ev.time || null,
      notes: ev.notes || null,
    });
  }
  return result;
}

// ---------- メイン ----------

async function main() {
  const issue = await github("GET", `/repos/${REPO}/issues/${ISSUE_NUMBER}`);
  const body = issue.body ?? "";

  const imageUrls = extractImageUrls(body);
  if (imageUrls.length === 0) {
    await comment(
      "⚠️ 写真が見つかりませんでした。この Issue にコメントで写真を添付するのではなく、" +
        "新しい Issue を作成して「おたよりの写真」欄に画像を添付してください。",
    );
    return;
  }

  // 「メモ」欄のテキスト(画像リンク以外)を補足として渡す
  const memo = body
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/<img[^>]*>/g, "")
    .replace(/https:\/\/(?:github\.com\/user-attachments|(?:private-)?user-images\.githubusercontent\.com)\S+/g, "")
    .replace(/###[^\n]*/g, "")
    .replace(/_No response_/g, "")
    .trim()
    .slice(0, 500);

  console.log(`画像 ${imageUrls.length} 枚を取得中...`);
  const images = [];
  for (const url of imageUrls.slice(0, 10)) {
    images.push(await downloadImage(url));
  }

  console.log("AI でおたよりを読み取り中...");
  let result;
  if (process.env.ANTHROPIC_API_KEY) {
    result = await extractWithClaude(images, memo);
  } else {
    console.log("ANTHROPIC_API_KEY が未設定のため GitHub Models (gpt-4o) を使用します");
    result = await extractWithGitHubModels(images, memo);
  }

  const events = sanitizeEvents(result.events);
  const summary = result.summary || "おたより";

  // events.json にマージ (日付+タイトルが同じものは上書き)
  const store = JSON.parse(fs.readFileSync(EVENTS_FILE, "utf8"));
  const byKey = new Map(store.events.map((ev) => [`${ev.date}|${ev.title}`, ev]));
  for (const ev of events) {
    byKey.set(`${ev.date}|${ev.title}`, { ...ev, source: Number(ISSUE_NUMBER), addedAt: todayJST() });
  }
  store.events = [...byKey.values()].sort((a, b) => a.date.localeCompare(b.date));
  fs.writeFileSync(EVENTS_FILE, JSON.stringify(store, null, 2) + "\n");

  // Issue に結果をコメントしてクローズ
  let commentBody = `## 📋 読み取り結果\n\n${summary}\n\n`;
  if (events.length === 0) {
    commentBody +=
      "日付のある予定・提出物は見つかりませんでした。読み取りに失敗している場合は、写真を撮り直して新しい Issue を作成してください。";
  } else {
    commentBody += "| 日付 | 種類 | 内容 | 時間 | メモ |\n|---|---|---|---|---|\n";
    for (const ev of events) {
      commentBody += `| ${formatDateJa(ev.date)} | ${CATEGORY_EMOJI[ev.category]}${ev.category} | ${ev.title} | ${ev.time ?? "-"} | ${ev.notes ?? "-"} |\n`;
    }
    commentBody += `\n${events.length}件の予定を登録しました。前日の朝7時にリマインドを送ります 📱\n`;
    commentBody += "\n内容が間違っている場合は `reminders/events.json` を直接編集してください。";
  }
  await comment(commentBody);
  await github("POST", `/repos/${REPO}/issues/${ISSUE_NUMBER}/labels`, { labels: ["おたより"] }).catch(() => {});
  await github("PATCH", `/repos/${REPO}/issues/${ISSUE_NUMBER}`, { state: "closed" });

  // 登録完了をスマホにも通知
  if (events.length > 0) {
    const lines = events.map(
      (ev) => `・${formatDateJa(ev.date)} ${CATEGORY_EMOJI[ev.category]}${ev.title}${ev.time ? ` (${ev.time})` : ""}`,
    );
    const { sent, errors } = await sendNotifications(
      `📋 おたよりを登録しました\n${summary}\n\n${lines.join("\n")}\n\n前日の朝にリマインドします。`,
    );
    console.log(`通知送信: ${sent.join(", ") || "なし"}${errors.length ? ` / エラー: ${errors.join("; ")}` : ""}`);
  }

  console.log(`完了: ${events.length}件の予定を登録`);
}

main().catch(async (err) => {
  console.error(err);
  try {
    await comment(
      `❌ おたよりの処理中にエラーが発生しました:\n\n\`\`\`\n${String(err.message ?? err).slice(0, 500)}\n\`\`\`\n\n` +
        "リポジトリの Actions タブでログを確認できます。写真を撮り直して新しい Issue を作成すると再実行されます。",
    );
  } catch {}
  process.exit(1);
});
