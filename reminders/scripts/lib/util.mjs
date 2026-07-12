// 日付まわりの共通ユーティリティ（日本時間基準）

const JST_OFFSET_MS = 9 * 60 * 60 * 1000;
const WEEKDAYS_JA = ["日", "月", "火", "水", "木", "金", "土"];

// 日本時間での「今日」を YYYY-MM-DD で返す
export function todayJST() {
  return new Date(Date.now() + JST_OFFSET_MS).toISOString().slice(0, 10);
}

// YYYY-MM-DD に日数を足す
export function addDays(dateStr, days) {
  const d = new Date(`${dateStr}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

// "2026-07-14" -> "7/14(火)"
export function formatDateJa(dateStr) {
  const d = new Date(`${dateStr}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return dateStr;
  return `${d.getUTCMonth() + 1}/${d.getUTCDate()}(${WEEKDAYS_JA[d.getUTCDay()]})`;
}

export const CATEGORY_EMOJI = {
  提出物: "📝",
  持ち物: "🎒",
  イベント: "🎈",
  その他: "📌",
};

// 予定1件を通知用の1行にする
export function formatEventLine(ev, { withDate = false } = {}) {
  const emoji = CATEGORY_EMOJI[ev.category] ?? "📌";
  const parts = [];
  if (withDate) parts.push(formatDateJa(ev.date));
  parts.push(`${emoji}${ev.title}`);
  if (ev.time) parts.push(`(${ev.time})`);
  if (ev.notes) parts.push(`- ${ev.notes}`);
  return `・${parts.join(" ")}`;
}
