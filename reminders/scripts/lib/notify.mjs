// スマホへの通知送信。設定されている通知先(シークレット)すべてに送る。
//   LINE   : LINE_CHANNEL_ACCESS_TOKEN + LINE_USER_ID (Messaging API の push)
//   Discord: DISCORD_WEBHOOK_URL
//   Slack  : SLACK_WEBHOOK_URL

async function post(url, headers, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`);
  }
}

export async function sendNotifications(text) {
  const targets = [];

  if (process.env.LINE_CHANNEL_ACCESS_TOKEN && process.env.LINE_USER_ID) {
    targets.push({
      name: "LINE",
      send: () =>
        post(
          "https://api.line.me/v2/bot/message/push",
          { Authorization: `Bearer ${process.env.LINE_CHANNEL_ACCESS_TOKEN}` },
          {
            to: process.env.LINE_USER_ID,
            messages: [{ type: "text", text: text.slice(0, 4900) }],
          },
        ),
    });
  }

  if (process.env.DISCORD_WEBHOOK_URL) {
    targets.push({
      name: "Discord",
      send: () =>
        post(process.env.DISCORD_WEBHOOK_URL, {}, { content: text.slice(0, 1900) }),
    });
  }

  if (process.env.SLACK_WEBHOOK_URL) {
    targets.push({
      name: "Slack",
      send: () => post(process.env.SLACK_WEBHOOK_URL, {}, { text }),
    });
  }

  const sent = [];
  const errors = [];
  for (const target of targets) {
    try {
      await target.send();
      sent.push(target.name);
    } catch (err) {
      errors.push(`${target.name}: ${err.message}`);
    }
  }
  return { configured: targets.length, sent, errors };
}
