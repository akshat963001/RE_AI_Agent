import requests
from config import WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_ACCESS_TOKEN

GRAPH_API_VERSION = "v19.0"
MESSAGES_URL = (
    f"https://graph.facebook.com/{GRAPH_API_VERSION}"
    f"/{WHATSAPP_PHONE_NUMBER_ID}/messages"
)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def send_message(to: str, text: str) -> bool:
    """
    Send a plain-text WhatsApp message.
    `to` must be a full phone number with country code, no '+' (e.g. '919876543210').
    Returns True on success.
    """
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }
    try:
        resp = requests.post(MESSAGES_URL, json=payload, headers=_headers(), timeout=10)
        resp.raise_for_status()
        print(f"[WA] Sent to {to}: {text[:60]}{'...' if len(text) > 60 else ''}")
        return True
    except requests.RequestException as e:
        print(f"[WA] FAILED sending to {to}: {e}")
        return False


def send_escalation_alert(
    owner_phone: str,
    lead_phone: str,
    lead_message: str,
    escalation_reason: str,
    conversation_snippet: str,
    lead_score: int = 0,
) -> bool:
    """
    Send a rich escalation alert to the owner's personal WhatsApp number.
    """
    score_bar = "🟩" * lead_score + "⬜" * (10 - lead_score) if lead_score else ""

    alert = (
        f"🚨 *ESCALATION ALERT* 🚨\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📱 *Lead:* {lead_phone}\n"
        f"📊 *Score:* {score_bar} ({lead_score}/10)\n"
        f"⚠️  *Reason:* {escalation_reason}\n\n"
        f"💬 *Their last message:*\n"
        f"_{lead_message[:200]}_\n\n"
        f"📜 *Recent conversation:*\n"
        f"{conversation_snippet}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"👉 Reply directly to *{lead_phone}* on WhatsApp to take over.\n"
        f"   Or call /takeover/{lead_phone} to pause the AI for this lead."
    )
    return send_message(owner_phone, alert)
