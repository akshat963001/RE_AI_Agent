import json
from flask import Flask, request, jsonify

from config import VERIFY_TOKEN, OWNER_PHONE_NUMBER
from database import (
    init_db,
    save_message,
    get_conversation,
    get_lead,
    update_lead,
    increment_escalation_count,
    get_all_leads,
    is_blocked,
    reset_lead,
)
from negotiation_agent import get_ai_response
from whatsapp_client import send_message, send_escalation_alert
from pricing import load_pricing_data
from link_fetcher import extract_bayut_urls, fetch_bayut_listing, format_for_ai

app = Flask(__name__)

init_db()
load_pricing_data()   # no-op until Excel file is provided


# ── Webhook verification ──────────────────────────────────────────────────────

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("[Webhook] Verified ✓")
        return challenge, 200

    print("[Webhook] Verification failed — token mismatch")
    return "Forbidden", 403


# ── Incoming messages ─────────────────────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json(silent=True) or {}

    try:
        entry   = (data.get("entry") or [{}])[0]
        changes = (entry.get("changes") or [{}])[0]
        value   = changes.get("value", {})

        if "messages" not in value:
            return "OK", 200

        message    = value["messages"][0]
        lead_phone = message["from"]

        if message.get("type") != "text":
            print(f"[Webhook] Non-text message from {lead_phone}, ignoring")
            return "OK", 200

        incoming_text = message["text"]["body"].strip()
        print(f"\n[IN] {lead_phone}: {incoming_text[:100]}")

        # ── Hard blocks: blacklisted or discarded ─────────────────────────────
        if is_blocked(lead_phone):
            print(f"[Webhook] {lead_phone} is blocked (blacklisted/discarded) — ignoring")
            return "OK", 200

        # ── Human-takeover / paused ───────────────────────────────────────────
        lead = get_lead(lead_phone)
        if lead and lead.get("status") in ("human_takeover", "paused"):
            print(f"[Webhook] {lead_phone} paused — forwarding to owner, no auto-reply")
            send_message(
                OWNER_PHONE_NUMBER,
                f"📨 *New message from {lead_phone}* (AI paused):\n\n_{incoming_text}_"
            )
            return "OK", 200

        # ── Save incoming message ─────────────────────────────────────────────
        save_message(lead_phone, "user", incoming_text)

        # ── Load known pricing data for this lead ─────────────────────────────
        lead          = get_lead(lead_phone) or {}
        building_name = lead.get("building_name") or None
        br_type       = lead.get("br_type") or None
        asking_psf    = lead.get("asking_psf") or None
        area_sqft     = lead.get("area_sqft") or None

        # ── Fetch conversation history (minus the message just saved) ─────────
        history = get_conversation(lead_phone, limit=20)
        if history and history[-1]["content"] == incoming_text:
            history = history[:-1]

        # ── Bayut link detection + property data extraction ───────────────────
        ai_input_text = incoming_text
        bayut_urls = extract_bayut_urls(incoming_text)
        if bayut_urls:
            prop = fetch_bayut_listing(bayut_urls[0])
            if prop:
                print(f"[Fetcher] Extracted listing: {prop.get('title')} | AED {prop.get('price_aed')}")
                ai_input_text = (
                    f"{incoming_text}\n\n"
                    f"{format_for_ai(prop)}"
                )
                # Auto-populate pricing fields from listing if not already known
                if not asking_psf and prop.get("price_aed") and prop.get("area_sqft"):
                    asking_psf = round(prop["price_aed"] / prop["area_sqft"], 2)
                if not area_sqft and prop.get("area_sqft"):
                    area_sqft = prop["area_sqft"]
                if not building_name and prop.get("project"):
                    building_name = prop["project"]
                if not br_type and prop.get("bedrooms") is not None:
                    br_type = "2br" if int(prop["bedrooms"]) >= 2 else "1br"
            else:
                print(f"[Fetcher] Could not extract listing from {bayut_urls[0]}")
                ai_input_text = (
                    f"{incoming_text}\n\n"
                    f"[NOTE: A Bayut link was shared but could not be accessed. "
                    f"Ask the agent to share the key details manually: "
                    f"project name, area in sqft, asking price, floor, and furnishing status.]"
                )

        # ── Get AI response ───────────────────────────────────────────────────
        ai = get_ai_response(
            history, ai_input_text,
            building_name=building_name,
            br_type=br_type,
            asking_psf=asking_psf,
            area_sqft=area_sqft,
        )

        action               = ai.get("action", "continue")
        reply_text           = ai.get("reply", "")
        notification_reason  = ai.get("notification_reason") or ""
        confidence           = float(ai.get("confidence", 0.7))
        lead_score           = int(ai.get("lead_score", 5))
        notes                = ai.get("notes", "")

        # Update extracted pricing data if AI found new values
        pricing_updates = {}
        if ai.get("building_name"):
            pricing_updates["building_name"] = ai["building_name"]
        if ai.get("br_type"):
            pricing_updates["br_type"] = ai["br_type"]
        if ai.get("asking_psf"):
            pricing_updates["asking_psf"] = float(ai["asking_psf"])
        if ai.get("area_sqft"):
            pricing_updates["area_sqft"] = float(ai["area_sqft"])

        # ── Execute action ────────────────────────────────────────────────────

        if action == "discard":
            # Silent — no reply, mark discarded
            update_lead(lead_phone, status="discarded", lead_score=lead_score,
                        notes=notes, **pricing_updates)
            print(f"[DISCARD] {lead_phone} — lead silently discarded")

        elif action == "blacklist":
            # Send one respectful closing message, then blacklist
            if reply_text:
                send_message(lead_phone, reply_text)
                save_message(lead_phone, "assistant", reply_text)
            update_lead(lead_phone, status="blacklisted", lead_score=lead_score,
                        notes=notes, **pricing_updates)
            print(f"[BLACKLIST] {lead_phone} — blacklisted after abusive behaviour")

        elif action == "reject":
            # Send polite rejection, mark closed
            if reply_text:
                send_message(lead_phone, reply_text)
                save_message(lead_phone, "assistant", reply_text)
            update_lead(lead_phone, status="closed", lead_score=lead_score,
                        notes=notes, **pricing_updates)
            print(f"[REJECT] {lead_phone} — deal rejected (legal/compliance issue)")

        elif action == "pause":
            # Send reply if any, then pause AI and notify owner
            if reply_text:
                send_message(lead_phone, reply_text)
                save_message(lead_phone, "assistant", reply_text)
            update_lead(lead_phone, status="paused", lead_score=lead_score,
                        notes=notes, **pricing_updates)
            increment_escalation_count(lead_phone)
            _notify_owner(lead_phone, incoming_text, notification_reason,
                          lead_score, confidence, paused=True)
            print(f"[PAUSE] {lead_phone} — AI paused, owner notified")

        elif action == "notify":
            # Send reply and notify owner — AI stays active
            if reply_text:
                send_message(lead_phone, reply_text)
                save_message(lead_phone, "assistant", reply_text)
            update_lead(lead_phone, status="active", lead_score=lead_score,
                        notes=notes, **pricing_updates)
            increment_escalation_count(lead_phone)
            _notify_owner(lead_phone, incoming_text, notification_reason,
                          lead_score, confidence, paused=False)
            print(f"[NOTIFY] {lead_phone} — owner notified, AI continues")

        else:
            # action == "continue" (or anything unrecognised)
            if reply_text:
                send_message(lead_phone, reply_text)
                save_message(lead_phone, "assistant", reply_text)
            update_lead(lead_phone, status="active", lead_score=lead_score,
                        notes=notes, **pricing_updates)

        print(
            f"[DONE] {lead_phone} | action={action} | "
            f"score={lead_score}/10 | conf={confidence:.0%}"
        )

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()

    return "OK", 200


def _notify_owner(
    lead_phone: str,
    incoming_text: str,
    reason: str,
    lead_score: int,
    confidence: float,
    paused: bool,
):
    recent = get_conversation(lead_phone, limit=8)
    snippet_lines = []
    for msg in recent:
        label = "Them" if msg["role"] == "user" else "AI"
        snippet_lines.append(f"*{label}:* {msg['content'][:120]}")
    snippet = "\n".join(snippet_lines)

    send_escalation_alert(
        owner_phone=OWNER_PHONE_NUMBER,
        lead_phone=lead_phone,
        lead_message=incoming_text,
        escalation_reason=reason + (" *(AI paused — reply /release/{} to resume)*".format(lead_phone) if paused else ""),
        conversation_snippet=snippet,
        lead_score=lead_score,
    )


# ── Management endpoints ──────────────────────────────────────────────────────

@app.route("/leads", methods=["GET"])
def view_leads():
    return jsonify(get_all_leads()), 200


@app.route("/leads/<phone>", methods=["GET"])
def view_lead(phone):
    return jsonify({
        "lead": get_lead(phone),
        "conversation": get_conversation(phone, limit=100),
    }), 200


@app.route("/release/<phone>", methods=["POST"])
def release_lead(phone):
    """Resume AI negotiation for a paused lead."""
    update_lead(phone, status="active")
    print(f"[Release] AI resumed for {phone}")
    return jsonify({"status": "ok", "phone": phone, "mode": "active"}), 200


@app.route("/takeover/<phone>", methods=["POST"])
def takeover_lead(phone):
    """Pause AI and take over manually."""
    update_lead(phone, status="human_takeover")
    print(f"[Takeover] Human takeover for {phone}")
    return jsonify({"status": "ok", "phone": phone, "mode": "human_takeover"}), 200


@app.route("/close/<phone>", methods=["POST"])
def close_lead(phone):
    update_lead(phone, status="closed")
    return jsonify({"status": "ok", "phone": phone, "mode": "closed"}), 200


@app.route("/reset/<phone>", methods=["POST"])
def reset_lead_endpoint(phone):
    """Wipe all history and lead data for a number — complete fresh start."""
    reset_lead(phone)
    print(f"[Reset] Fresh start for {phone}")
    return jsonify({"status": "ok", "phone": phone, "mode": "reset"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)
