import os
from dotenv import load_dotenv

load_dotenv()

# WhatsApp Cloud API
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "myverifytoken123")

# Anthropic API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Owner's personal WhatsApp number for escalation alerts (country code + number, no +)
# e.g. "919876543210" for India
OWNER_PHONE_NUMBER = os.environ.get("OWNER_PHONE_NUMBER")

# SQLite DB path — /tmp persists within a Render session
DB_PATH = os.environ.get("DB_PATH", "/tmp/re_agent.db")
