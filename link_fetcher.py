"""
Link fetcher — extracts property details from Bayut listing URLs.

Strategy (in order):
  1. Parse __NEXT_DATA__ embedded JSON (most complete, most reliable)
  2. Parse JSON-LD structured data
  3. Parse Open Graph / meta tags (minimal fallback)
  4. Return None so caller can ask agent to share details manually
"""

import re
import json
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

BAYUT_PATTERN = re.compile(
    r"https?://(?:www\.)?bayut\.com/(?:property/details-\d+\.html|[^\s]+)",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)


# ── URL detection ─────────────────────────────────────────────────────────────

def extract_urls(text: str) -> list[str]:
    """Return all URLs found in a message."""
    return URL_PATTERN.findall(text)


def is_bayut_url(url: str) -> bool:
    return "bayut.com" in url.lower()


def extract_bayut_urls(text: str) -> list[str]:
    return [u for u in extract_urls(text) if is_bayut_url(u)]


# ── Fetching ──────────────────────────────────────────────────────────────────

def _fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if r.status_code == 200:
            return r.text
        print(f"[Fetcher] HTTP {r.status_code} for {url}")
        return None
    except Exception as e:
        print(f"[Fetcher] Error fetching {url}: {e}")
        return None


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_next_data(soup: BeautifulSoup) -> dict | None:
    """Extract property details from Bayut's embedded __NEXT_DATA__ JSON."""
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        data = json.loads(tag.string)
        # Bayut path: props → pageProps → propertyDetails
        prop = (
            data.get("props", {})
                .get("pageProps", {})
                .get("propertyDetails") or
            data.get("props", {})
                .get("pageProps", {})
                .get("listing")
        )
        if not prop:
            return None
        return _extract_from_prop(prop)
    except Exception as e:
        print(f"[Fetcher] __NEXT_DATA__ parse error: {e}")
        return None


def _extract_from_prop(prop: dict) -> dict:
    """Normalise a Bayut propertyDetails object into our standard dict."""

    # Location hierarchy: list of dicts with 'name' and 'level'
    location_parts = []
    for loc in prop.get("location", []):
        name = loc.get("name") or loc.get("slug", "")
        if name:
            location_parts.append(name)
    location_str = " → ".join(location_parts) if location_parts else None

    # Building / project name is often the last or second-last location level,
    # or available as a separate field
    project = (
        prop.get("project", {}) or {}
    ).get("name") or (
        prop.get("building", {}) or {}
    ).get("name") or (
        location_parts[-1] if len(location_parts) >= 1 else None
    )

    # Amenities
    amenities = []
    for a in prop.get("amenities", []):
        if isinstance(a, dict):
            amenities.append(a.get("text") or a.get("name") or "")
        elif isinstance(a, str):
            amenities.append(a)
    amenities = [a for a in amenities if a]

    # Price
    price = prop.get("price") or prop.get("priceMin")

    # Area — Bayut typically provides in sqft but check unit
    area = prop.get("area") or prop.get("areaMin")

    return {
        "title":        prop.get("title") or prop.get("name"),
        "price_aed":    price,
        "area_sqft":    area,
        "bedrooms":     prop.get("rooms") or prop.get("bedrooms"),
        "bathrooms":    prop.get("baths") or prop.get("bathrooms"),
        "type":         prop.get("type", {}).get("name") if isinstance(prop.get("type"), dict) else prop.get("type"),
        "purpose":      prop.get("purpose"),          # 'for-sale' or 'for-rent'
        "furnishing":   prop.get("furnishingStatus"),
        "floor":        prop.get("floor") or prop.get("floorNumber"),
        "project":      project,
        "location":     location_str,
        "description":  (prop.get("description") or "")[:600],
        "amenities":    amenities[:15],
        "agent_name":   (prop.get("agency") or {}).get("name") or (prop.get("agent") or {}).get("name"),
        "listing_url":  prop.get("shareURL") or prop.get("externalID"),
    }


def _parse_jsonld(soup: BeautifulSoup) -> dict | None:
    """Fallback: extract from JSON-LD structured data."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            if not isinstance(data, dict):
                continue
            offers = data.get("offers", {})
            return {
                "title":       data.get("name"),
                "price_aed":   offers.get("price"),
                "area_sqft":   None,
                "bedrooms":    data.get("numberOfRooms"),
                "bathrooms":   None,
                "type":        data.get("@type"),
                "purpose":     None,
                "furnishing":  None,
                "floor":       None,
                "project":     None,
                "location":    data.get("address", {}).get("addressLocality"),
                "description": (data.get("description") or "")[:600],
                "amenities":   [],
                "agent_name":  None,
                "listing_url": data.get("url"),
            }
        except Exception:
            continue
    return None


def _parse_meta(soup: BeautifulSoup) -> dict | None:
    """Last resort: Open Graph / meta tags."""
    def meta(prop):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        return tag["content"] if tag and tag.get("content") else None

    title = meta("og:title") or (soup.title.string if soup.title else None)
    desc  = meta("og:description") or meta("description")

    if not title and not desc:
        return None

    return {
        "title":       title,
        "price_aed":   None,
        "area_sqft":   None,
        "bedrooms":    None,
        "bathrooms":   None,
        "type":        None,
        "purpose":     None,
        "furnishing":  None,
        "floor":       None,
        "project":     None,
        "location":    None,
        "description": (desc or "")[:600],
        "amenities":   [],
        "agent_name":  None,
        "listing_url": meta("og:url"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_bayut_listing(url: str) -> dict | None:
    """
    Fetch a Bayut listing and return a normalised property dict.
    Returns None if the page cannot be fetched or parsed.
    """
    html = _fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    result = _parse_next_data(soup) or _parse_jsonld(soup) or _parse_meta(soup)
    if result:
        result["source_url"] = url
    return result


def format_for_ai(prop: dict) -> str:
    """
    Format extracted property data as a concise context block for the AI.
    """
    lines = ["[PROPERTY LISTING — extracted from Bayut]"]

    if prop.get("title"):
        lines.append(f"Title: {prop['title']}")
    if prop.get("type"):
        lines.append(f"Type: {prop['type']}")
    if prop.get("bedrooms") is not None:
        lines.append(f"Bedrooms: {prop['bedrooms']}")
    if prop.get("bathrooms") is not None:
        lines.append(f"Bathrooms: {prop['bathrooms']}")
    if prop.get("area_sqft"):
        lines.append(f"Area: {prop['area_sqft']:,} sqft")
    if prop.get("price_aed"):
        lines.append(f"Asking Price: AED {prop['price_aed']:,}")
        if prop.get("area_sqft") and prop["area_sqft"] > 0:
            psf = prop["price_aed"] / prop["area_sqft"]
            lines.append(f"Asking Price/sqft: AED {psf:,.0f}")
    if prop.get("project"):
        lines.append(f"Project/Building: {prop['project']}")
    if prop.get("location"):
        lines.append(f"Location: {prop['location']}")
    if prop.get("floor"):
        lines.append(f"Floor: {prop['floor']}")
    if prop.get("furnishing"):
        lines.append(f"Furnishing: {prop['furnishing']}")
    if prop.get("amenities"):
        lines.append(f"Amenities: {', '.join(prop['amenities'][:10])}")
    if prop.get("description"):
        lines.append(f"Description: {prop['description']}")
    if prop.get("source_url"):
        lines.append(f"Link: {prop['source_url']}")

    return "\n".join(lines)
