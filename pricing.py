"""
Pricing module — historical AED/sqft lookup by project name and bedroom type.

Data source: DLD (Dubai Land Department) transactions 2026, project level.
CSV columns used:
  PROJECT_EN                                           → lookup key
  Median price per sq-meter for 1BR ... Mar-May 2026  → primary 1BR reference
  Median price per sq-meter for 2BR ... Mar-May 2026  → primary 2BR reference
  Median price per sq-meter for 1BR ... Jan-Feb 2026  → fallback 1BR reference
  Median price per sq-meter for 2BR ... Jan-Feb 2026  → fallback 2BR reference

All CSV prices are in AED/sqm. This module converts to AED/sqft on output.
1 sqm = 10.7639 sqft
"""

import csv
import os

SQM_TO_SQFT = 10.7639

CSV_PATH = os.environ.get(
    "PRICING_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "dld_transactions_2026_project_level.csv"),
)

# Internal store: project_key → {1br_recent, 2br_recent, 1br_old, 2br_old} (all AED/sqm)
_DATA: dict[str, dict] = {}


def _to_float(val) -> float | None:
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _normalise(name: str) -> str:
    return name.strip().lower()


def load_pricing_data(path: str = CSV_PATH) -> int:
    """
    Load the DLD CSV. Returns number of projects loaded.
    Called once at startup in app.py.
    """
    _DATA.clear()
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                project = row.get("PROJECT_EN", "").strip()
                if not project:
                    continue
                key = _normalise(project)
                _DATA[key] = {
                    "project":    project,
                    "area":       row.get("PREFERRED_AREA_MAPPING", "").strip(),
                    "1br_recent": _to_float(row.get(
                        "Median price per sq-meter for 1BR apartment from 1 Mar 2026 - 28 May 2026")),
                    "2br_recent": _to_float(row.get(
                        "Median price per sq-meter for 2BR apartment from 1 Mar 2026 - 28 May 2026")),
                    "1br_old":    _to_float(row.get(
                        "Median price per sq-meter for 1BR apartment from 1 Jan 2026 - 28 Feb 2026")),
                    "2br_old":    _to_float(row.get(
                        "Median price per sq-meter for 2BR apartment from 1 Jan 2026 - 28 Feb 2026")),
                    "building_age": _to_float(row.get("Building age")),
                    "developer":  row.get("Builder / Developer", "").strip(),
                }
                count += 1
        print(f"[Pricing] Loaded {count} projects from {path}")
        return count
    except FileNotFoundError:
        print(f"[Pricing] CSV not found at {path}")
        return 0
    except Exception as e:
        print(f"[Pricing] Error loading CSV: {e}")
        return 0


def _find_project(building_name: str) -> dict | None:
    """Return the best matching project row, or None."""
    if not building_name or not _DATA:
        return None
    key = _normalise(building_name)
    # 1. Exact match
    if key in _DATA:
        return _DATA[key]
    # 2. Starts-with match
    for k, v in _DATA.items():
        if k.startswith(key) or key.startswith(k):
            return v
    # 3. Substring match
    for k, v in _DATA.items():
        if key in k or k in key:
            return v
    return None


def get_reference_psf(
    building_name: str | None,
    br_type: str | None,       # "1br" | "2br" | None
) -> float | None:
    """
    Return historical median price in AED/sqft for the given project and BR type.
    Uses Mar-May 2026 data first; falls back to Jan-Feb 2026 if unavailable.
    Returns None if project not found or no data for that BR type.
    """
    row = _find_project(building_name)
    if not row:
        return None

    br = (br_type or "").lower().replace(" ", "")
    if br in ("2br", "2bed", "2bedroom", "2bhk"):
        psm = row["2br_recent"] or row["2br_old"]
    else:
        # Default to 1BR if unknown
        psm = row["1br_recent"] or row["1br_old"]

    if psm is None:
        return None
    return round(psm / SQM_TO_SQFT, 2)


def get_target_psf(
    building_name: str | None,
    br_type: str | None,
    asking_psf: float | None,
    discount: float = 0.25,
) -> float | None:
    """
    Target = min(historical_psf, asking_psf) * (1 - discount).
    Returns None if no price data is available at all.
    """
    historical = get_reference_psf(building_name, br_type)
    candidates = [p for p in [historical, asking_psf] if p and p > 0]
    if not candidates:
        return None
    return round(min(candidates) * (1 - discount), 2)


def pricing_context(
    building_name: str | None,
    br_type: str | None,
    asking_psf: float | None,
    area_sqft: float | None,
) -> str:
    """
    Short pricing summary injected into the AI system prompt each turn.
    All prices in AED/sqft.
    """
    row = _find_project(building_name) if building_name else None
    historical_psf = get_reference_psf(building_name, br_type) if building_name else None
    target_psf = get_target_psf(building_name, br_type, asking_psf)

    if not row and not asking_psf:
        return (
            "No price data available yet.\n"
            "Priority: extract the building/project name, bedroom type (1BR or 2BR), "
            "asking price, and area (sqft) from the conversation as early as possible."
        )

    lines = []
    if row:
        lines.append(f"Building: {row['project']} | Area: {row['area']} | Age: {row.get('building_age') or 'N/A'} yrs | Developer: {row['developer']}")
    if historical_psf:
        br_label = (br_type or "1BR").upper()
        lines.append(f"DLD historical median ({br_label}, recent): AED {historical_psf:,.0f}/sqft")
    if asking_psf:
        lines.append(f"Seller asking price: AED {asking_psf:,.0f}/sqft")
    if target_psf:
        lines.append(f"YOUR TARGET (≥25% off reference): AED {target_psf:,.0f}/sqft or lower")
        if area_sqft:
            total = target_psf * area_sqft
            lines.append(f"Target total ({area_sqft:,.0f} sqft): AED {total:,.0f} (~AED {total/1e6:.2f}M)")

    return "\n".join(lines)
