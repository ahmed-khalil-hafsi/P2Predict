"""Fetch Battery Management IC pricing data from the DigiKey ProductSearch v4 API.

DigiKey uses OAuth2 client_credentials → Bearer token → REST. This
script handles the whole flow:

  1. Loads ``client_id`` and ``client_secret`` from ``~/.digikey/credentials``
     (JSON file, chmod 600). Setup instructions below.
  2. Exchanges them for a Bearer token at
     ``https://api.digikey.com/v1/oauth2/token`` (token TTL ~10 min;
     refreshed automatically during a long pull).
  3. Calls ``POST /products/v4/search/keyword`` for "battery management",
     paging through ``limit=50`` records per request up to ``--limit``
     total parts. The DigiKey free tier allows 1,000 requests/day —
     vastly more headroom than the Nexar 100-part lifetime cap that
     pushed us off Octopart.
  4. For each part, extracts: manufacturer, MPN, category, every
     ``Parameter`` (the spec rows visible on the product page), the
     at-1 catalog price, the at-1k price-break price when available,
     stock quantity, and a couple of provenance fields.
  5. Writes a wide-form CSV (one column per Parameter we keep).

Setup
-----
1. Register at https://developer.digikey.com and create a Production
   app. Subscribe it to "Product Information V4". Note the Client ID
   and Client Secret (the secret is shown only once).
2. Save them to ``~/.digikey/credentials`` (chmod 600)::

       mkdir -p ~/.digikey && chmod 700 ~/.digikey
       cat > ~/.digikey/credentials <<'EOF'
       {"client_id": "...", "client_secret": "..."}
       EOF
       chmod 600 ~/.digikey/credentials

Usage
-----
::

    python fetch_data.py [--query "battery management"] [--limit 150]

The output CSV goes to ``data/bmics.csv`` (gitignored — DigiKey catalog
data is not redistributable per their terms; the schema and code here
*are* checked in, just not the data).

ToS note
--------
DigiKey's developer agreement allows programmatic use of their catalog
data for "legitimate business purposes." Building a parametric pricing
case study on the API's intended response shape, well under the free-tier
quota, and not republishing raw rows is the intended use. Don't check
the raw CSV into git. Don't re-host it on a public mirror.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

CREDS_PATH = Path.home() / ".digikey" / "credentials"
TOKEN_URL = "https://api.digikey.com/v1/oauth2/token"
SEARCH_URL = "https://api.digikey.com/products/v4/search/keyword"

# Parameter names we keep as feature columns. Picked from a probe of the
# first 50 BMICs returned by the API — these are the high-coverage,
# procurement-relevant ones.
TARGETED_PARAMS = {
    "Function",
    "Battery Chemistry",
    "Number of Cells",
    "Fault Protection",
    "Interface",
    "Operating Temperature",
    "Mounting Type",
    "Package / Case",
    "Supplier Device Package",
    "Voltage - Supply",
    "Current - Supply",
    "Current - Output",
    "Voltage - Input",
    "Voltage - Output",
}


def _load_creds() -> dict:
    if not CREDS_PATH.exists():
        sys.exit(
            f"No DigiKey credentials at {CREDS_PATH}.\n"
            f"See the setup instructions at the top of {__file__}."
        )
    return json.loads(CREDS_PATH.read_text())


def _mint_token(creds: dict) -> dict:
    """Exchange client credentials for a Bearer token. Token TTL ~10 min."""
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    body["minted_at"] = time.time()
    return body


def _refresh_if_needed(creds: dict, token_state: dict) -> dict:
    """Mint a new token if the current one is within 30s of expiry."""
    elapsed = time.time() - token_state["minted_at"]
    if elapsed >= token_state["expires_in"] - 30:
        return _mint_token(creds)
    return token_state


def _headers(creds: dict, token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-DIGIKEY-Client-Id": creds["client_id"],
        "X-DIGIKEY-Locale-Site": "US",
        "X-DIGIKEY-Locale-Language": "en",
        "X-DIGIKEY-Locale-Currency": "USD",
        "Content-Type": "application/json",
    }


_OP_TEMP_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[°º]?C\s*~\s*(-?\d+(?:\.\d+)?)\s*[°º]?C",
    re.IGNORECASE,
)


def _parse_op_temp(text: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """Parse 'Operating Temperature' strings like '-40°C ~ 85°C (TA)'.

    Returns (min_C, max_C). When unparseable, returns (None, None).
    """
    if not text or text == "-":
        return None, None
    m = _OP_TEMP_RE.search(text)
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None, None


def _price_at_quantity(variations: list[dict], target_qty: int) -> Optional[float]:
    """From the per-variation StandardPricing rows, return the unit price
    available at quantity >= ``target_qty`` (the cheapest break above
    that threshold). Returns None when no variation reaches that qty.
    """
    best: Optional[float] = None
    for v in variations or []:
        for pb in v.get("StandardPricing") or []:
            bq = pb.get("BreakQuantity") or 0
            up = pb.get("UnitPrice")
            if bq >= target_qty and up is not None:
                if best is None or up < best:
                    best = up
                break  # within a single variation, breaks are sorted asc
    return best


def _row_from_product(prod: dict) -> Optional[dict]:
    """Flatten one DigiKey product into a single-row dict.

    Drops rows missing a price target. We use the at-1 ``UnitPrice`` as
    the primary target (always populated, comparable across parts);
    record the at-1k break-down price as a secondary column when it
    exists.
    """
    unit_price = prod.get("UnitPrice")
    if unit_price is None or unit_price <= 0:
        return None

    row: dict[str, object] = {
        "mpn": prod.get("ManufacturerProductNumber"),
        "manufacturer": (prod.get("Manufacturer") or {}).get("Name"),
        "category": (prod.get("Category") or {}).get("Name"),
        "description": (prod.get("Description") or {}).get("ProductDescription"),
        "unit_price_at_1_usd": float(unit_price),
        "price_at_1k_usd": _price_at_quantity(
            prod.get("ProductVariations") or [], 1_000
        ),
        "quantity_available": prod.get("QuantityAvailable"),
    }

    op_min, op_max = None, None
    for p in prod.get("Parameters") or []:
        name = p.get("ParameterText")
        if name not in TARGETED_PARAMS:
            continue
        val = p.get("ValueText")
        if val is None or val == "-":
            continue
        if name == "Operating Temperature":
            op_min, op_max = _parse_op_temp(val)
            row["op_temp_min_C"] = op_min
            row["op_temp_max_C"] = op_max
        else:
            row[name] = val

    return row


def fetch(query: str, limit: int, batch: int = 50, sleep_s: float = 0.3,
          incremental_out: Optional[Path] = None) -> pd.DataFrame:
    """Page through the DigiKey keyword search until ``limit`` parts collected.

    Saves an incremental CSV after each page so an unexpected error mid-pull
    doesn't lose the data already pulled.
    """
    creds = _load_creds()
    token_state = _mint_token(creds)
    print(f"OAuth token acquired. Pulling up to {limit:,} parts for "
          f"'{query}' from DigiKey ...")

    rows: list[dict] = []
    offset = 0
    page = 0
    while len(rows) < limit:
        page += 1
        page_limit = min(batch, limit - len(rows))
        token_state = _refresh_if_needed(creds, token_state)
        r = requests.post(
            SEARCH_URL,
            headers=_headers(creds, token_state["access_token"]),
            json={"Keywords": query, "Limit": page_limit, "Offset": offset},
            timeout=60,
        )
        if r.status_code == 429:
            ra = r.headers.get("Retry-After", "60")
            print(f"  429 rate-limited. sleeping {ra}s.")
            time.sleep(int(ra))
            continue
        r.raise_for_status()
        data = r.json()

        products = data.get("Products") or []
        total = data.get("ProductsCount") or "?"
        if not products:
            print(f"  page {page}: no more products, stopping.")
            break

        kept = 0
        for prod in products:
            row = _row_from_product(prod)
            if row:
                rows.append(row)
                kept += 1

        remaining = r.headers.get("x-ratelimit-remaining", "?")
        print(f"  page {page:>2}: offset={offset:>4} returned={len(products):>3} "
              f"kept={kept:>3} total={len(rows):,}/{limit}  "
              f"quota_remaining={remaining}/1000 catalog_total={total}")

        offset += len(products)

        if incremental_out is not None and rows:
            incremental_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(incremental_out, index=False)

        time.sleep(sleep_s)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", default="battery management")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "data" / "bmics.csv",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch(args.query, args.limit, incremental_out=args.out)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df):,} rows × {df.shape[1]} columns -> {args.out}")
    print(f"  manufacturers ({df['manufacturer'].nunique()}): "
          f"{', '.join(sorted(df['manufacturer'].dropna().unique())[:8])} ...")
    print(f"  price-at-1 range: ${df['unit_price_at_1_usd'].min():.3f} - "
          f"${df['unit_price_at_1_usd'].max():.2f}  "
          f"median ${df['unit_price_at_1_usd'].median():.2f}")
    n1k = df['price_at_1k_usd'].notna().sum()
    if n1k:
        print(f"  price-at-1k available for {n1k}/{len(df)} parts: "
              f"median ${df['price_at_1k_usd'].dropna().median():.3f}")


if __name__ == "__main__":
    main()
