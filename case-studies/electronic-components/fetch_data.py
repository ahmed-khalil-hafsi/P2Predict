"""Fetch electronic-component pricing data via the Octopart API v4.

Run once before train.py and predict_examples.py.

Usage:
    export OCTOPART_API_KEY="..."
    python fetch_data.py [--category capacitors|resistors|connectors]
                        [--limit 10000]
                        [--out data/components.csv]

The output CSV has one row per component, with the columns the case study
expects (manufacturer, package, voltage, capacitance, tolerance, lead_time,
unit_price_1k). See README.md in this folder for the feature shape.

Status: TEMPLATE — the Octopart query is sketched but not wired. See the
TODOs below.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# TODO: pin the Octopart client version once you decide on a library.
# Options:
#   - The official Python client (if it exists) — check
#     https://octopart.com/api/v4/reference for the recommended client.
#   - Or call the GraphQL endpoint directly with `requests` — simpler,
#     fewer surprises.
# OCTOPART_GRAPHQL_URL = "https://octopart.com/api/v4/endpoint"


def fetch_components(api_key: str, category: str, limit: int) -> list[dict]:
    """Pull `limit` components in `category` from Octopart.

    Returns a list of dicts shaped like the CSV columns the case study
    expects:
        manufacturer, package, voltage, capacitance, tolerance,
        lead_time, unit_price_1k

    TODO: implement the actual GraphQL query. See
    https://octopart.com/api/v4/reference for the schema. The query
    should:
      - Filter by category (capacitors / resistors / connectors).
      - Page through results (Octopart returns ~100 per page).
      - Pull the price at the 1,000-unit quantity break specifically —
        prices vary substantially across quantity breaks and the case
        study target needs to be consistent.
      - Skip components missing any of the spec features we model on
        (rather than imputing — outlier handling at training time is
        better than synthetic spec values).
    """
    raise NotImplementedError(
        "Octopart query not yet wired. See the docstring TODOs and "
        "https://octopart.com/api/v4/reference for the GraphQL schema."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        choices=["capacitors", "resistors", "connectors"],
        default="capacitors",
        help="Component category. Pick one — the case study reads cleaner "
        "with a single focus.",
    )
    parser.add_argument(
        "--limit", type=int, default=10_000,
        help="Maximum number of components to fetch.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "data" / "components.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OCTOPART_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set OCTOPART_API_KEY in your environment. See "
            "https://octopart.com/api/v4/reference for how to get one."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    components = fetch_components(api_key, args.category, args.limit)

    # TODO: serialise to CSV. Use pandas so the schema matches what
    # p2predict-train expects (header row of column names).
    import pandas as pd
    pd.DataFrame(components).to_csv(args.out, index=False)
    print(f"wrote {len(components)} rows → {args.out}")


if __name__ == "__main__":
    main()
