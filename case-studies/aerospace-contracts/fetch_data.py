"""Fetch FPDS contract data from USAspending.gov.

Public-domain federal procurement data. Filter to one PSC code and a
fiscal-year range so the model is learning within a single commodity
category (mixing PSCs gives a meaningless model — same logic as not
mixing capacitors with connectors in the EE case study).

Run once before train.py and predict_examples.py.

Usage:
    python fetch_data.py --psc 1560 --fiscal-year 2023
    python fetch_data.py --psc 1560 --fiscal-year-from 2018 \\
                         --fiscal-year-to 2023 --out data/contracts.csv

Status: TEMPLATE — USAspending API call is sketched but needs to be
wired to the actual award-search endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"


def fetch_contracts(
    psc: str, fy_from: int, fy_to: int, out: Path
) -> Path:
    """Pull FPDS contract awards for one PSC across a fiscal-year range.

    Writes CSV with columns aligned to the case study's feature list
    (awarding_agency, contractor, set_aside, period_of_performance_months,
    place_of_performance_state, competition_type, fiscal_year,
    award_date, obligated_amount).

    TODO: implement the actual POST against USAspending's spending_by_award
    endpoint. The relevant payload shape:

        {
          "subawards": false,
          "fields": [...],
          "filters": {
            "award_type_codes": ["A", "B", "C", "D"],  # contracts only
            "time_period": [...],
            "psc_codes": [psc],
          },
          "page": 1,
          "limit": 100,
          "sort": "Award Amount",
          "order": "desc"
        }

    See https://api.usaspending.gov/docs/endpoints for full schema.
    Page through until empty results. Don't combine PSCs in one model.
    """
    raise NotImplementedError(
        "USAspending fetch not wired yet. See the docstring TODO and "
        "https://api.usaspending.gov/docs/endpoints for the schema."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--psc", required=True,
        help='Product/Service Code, e.g. "1560" for Airframe Structural '
        "Components. https://www.acquisition.gov/psc-manual",
    )
    parser.add_argument(
        "--fiscal-year", type=int,
        help="Single fiscal year (US Government FY: Oct 1 – Sep 30).",
    )
    parser.add_argument(
        "--fiscal-year-from", type=int,
        help="Range start, used with --fiscal-year-to.",
    )
    parser.add_argument(
        "--fiscal-year-to", type=int,
        help="Range end, inclusive.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "data" / "contracts.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.fiscal_year:
        fy_from = fy_to = args.fiscal_year
    elif args.fiscal_year_from and args.fiscal_year_to:
        fy_from = args.fiscal_year_from
        fy_to = args.fiscal_year_to
    else:
        sys.exit(
            "Pass either --fiscal-year or both --fiscal-year-from / "
            "--fiscal-year-to."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fetch_contracts(args.psc, fy_from, fy_to, args.out)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
