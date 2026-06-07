"""Compose three P2Predict models (components + PCB + assembly) into a
total PCBA cost prediction, with per-stage breakdown and a what-if mode.

This is the thin orchestration pattern — three independently-trained
models, composed in pure Python via the v0.8 public API. The thin
version is useful today; the principled version (correct interval
aggregation, hybrid SHAP composition) is flagged as Pro-tier / v1.x
in this case study's README.

Usage:
    python compose.py \\
        --component-model models/components.model \\
        --pcb-model models/pcb.model \\
        --assembly-model models/assembly.model \\
        --bom example_bom.csv

    # Or with a what-if scenario on top:
    python compose.py ... --bom example_bom.csv \\
        --swap-component "U1:NewIC,J1:NewConnector"

Status: TEMPLATE — runs against three trained models, but the example
BOM and the per-stage feature shapes need to match what the user
actually trained. See the TODO markers.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from p2predict import explain, load_model, predict_interval


# ──────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────


@dataclass
class StageResult:
    """One stage of the composed prediction (components, PCB, or assembly)."""

    name: str
    point_estimate: float
    low: Optional[float] = None
    high: Optional[float] = None
    # Per-item breakdown for the components stage (list of (refdes, prediction)
    # tuples). Empty for PCB / assembly which produce one number per board.
    items: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class CompositionResult:
    """Total PCBA cost and its decomposition into the three stages."""

    total: float
    naive_low: float
    naive_high: float
    stages: list[StageResult]

    def percent_of_total(self, stage_name: str) -> float:
        for stage in self.stages:
            if stage.name == stage_name:
                return 100.0 * stage.point_estimate / self.total if self.total else 0.0
        return 0.0


# ──────────────────────────────────────────────────────────────────
# Composition logic
# ──────────────────────────────────────────────────────────────────


def predict_stage(
    model_metadata: dict, rows: pd.DataFrame, coverage: float = 0.90
) -> StageResult:
    """Run point prediction + likely-range interval on one stage's model.

    ``rows`` is the feature DataFrame for that stage:
      - For the component stage: one row per BOM line, the columns
        matching what the component model was trained on.
      - For PCB and assembly: a single-row DataFrame describing the
        bare board / assembly operation.
    """
    model = model_metadata["model"]
    calibration = model_metadata.get("calibration")
    target = model_metadata["target_feature"]

    predictions = model.predict(rows)
    point = float(predictions.sum())

    low: Optional[float] = None
    high: Optional[float] = None
    if calibration is not None:
        intervals = predict_interval(model, rows, calibration, coverage)
        low = float(sum(ir.low for ir in intervals))
        high = float(sum(ir.high for ir in intervals))

    items: list[tuple[str, float]] = []
    if "refdes" in rows.columns and len(rows) > 1:
        # Component stage — surface per-component contributions so the
        # caller can show the biggest line-item drivers.
        items = list(zip(rows["refdes"].tolist(), predictions.tolist()))

    return StageResult(
        name=target, point_estimate=point, low=low, high=high, items=items
    )


def compose_pcba(
    component_model: dict,
    pcb_model: dict,
    assembly_model: dict,
    bom: pd.DataFrame,
    pcb_spec: pd.DataFrame,
    assembly_spec: pd.DataFrame,
    coverage: float = 0.90,
) -> CompositionResult:
    """The thin composition pattern: three predictions, summed.

    Returns the breakdown alongside the total so the caller can show
    which stage is driving cost — the procurement-natural question.
    """
    components = predict_stage(component_model, bom, coverage)
    components.name = "Components"
    pcb = predict_stage(pcb_model, pcb_spec, coverage)
    pcb.name = "PCB"
    assembly = predict_stage(assembly_model, assembly_spec, coverage)
    assembly.name = "Assembly"

    stages = [components, pcb, assembly]
    total = sum(s.point_estimate for s in stages)

    # Naive summed interval. See the README — this *overshoots* the true
    # coverage at the BOM level. We carry it because it's useful as a
    # rough upper bound, but compose.py prints a clear caveat alongside.
    have_intervals = all(s.low is not None and s.high is not None for s in stages)
    naive_low = sum(s.low for s in stages) if have_intervals else float("nan")
    naive_high = sum(s.high for s in stages) if have_intervals else float("nan")

    return CompositionResult(
        total=total, naive_low=naive_low, naive_high=naive_high, stages=stages
    )


# ──────────────────────────────────────────────────────────────────
# Pretty printing — the procurement-natural output
# ──────────────────────────────────────────────────────────────────


def print_breakdown(result: CompositionResult) -> None:
    print(f"\nPredicted PCBA cost: ${result.total:,.2f}\n")
    for stage in result.stages:
        share = result.percent_of_total(stage.name)
        if stage.low is not None and stage.high is not None:
            range_str = f"  (per-stage 90% range ${stage.low:,.2f}–${stage.high:,.2f})"
        else:
            range_str = ""
        print(f"  {stage.name:<12} ${stage.point_estimate:7,.2f}  ({share:4.1f}%){range_str}")

    if not (result.naive_low != result.naive_low):  # not NaN
        print(
            f"\nNaive summed 90% range:  ${result.naive_low:,.2f} – ${result.naive_high:,.2f}\n"
            "  ⚠ This overshoots the true 90% coverage at the BOM level — the sum of\n"
            "    per-stage bounds is wider than a proper aggregated interval. Treat\n"
            "    per-stage ranges as informative; the summed range as an upper bound.\n"
            "    Correct interval aggregation is a Pro-tier / v1.x feature."
        )

    if any(s.items for s in result.stages):
        print("\nTop component contributors:\n")
        all_items: list[tuple[str, float]] = []
        for stage in result.stages:
            all_items.extend(stage.items)
        for refdes, value in sorted(all_items, key=lambda kv: kv[1], reverse=True)[:5]:
            print(f"  {refdes:<10} ${value:,.2f}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-model", type=Path, required=True,
        help="Path to a P2Predict component-cost model (train from "
        "case-studies/battery-management-ics/).",
    )
    parser.add_argument(
        "--pcb-model", type=Path, required=True,
        help="Path to a P2Predict PCB-cost model.",
    )
    parser.add_argument(
        "--assembly-model", type=Path, required=True,
        help="Path to a P2Predict assembly-cost model.",
    )
    parser.add_argument(
        "--bom", type=Path, required=True,
        help="CSV with one row per BOM line. Columns must match what the "
        "component model was trained on, plus a 'refdes' column for the "
        "reference designator (e.g. C1, R5, U2).",
    )
    parser.add_argument(
        "--pcb-spec", type=Path, default=None,
        help="CSV (single row) with the bare-PCB specs. Columns must match "
        "what the PCB model was trained on (layer_count, area_sq_in, "
        "finish, copper_weight, ...). If omitted, uses example_pcb_spec.csv.",
    )
    parser.add_argument(
        "--assembly-spec", type=Path, default=None,
        help="CSV (single row) with the assembly operation specs. Columns "
        "must match what the assembly model was trained on (placement_count, "
        "double_sided, test_coverage, ...). If omitted, uses "
        "example_assembly_spec.csv.",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.90,
        help="Per-stage likely-range coverage (default 0.90).",
    )
    parser.add_argument(
        "--swap-component", default=None,
        help='What-if: comma-separated REFDES:NEW_PART_NUMBER swaps. '
        'Example: --swap-component "U1:NEW-IC-PN,J1:NEW-CONNECTOR-PN"',
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the procurement-readable "
        "table. Useful when calling from an agent or a downstream script.",
    )
    args = parser.parse_args()

    component_model = load_model(args.component_model)
    pcb_model = load_model(args.pcb_model)
    assembly_model = load_model(args.assembly_model)

    bom = pd.read_csv(args.bom)
    # TODO: load or default the PCB / assembly spec rows. For now, error
    # if the user hasn't provided one — keeps the scaffolding honest.
    if args.pcb_spec is None or args.assembly_spec is None:
        raise SystemExit(
            "Provide --pcb-spec and --assembly-spec for now. Example specs "
            "should be a single-row CSV with columns matching what each "
            "respective model was trained on. (Default specs will be added "
            "once the case study has real data.)"
        )
    pcb_spec = pd.read_csv(args.pcb_spec)
    assembly_spec = pd.read_csv(args.assembly_spec)

    # Base prediction.
    base = compose_pcba(
        component_model, pcb_model, assembly_model,
        bom, pcb_spec, assembly_spec, coverage=args.coverage,
    )

    # What-if path.
    if args.swap_component:
        swaps: dict[str, str] = {}
        for token in args.swap_component.split(","):
            ref, _, new_pn = token.strip().partition(":")
            if not ref or not new_pn:
                raise SystemExit(f"Bad --swap-component token: '{token}'")
            swaps[ref] = new_pn

        cf_bom = bom.copy()
        for ref, new_pn in swaps.items():
            mask = cf_bom["refdes"] == ref
            if not mask.any():
                raise SystemExit(f"Refdes {ref} not in BOM.")
            # TODO: in a real demo, replace not just the part-number but
            # also the technical features the component model was trained
            # on (capacitance, voltage, package, ...). The cleanest path
            # is to look up the new part in the component catalog and
            # substitute the row entirely.
            cf_bom.loc[mask, "part_number"] = new_pn

        cf = compose_pcba(
            component_model, pcb_model, assembly_model,
            cf_bom, pcb_spec, assembly_spec, coverage=args.coverage,
        )

        if args.json:
            print(json.dumps({
                "base": _result_to_dict(base),
                "counterfactual": _result_to_dict(cf),
                "delta": cf.total - base.total,
                "delta_pct": 100.0 * (cf.total - base.total) / base.total
                if base.total else 0.0,
                "swaps": swaps,
            }, indent=2))
            return

        print_breakdown(base)
        print("\n" + "─" * 60)
        print(f"What-if: swapped {swaps}\n")
        print_breakdown(cf)
        delta = cf.total - base.total
        pct = 100.0 * delta / base.total if base.total else 0.0
        print(f"\nNet PCBA cost change: ${delta:+,.2f} ({pct:+.1f}%)")
        return

    if args.json:
        print(json.dumps(_result_to_dict(base), indent=2))
        return

    print_breakdown(base)


def _result_to_dict(result: CompositionResult) -> dict:
    return {
        "total": result.total,
        "naive_summed_range": [result.naive_low, result.naive_high],
        "stages": [
            {
                "name": s.name,
                "point_estimate": s.point_estimate,
                "low": s.low,
                "high": s.high,
                "share_of_total": result.percent_of_total(s.name),
                "items": [{"ref": ref, "predicted": v} for ref, v in s.items],
            }
            for s in result.stages
        ],
    }


if __name__ == "__main__":
    main()
