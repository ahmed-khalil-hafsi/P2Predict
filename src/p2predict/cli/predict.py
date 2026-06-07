import os
import sys
from typing import Optional

import click
import numpy as np
import pandas as pd
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from p2predict.cmdline_io import print_logo
from p2predict.explain import Explanation, explain_batch, explain_row, top_drivers
from p2predict.intervals import coverage_health, predict_interval
from p2predict.json_output import JSON_SCHEMA_VERSION, emit, emit_error
from p2predict.trained_model_io import LoadModel
from p2predict.whatif import (
    WhatIfResult,
    compute_whatif,
    interaction_is_material,
    parse_changes,
)


def _inner_pipeline(model):
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def _extract_feature_info(pipeline):
    """Return (feature_types, all_categories) from the fitted preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_types = {}
    all_categories = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_types.update({col: "Numerical" for col in columns})
        elif name == "cat":
            feature_types.update({col: "Categorical" for col in columns})

            encoder = transformer
            if isinstance(transformer, Pipeline) and "onehot" in transformer.named_steps:
                encoder = transformer.named_steps["onehot"]

            if isinstance(encoder, (OneHotEncoder, OrdinalEncoder)) and hasattr(
                encoder, "categories_"
            ):
                all_categories = {
                    col: cat.tolist()
                    for col, cat in zip(columns, encoder.categories_)
                }

    return feature_types, all_categories


def _coerce_features(features_df, feature_types):
    for col, kind in feature_types.items():
        if col in features_df.columns and kind == "Numerical":
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
    return features_df


# ---------------------------------------------------------------------------
# Error path that respects --json. Use this instead of
# ``console.print(...); raise SystemExit(1)`` so an agent piping stdout
# still gets a parseable JSON error document on failure.
# ---------------------------------------------------------------------------


def _abort(json_mode: bool, console, code: str, message: str) -> None:
    if json_mode:
        emit_error("predict", code, message)
    console.print(f"Aborted: {message}", style="bold red")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Building blocks for the JSON result document. Each helper takes an
# in-memory object from the model stack and turns it into the schema
# shape defined in p2predict.json_output.
# ---------------------------------------------------------------------------


def _model_block(model_path: str, loaded: dict, target_name: str) -> dict:
    return {
        "path": model_path,
        "algorithm": loaded.get("model_name"),
        "target": target_name,
        "version": loaded.get("p2predict_version"),
        "log_target": bool(loaded.get("log_target", False)),
        "features": list(loaded.get("features", [])),
    }


def _interval_per_row(intervals) -> list[dict]:
    return [
        {"low": float(ir.low), "prediction": float(ir.prediction), "high": float(ir.high)}
        for ir in intervals
    ]


def _explanation_to_dict(explanation: Explanation) -> dict:
    out = {
        "baseline": float(explanation.baseline),
        "prediction": float(explanation.prediction),
        "log_target": bool(explanation.log_target),
        "contributions": [
            {"feature": k, "value": float(v)}
            for k, v in sorted(
                explanation.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
            )
        ],
        "residual": float(explanation.residual),
    }
    if explanation.log_target and explanation.multiplicative_factors is not None:
        out["multiplicative_factors"] = [
            {"feature": k, "factor": float(v)}
            for k, v in sorted(
                explanation.multiplicative_factors.items(),
                key=lambda kv: abs(np.log(kv[1])) if kv[1] > 0 else 0.0,
                reverse=True,
            )
        ]
        out["dollar_attribution"] = (
            [
                {"feature": k, "value": float(v)}
                for k, v in sorted(
                    explanation.dollar_attribution.items(),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )
            ]
            if explanation.dollar_attribution is not None
            else None
        )
    else:
        out["multiplicative_factors"] = None
        out["dollar_attribution"] = None
    return out


def _whatif_to_dict(result: WhatIfResult) -> dict:
    out = {
        "changes": {
            col: {"from": base_val, "to": cf_val}
            for col, (base_val, cf_val) in result.changes.items()
        },
        "base_prediction": float(result.base_prediction),
        "counterfactual_prediction": float(result.counterfactual_prediction),
        "delta": float(result.delta),
        "delta_pct": float(result.delta_pct),
        "log_target": bool(result.log_target),
        "multiplicative_factor": (
            float(result.multiplicative_factor)
            if result.multiplicative_factor is not None
            else None
        ),
        "changed_contributions": [
            {"feature": k, "value": float(v)}
            for k, v in sorted(
                result.changed_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
            )
        ],
        "interaction_contribution": float(result.interaction_contribution),
        "interaction_is_material": bool(interaction_is_material(result)),
        "base_interval": (
            {"low": float(result.base_interval.low), "high": float(result.base_interval.high)}
            if result.base_interval is not None
            else None
        ),
        "cf_interval": (
            {"low": float(result.cf_interval.low), "high": float(result.cf_interval.high)}
            if result.cf_interval is not None
            else None
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Rich rendering helpers. Unchanged from prior versions — they only run
# when --json is absent.
# ---------------------------------------------------------------------------


def _print_explanation(console, explanation: Explanation, target_name: str) -> None:
    table = Table(
        title="Prediction Explanation (SHAP)",
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )

    if not explanation.log_target:
        table.add_column("Feature")
        table.add_column("Contribution", justify="right")
        table.add_row(
            "[dim]Baseline (model expected value)[/dim]",
            f"{explanation.baseline:+.2f}",
        )
        ordered = sorted(
            explanation.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        for col, value in ordered:
            sign_style = "green" if value >= 0 else "red"
            table.add_row(col, f"[{sign_style}]{value:+.2f}[/{sign_style}]")
        table.add_row(
            "[bold]Predicted " + target_name + "[/bold]",
            f"[bold yellow]{explanation.prediction:+.2f}[/bold yellow]",
        )
        console.print(table)
    else:
        table.add_column("Feature")
        table.add_column("× Factor", justify="right")
        table.add_column("Effect", justify="right")
        baseline = explanation.baseline_price
        prediction = explanation.predicted_price
        table.add_row(
            "[dim]Baseline (geometric mean)[/dim]",
            "—",
            f"{baseline:,.2f}",
        )
        ordered = sorted(
            explanation.multiplicative_factors.items(),
            key=lambda kv: abs(np.log(kv[1])) if kv[1] > 0 else 0.0,
            reverse=True,
        )
        for col, factor in ordered:
            pct = (factor - 1.0) * 100.0
            sign_style = "green" if pct >= 0 else "red"
            table.add_row(
                col,
                f"×{factor:.3f}",
                f"[{sign_style}]{pct:+.1f}%[/{sign_style}]",
            )
        table.add_row(
            "[bold]Predicted " + target_name + "[/bold]",
            "—",
            f"[bold yellow]{prediction:,.2f}[/bold yellow]",
        )
        console.print(table)
        console.print(
            "Multiplicative factors are strict SHAP in price space "
            "(their product equals predicted / baseline).",
            style="italic dim",
        )
        if explanation.dollar_attribution is not None:
            d_table = Table(
                title="Approximate Dollar Attribution (rescaled, not strict SHAP)",
                show_header=True,
                header_style="bold magenta",
                expand=False,
            )
            d_table.add_column("Feature")
            d_table.add_column("Approx. contribution", justify="right")
            ordered_d = sorted(
                explanation.dollar_attribution.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            for col, value in ordered_d:
                sign_style = "green" if value >= 0 else "red"
                d_table.add_row(col, f"[{sign_style}]{value:+,.2f}[/{sign_style}]")
            console.print(d_table)

    if abs(explanation.residual) > 1e-3 * max(1.0, abs(explanation.prediction)):
        console.print(
            f"Note: local-accuracy residual is {explanation.residual:+.3g} "
            "(non-trivial; the SHAP/model wiring may need a look).",
            style="italic yellow",
        )


def _print_interval(console, interval_result, target_name: str, coverage_pct: int) -> None:
    out_of_10 = round(coverage_pct / 10)
    table = Table(
        title=f"Likely range ({coverage_pct}%)",
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column(f"Low {target_name}", justify="right")
    table.add_column(f"Predicted {target_name}", justify="right")
    table.add_column(f"High {target_name}", justify="right")
    table.add_row(
        f"[cyan]{interval_result.low:,.2f}[/cyan]",
        f"[bold yellow]{interval_result.prediction:,.2f}[/bold yellow]",
        f"[cyan]{interval_result.high:,.2f}[/cyan]",
    )
    console.print(table)
    console.print(
        f"The {target_name.lower()} for about {out_of_10} in 10 similar "
        f"parts falls in this range. Quotes outside it are unusual "
        "and worth questioning.",
        style="italic dim",
    )


def _print_whatif(console, result: WhatIfResult, target_name: str) -> None:
    headline_table = Table(
        title="What-if Analysis", show_header=True, header_style="bold magenta",
        expand=False,
    )
    headline_table.add_column("Scenario")
    headline_table.add_column(f"Predicted {target_name}", justify="right")
    if result.base_interval is not None:
        headline_table.add_column("Likely range", justify="right")
    headline_table.add_row(
        "Base",
        f"{result.base_prediction:,.2f}",
        *(
            [f"{result.base_interval.low:,.2f} – {result.base_interval.high:,.2f}"]
            if result.base_interval is not None
            else []
        ),
    )
    headline_table.add_row(
        "Counterfactual",
        f"{result.counterfactual_prediction:,.2f}",
        *(
            [f"{result.cf_interval.low:,.2f} – {result.cf_interval.high:,.2f}"]
            if result.cf_interval is not None
            else []
        ),
    )
    sign_style = "green" if result.delta >= 0 else "red"
    delta_label = "Change"
    delta_value = f"[{sign_style}]{result.delta:+,.2f}[/{sign_style}]"
    if result.log_target and result.multiplicative_factor is not None:
        delta_pct = (result.multiplicative_factor - 1.0) * 100.0
        delta_value += (
            f" ([{sign_style}]{delta_pct:+.1f}%[/{sign_style}], "
            f"×{result.multiplicative_factor:.3f})"
        )
    else:
        delta_value += f" ([{sign_style}]{result.delta_pct:+.1f}%[/{sign_style}])"
    headline_table.add_row(delta_label, delta_value, *([""] if result.base_interval is not None else []))
    console.print(headline_table)

    changes_table = Table(
        title="Changes applied",
        show_header=True, header_style="bold magenta", expand=False,
    )
    changes_table.add_column("Feature")
    changes_table.add_column("Base", justify="right")
    changes_table.add_column("Counterfactual", justify="right")
    for col, (base_val, cf_val) in result.changes.items():
        changes_table.add_row(col, str(base_val), str(cf_val))
    console.print(changes_table)

    attribution_table = Table(
        title=("Drivers of the change (SHAP × factor)" if result.log_target
               else "Drivers of the change (SHAP)"),
        show_header=True, header_style="bold magenta", expand=False,
    )
    attribution_table.add_column("Feature")
    if result.log_target:
        attribution_table.add_column("× Factor", justify="right")
        attribution_table.add_column("Effect", justify="right")
    else:
        attribution_table.add_column("Contribution", justify="right")
        attribution_table.add_column("Share", justify="right")

    abs_total = sum(abs(v) for v in result.changed_contributions.values()) + abs(
        result.interaction_contribution
    )
    abs_total = abs_total if abs_total > 1e-12 else 1.0
    ordered = sorted(
        result.changed_contributions.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    for col, value in ordered:
        if result.log_target and result.changed_multiplicative_factors is not None:
            factor = result.changed_multiplicative_factors[col]
            pct = (factor - 1.0) * 100.0
            sign_style = "green" if pct >= 0 else "red"
            attribution_table.add_row(
                col,
                f"×{factor:.3f}",
                f"[{sign_style}]{pct:+.1f}%[/{sign_style}]",
            )
        else:
            share = abs(value) / abs_total * 100.0
            sign_style = "green" if value >= 0 else "red"
            attribution_table.add_row(
                col,
                f"[{sign_style}]{value:+,.2f}[/{sign_style}]",
                f"{share:.0f}%",
            )

    if interaction_is_material(result):
        if result.log_target and result.interaction_multiplicative_factor is not None:
            factor = result.interaction_multiplicative_factor
            pct = (factor - 1.0) * 100.0
            sign_style = "green" if pct >= 0 else "red"
            attribution_table.add_row(
                "[dim]Other interaction effects[/dim]",
                f"×{factor:.3f}",
                f"[{sign_style}]{pct:+.1f}%[/{sign_style}]",
            )
        else:
            share = abs(result.interaction_contribution) / abs_total * 100.0
            sign_style = "green" if result.interaction_contribution >= 0 else "red"
            attribution_table.add_row(
                "[dim]Other interaction effects[/dim]",
                f"[{sign_style}]{result.interaction_contribution:+,.2f}[/{sign_style}]",
                f"{share:.0f}%",
            )
    console.print(attribution_table)

    if result.log_target:
        console.print(
            "Factors multiply: × Region × Supplier × ... = total change factor.",
            style="italic dim",
        )
    else:
        console.print(
            "Contributions add up to the total change. "
            "Features you didn't change can still show up here when there are interactions in the model.",
            style="italic dim",
        )


@click.command()
@click.option("-m", "--model", type=click.Path(exists=True),
              help="Path to the trained model file (.model)")
@click.option("-p", "--predict_using",
              help='Feature values, e.g. "weight:100,color:red"')
@click.option("-i", "--predict_file", type=click.Path(exists=True),
              help="CSV file containing feature values for batch prediction")
@click.option("--explain", "explain_flag", is_flag=True, default=False,
              help="Show per-feature SHAP attribution alongside the prediction.")
@click.option("--interval", "interval_coverage", type=click.IntRange(1, 99),
              default=None,
              help="Show the model's likely range for the prediction "
                   "(e.g. --interval 90 for a range that contains the value "
                   "for about 9 in 10 similar parts). Range is calibrated on "
                   "the training holdout; see the README for the math.")
@click.option("--whatif", "whatif_spec", default=None,
              help='What-if comparison. Takes a base scenario from -p (or '
                   'CSV row 0 from -i) and compares it to a counterfactual '
                   'where one or more features change. Same format as -p, '
                   'e.g. --whatif "Region:EU,Supplier:B".')
@click.option("--json", "json_mode", is_flag=True, default=False,
              help="Emit machine-readable JSON to stdout instead of "
                   "Rich-formatted tables. Useful for agents and scripts. "
                   "See p2predict.json_output for the schema.")
def main(model, predict_using, predict_file, explain_flag, interval_coverage,
         whatif_spec, json_mode):
    # Under --json, redirect Rich's console to /dev/null so any
    # console.print() that escapes a guard does not corrupt the JSON
    # document on stdout. The schema is the contract; this is the belt.
    if json_mode:
        console = Console(file=open(os.devnull, "w"))
    else:
        console = Console()

    if not json_mode:
        print("")
        print_logo()
        print("")

    if not model:
        if json_mode:
            _abort(json_mode, console, "missing_model",
                   "--model (or -m) is required when using --json.")
        model = questionary.path("Enter model file path (.model)").ask()
        if not model:
            _abort(json_mode, console, "missing_model",
                   "please enter the path to the trained model.")

    loaded = LoadModel(model)
    trained = loaded["model"]
    if not trained:
        _abort(json_mode, console, "corrupt_model",
               "the selected model is corrupt.")

    if not json_mode:
        console.print(f"'{model}' successfully loaded.", style="bold white")
        if loaded.get("log_target"):
            console.print("(log-target transform active)", style="italic")
        print("")

    inner = _inner_pipeline(trained)
    feature_types, all_categories = _extract_feature_info(inner)

    if not json_mode:
        table = Table(title="Model Features", show_header=True, header_style="bold magenta")
        table.add_column("Feature", style="dim", width=20)
        table.add_column("Type", justify="right")
        for feature, feature_type in feature_types.items():
            table.add_row(feature, feature_type)
        console.print(table)

        console.print(f"\nTarget feature: [bold blue]'{loaded['target_feature']}'[/bold blue]")

        if all_categories:
            console.print("\nCategorical Features:")
            for feature, categories in all_categories.items():
                console.print(f"[bold]{feature}[/bold]")
                for category in categories:
                    console.print(f"  • {category}")
                console.print("")
        else:
            console.print("No categorical features to display.", style="italic")

        console.print("\n" + "=" * 50 + "\n")

    background = loaded.get("background_sample")
    target_name = loaded["target_feature"]
    calibration = loaded.get("calibration")

    # Decide whether the model can support a likely-range interval at all,
    # and whether to soft-warn about a small calibration set.
    interval_soft_warning: Optional[str] = None
    if interval_coverage is not None:
        warning = coverage_health(calibration)
        if warning and (calibration is None or calibration.get("n_calibration", 0) == 0):
            if not json_mode:
                console.print(
                    f"Likely range disabled: {warning}.", style="italic yellow"
                )
            interval_coverage = None
        elif warning:
            interval_soft_warning = warning

    # --whatif is inline-only (needs a single base scenario).
    if whatif_spec is not None and predict_file:
        _abort(json_mode, console, "whatif_in_batch",
               "--whatif is not supported in batch mode (-i). "
               "Use -p to specify a base scenario.")

    # Build the JSON response throughout. Mode-dependent blocks get
    # added as we go; we emit the whole thing at the end under --json.
    response: dict = {
        "schema_version": JSON_SCHEMA_VERSION,
        "command": "predict",
        "model": _model_block(model, loaded, target_name),
    }

    features_dict = {}
    if predict_using:
        response["mode"] = "inline"
        features_dict = dict(item.split(":") for item in predict_using.split(","))
        features_df = _coerce_features(pd.DataFrame([features_dict]), feature_types)
        y = trained.predict(features_df)
        features_df["prediction"] = y
        if not json_mode:
            console.print(Panel(Pretty(features_df), title="Prediction"))

        response["predictions"] = [
            {"input": features_dict, "prediction": float(y[0])}
        ]

        if interval_coverage is not None:
            [interval_result] = predict_interval(
                trained, features_df[loaded["features"]],
                calibration, coverage=interval_coverage / 100.0,
            )
            if not json_mode:
                print("")
                _print_interval(console, interval_result, target_name, interval_coverage)
                if interval_soft_warning:
                    console.print(f"Note: {interval_soft_warning}.", style="italic yellow")
            response["interval"] = {
                "coverage": interval_coverage / 100.0,
                "per_row": _interval_per_row([interval_result]),
                "soft_warning": interval_soft_warning,
            }

        if explain_flag:
            explanation = explain_row(trained, features_df[loaded["features"]], background)
            if not json_mode:
                print("")
                _print_explanation(console, explanation, target_name)
            response["explanation"] = [_explanation_to_dict(explanation)]

        if whatif_spec is not None:
            try:
                changes = parse_changes(whatif_spec)
            except ValueError as exc:
                _abort(json_mode, console, "bad_whatif", str(exc))
            try:
                whatif_result = compute_whatif(
                    trained,
                    features_df[loaded["features"]],
                    changes,
                    feature_types,
                    background_X=background,
                    calibration=calibration if interval_coverage is not None else None,
                    coverage=(interval_coverage or 90) / 100.0,
                )
            except ValueError as exc:
                _abort(json_mode, console, "bad_whatif", str(exc))
            if not json_mode:
                print("")
                _print_whatif(console, whatif_result, target_name)
            response["whatif"] = _whatif_to_dict(whatif_result)

    elif predict_file:
        response["mode"] = "batch"
        features_df = pd.read_csv(predict_file)
        features_df = _coerce_features(features_df, feature_types)
        y = trained.predict(features_df)
        features_df[target_name] = y

        per_row = [
            {"input": features_df[loaded["features"]].iloc[i].to_dict(),
             "prediction": float(y[i])}
            for i in range(len(features_df))
        ]
        response["predictions"] = per_row

        if interval_coverage is not None:
            intervals = predict_interval(
                trained, features_df[loaded["features"]],
                calibration, coverage=interval_coverage / 100.0,
            )
            features_df[f"{target_name}_low"] = [ir.low for ir in intervals]
            features_df[f"{target_name}_high"] = [ir.high for ir in intervals]
            response["interval"] = {
                "coverage": interval_coverage / 100.0,
                "per_row": _interval_per_row(intervals),
                "soft_warning": interval_soft_warning,
            }
        if explain_flag:
            top1, top2, top3 = [], [], []
            per_row_explanations = []
            explanations = explain_batch(
                trained, features_df[loaded["features"]], background
            )
            for ex in explanations:
                per_row_explanations.append(_explanation_to_dict(ex))
                drivers = top_drivers(ex, n=3)
                formatted = []
                for col, value in drivers:
                    if ex.log_target:
                        pct = (value - 1.0) * 100.0
                        formatted.append(f"{col} ({pct:+.1f}%)")
                    else:
                        formatted.append(f"{col} ({value:+.2f})")
                while len(formatted) < 3:
                    formatted.append("")
                top1.append(formatted[0])
                top2.append(formatted[1])
                top3.append(formatted[2])
            features_df["top1_driver"] = top1
            features_df["top2_driver"] = top2
            features_df["top3_driver"] = top3
            response["explanation"] = per_row_explanations

        features_df.to_csv(predict_file, index=False)
        response["batch"] = {
            "csv_path": str(predict_file), "n_rows": int(len(features_df)),
        }
        if not json_mode:
            console.print(Panel(Pretty(features_df), title="Prediction"))

    else:
        # Interactive mode is incompatible with --json.
        if json_mode:
            _abort(json_mode, console, "missing_input",
                   "interactive mode is not supported with --json. "
                   "Use -p (inline) or -i (batch).")
        response["mode"] = "interactive"
        for feature in loaded["features"]:
            if feature in all_categories:
                value = questionary.select(
                    f"Select a value for {feature}:",
                    choices=[str(c) for c in all_categories[feature]],
                ).ask()
            else:
                value = questionary.text(f"Enter a numeric value for {feature}:").ask()
            if not value:
                _abort(json_mode, console, "missing_input",
                       f"please enter a value for {feature}.")
            features_dict[feature] = value

        features_df = _coerce_features(pd.DataFrame([features_dict]), feature_types)
        y = trained.predict(features_df)
        features_df["prediction"] = y

        table = Table(title="Prediction Results", show_header=True, header_style="bold magenta")
        for column in features_df.columns:
            table.add_column(column, style="cyan", justify="right")
        table.add_row(*[str(val) for val in features_df.iloc[0]])
        console.print(Panel(table, expand=False, border_style="green", padding=(1, 1)))

        prediction_value = features_df["prediction"].iloc[0]
        console.print(
            f"\n[bold]Predicted {loaded['target_feature']}:[/bold] "
            f"[yellow]{prediction_value:.2f}[/yellow]"
        )
        if interval_coverage is not None:
            print("")
            [interval_result] = predict_interval(
                trained, features_df[loaded["features"]],
                calibration, coverage=interval_coverage / 100.0,
            )
            _print_interval(console, interval_result, target_name, interval_coverage)
            if interval_soft_warning:
                console.print(f"Note: {interval_soft_warning}.", style="italic yellow")
        if explain_flag:
            print("")
            explanation = explain_row(
                trained, features_df[loaded["features"]], background
            )
            _print_explanation(console, explanation, target_name)

    if json_mode:
        emit(response)
    return y


if __name__ == "__main__":
    main()
