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

from modules.cmdline_io import print_logo
from modules.explain import Explanation, explain_row, top_drivers
from modules.intervals import coverage_health, predict_interval
from modules.trained_model_io import LoadModel


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


def _print_explanation(console, explanation: Explanation, target_name: str) -> None:
    """Render a SHAP attribution table for a single-row prediction.

    Non-log models: standard additive decomposition in target units. The
    baseline plus the sum of contributions exactly reproduces the
    prediction (within floating-point) per SHAP's local-accuracy axiom.

    Log-target models: per-feature *multiplicative factors* are the
    axiomatically clean per-feature attribution in price space (their
    product times the baseline reproduces the prediction). We additionally
    show an approximate dollar attribution — clearly labelled as approximate
    — for procurement readability.
    """
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
        # Show contributions in decreasing |magnitude| so the drivers are obvious.
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
        # Log-target case: lead with the strict (multiplicative) attribution
        # and follow with the approximate dollar attribution.
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
    """Render one likely-range result. Deliberate language choices:
    'likely range', natural-frequency framing (n in 10), no
    'confidence interval' or 'alpha' anywhere — the audience is
    procurement and engineering, not statisticians.
    """
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
def main(model, predict_using, predict_file, explain_flag, interval_coverage):
    console = Console()

    print("")
    print_logo()
    print("")

    if not model:
        model = questionary.path("Enter model file path (.model)").ask()
        if not model:
            console.print("Aborted: please enter the path to the trained model.", style="bold red")
            raise SystemExit(1)

    loaded = LoadModel(model)
    trained = loaded["model"]
    if not trained:
        console.print("Aborted: the selected model is corrupt.", style="bold red")
        raise SystemExit(1)

    console.print(f"'{model}' successfully loaded.", style="bold white")
    if loaded.get("log_target"):
        console.print("(log-target transform active)", style="italic")
    print("")

    inner = _inner_pipeline(trained)
    feature_types, all_categories = _extract_feature_info(inner)

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
            console.print(
                f"Likely range disabled: {warning}.", style="italic yellow"
            )
            interval_coverage = None
        elif warning:
            interval_soft_warning = warning

    features_dict = {}
    if predict_using:
        features_dict = dict(item.split(":") for item in predict_using.split(","))
        features_df = _coerce_features(pd.DataFrame([features_dict]), feature_types)
        y = trained.predict(features_df)
        features_df["prediction"] = y
        console.print(Panel(Pretty(features_df), title="Prediction"))
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
            explanation = explain_row(trained, features_df[loaded["features"]], background)
            _print_explanation(console, explanation, target_name)

    elif predict_file:
        features_df = pd.read_csv(predict_file)
        features_df = _coerce_features(features_df, feature_types)
        y = trained.predict(features_df)
        features_df[target_name] = y
        if interval_coverage is not None:
            intervals = predict_interval(
                trained, features_df[loaded["features"]],
                calibration, coverage=interval_coverage / 100.0,
            )
            features_df[f"{target_name}_low"] = [ir.low for ir in intervals]
            features_df[f"{target_name}_high"] = [ir.high for ir in intervals]
        if explain_flag:
            # Add top-3 driver columns. We could batch-explain in one shot,
            # but per-row keeps the API simple and the call count is tiny in
            # the typical procurement RFQ batch (tens to low hundreds).
            top1, top2, top3 = [], [], []
            for i in range(len(features_df)):
                row = features_df.iloc[[i]][loaded["features"]]
                ex = explain_row(trained, row, background)
                drivers = top_drivers(ex, n=3)
                # Format: "Feature (factor)" for log-target, "Feature (±value)" otherwise.
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
        features_df.to_csv(predict_file, index=False)
        console.print(Panel(Pretty(features_df), title="Prediction"))

    else:
        for feature in loaded["features"]:
            if feature in all_categories:
                value = questionary.select(
                    f"Select a value for {feature}:",
                    choices=[str(c) for c in all_categories[feature]],
                ).ask()
            else:
                value = questionary.text(f"Enter a numeric value for {feature}:").ask()
            if not value:
                console.print(f"Aborted: please enter a value for {feature}.", style="bold red")
                raise SystemExit(1)
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

    return y


if __name__ == "__main__":
    main()
