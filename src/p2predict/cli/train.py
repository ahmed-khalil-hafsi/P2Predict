import datetime
import os
import sys

# Check for --json *before* starting any module-level Halo spinner or
# Rich output — under JSON mode stdout must be exclusively the response
# document. Light-touch argv sniffing is fine here because Click won't
# rewrite "--json" in any way that breaks this check.
_JSON_MODE_FROM_ARGV = "--json" in sys.argv

from halo import Halo

if not _JSON_MODE_FROM_ARGV:
    spinner = Halo(text="Loading P2Predict", spinner="pong")
    spinner.start()
else:
    spinner = None

import click
import pandas as pd
import questionary
from rich.console import Console
from rich.prompt import Prompt

from p2predict import plotting
from p2predict.cmdline_io import print_feature_stats, print_feature_weights, print_logo
from p2predict.hpo_training import hyper_parameter_tuning
from p2predict.json_output import JSON_SCHEMA_VERSION, emit, emit_error
from p2predict.model_evals import evaluate_model
from p2predict.intervals import compute_calibration_residuals
from p2predict.outliers import (
    POLICIES as OUTLIER_POLICIES,
    apply_feature_outlier_policy,
    apply_outlier_policy,
)
from p2predict.feature_selection import (
    find_high_variation_features,
    find_no_variation_features,
    get_most_predictable_features,
)
from p2predict.prepare_data import prepare_data
from p2predict.trained_model_io import SaveModel, Serialize_Trained_Model, load_csv_file
from p2predict.training import (
    ALGORITHMS,
    auto_train,
    extract_feature_importances,
    resolve_log_target,
    start_training,
)
from p2predict.ui_console import print_dataframe

if spinner is not None:
    spinner.stop()


def _abort(json_mode: bool, console, code: str, message: str) -> None:
    """Same shape as predict.py — emit JSON error or red Rich abort."""
    if json_mode:
        emit_error("train", code, message)
    console.print(f"Aborted: {message}", style="bold red")
    raise SystemExit(1)


def _outlier_summary_block(summary: dict) -> dict:
    """Coerce the target-side outlier summary into JSON-shaped values."""
    return {
        "policy": summary.get("policy"),
        "applied": summary.get("applied"),
        "n_outliers": int(summary.get("n_outliers", 0)),
        "n_total": int(summary.get("n_total", 0)),
        "lower": (None if pd.isna(summary.get("lower", float("nan")))
                  else float(summary.get("lower"))),
        "upper": (None if pd.isna(summary.get("upper", float("nan")))
                  else float(summary.get("upper"))),
    }


def _feature_outlier_summary_block(summary: dict) -> dict:
    return {
        "policy": summary.get("policy"),
        "applied": summary.get("applied"),
        "n_outliers_total": int(summary.get("n_outliers_total", 0)),
        "n_total": int(summary.get("n_total", 0)),
        "per_column": {
            col: {
                "n_outliers": int(stats.get("n_outliers", 0)),
                "lower": (None if pd.isna(stats.get("lower", float("nan")))
                          else float(stats.get("lower"))),
                "upper": (None if pd.isna(stats.get("upper", float("nan")))
                          else float(stats.get("upper"))),
            }
            for col, stats in summary.get("per_column", {}).items()
        },
    }


@click.command()
@click.option("-i", "--input", type=click.Path(exists=True), default=None,
              help="Path to the CSV file containing the training dataset.")
@click.option("-t", "--target",
              help='Name of the feature to predict (e.g., "Price").')
@click.option("-x", "--expert", is_flag=True, default=None,
              help="Enable Expert Mode for more control over the training process.")
@click.option("-a", "--algorithm", type=click.Choice(list(ALGORITHMS)),
              help="ML algorithm for expert mode.")
@click.option("-v", "--verbose", is_flag=True, default=None,
              help="Enable verbose output.")
@click.option("-c", "--interactive", is_flag=True, default=None,
              help="Enable interactive mode for guided input.")
@click.option("-tf", "--training_features",
              help='Comma-separated list of features (e.g., "Weight,Size,Color").')
@click.option("-b", "--budget", type=click.Choice(["fast", "thorough"]), default="fast",
              help="HPO search budget. 'fast' = small search, 'thorough' = wider search (slower).")
@click.option("--max-features", "max_features", type=click.IntRange(min=2), default=6,
              show_default=True,
              help="Auto mode only: cap on how many top-ranked features auto-selection keeps. "
                   "Default 6 preserves prior behaviour. Pass a higher number (or use -tf to pick "
                   "features explicitly) when ranking suggests more columns are predictive. "
                   "Ignored in expert mode (the user selects features interactively or via -tf).")
@click.option("--tune/--no-tune", default=None,
              help="Expert mode only: run HPO on the chosen algorithm and save the tuned model.")
@click.option("--outliers", type=click.Choice(list(OUTLIER_POLICIES)), default="warn",
              help="How to handle outliers in the target column (Tukey IQR rule). "
                   "'warn' (default) = report only; 'drop' = remove rows; "
                   "'winsorize' = cap values; 'keep' = silent.")
@click.option("--feature-outliers", type=click.Choice(list(OUTLIER_POLICIES)), default="warn",
              help="How to handle outliers in the numerical feature columns "
                   "(Tukey IQR per column). 'drop' removes any row that has an "
                   "outlier in any feature column; 'winsorize' caps each column "
                   "at its own IQR bounds. 'warn' (default) = report only. "
                   "Categorical features are ignored.")
@click.option("--time-column", default=None,
              help="Name of a date/time column. When given, the train/test split and CV "
                   "become chronological (TimeSeriesSplit), which prevents look-ahead bias "
                   "for time-ordered data. The column is excluded from features.")
@click.option("--log-target", "log_target_mode",
              type=click.Choice(["auto", "on", "off"]), default="auto",
              show_default=True,
              help="Override the automatic skew-based decision on whether to wrap "
                   "the target with log/exp. 'auto' = wrap when scipy.stats.skew(y) > 1.0 "
                   "(the prior behaviour). 'on' = always wrap (the right default for "
                   "multiplicative quantities like prices/costs/weights, regardless of "
                   "sample skew — keeps conformal intervals strictly positive and SHAP "
                   "factors multiplicative). 'off' = never wrap. 'on' aborts cleanly "
                   "if any training target is non-positive.")
@click.option("--report", "report", type=click.Path(), default=None,
              help="Write the procurement-style PDF model-quality report to PATH "
                   "after training. Works in both auto and expert mode, and with "
                   "or without --interactive — pass this whenever you want the "
                   "PDF without answering an interactive prompt.")
@click.option("--json", "json_mode", is_flag=True, default=False,
              help="Emit machine-readable JSON to stdout instead of "
                   "Rich-formatted output. Useful for agents and scripts. "
                   "See p2predict.json_output for the schema.")
def train(input, target, expert, algorithm, verbose, interactive, training_features,
          budget, max_features, tune, outliers, feature_outliers, time_column,
          log_target_mode, report, json_mode):

    # Redirect Rich to /dev/null under --json so any console.print that
    # escapes a guard cannot corrupt the JSON document on stdout.
    if json_mode:
        console = Console(file=open(os.devnull, "w"))
    else:
        console = Console()

    response: dict = {
        "schema_version": JSON_SCHEMA_VERSION,
        "command": "train",
    }

    if not json_mode:
        print("")
        print_logo()
        print("")

    mode_label = "Expert mode" if expert else "Auto mode"
    response["mode"] = "expert" if expert else "auto"
    if not json_mode:
        console.print(f"Welcome to P2Predict! '{mode_label}' is active.", style="bold blue")

    # Interactive mode is incompatible with --json (it would prompt).
    if json_mode and interactive:
        _abort(json_mode, console, "interactive_with_json",
               "interactive mode is not supported with --json.")

    if interactive:
        if not input:
            input = questionary.path("Enter CSV file path").ask()
            if not input:
                _abort(json_mode, console, "missing_input",
                       "You must provide an input file.")
    else:
        if not input:
            _abort(json_mode, console, "missing_input",
                   "You must provide --input. Use -c for interactive mode.")
        if not target:
            _abort(json_mode, console, "missing_target",
                   "You must provide --target. Use -c for interactive mode.")

    if expert:
        if interactive and not algorithm:
            algorithm = questionary.select(
                "Please choose an ML algorithm:", choices=list(ALGORITHMS)
            ).ask()
            if not algorithm:
                _abort(json_mode, console, "missing_algorithm",
                       "You must select a training algorithm.")
        elif not interactive:
            if not algorithm:
                _abort(json_mode, console, "missing_algorithm",
                       "You must pre-select --algorithm in expert mode (or use -c).")
            if not training_features:
                _abort(json_mode, console, "missing_features",
                       "You must provide --training_features in expert mode (or use -c).")

    data = load_csv_file(input)
    rows_loaded = int(data.shape[0])

    if not json_mode:
        print("")
        console.print(
            f"Training file '{input}' imported into P2Predict > "
            f"{data.shape[0]} rows  x {data.shape[1]} columns loaded."
        )
        print("")

    if not target:
        target = questionary.select("Enter target column", choices=data.columns.tolist()).ask()
        if not target:
            _abort(json_mode, console, "missing_target",
                   "A target feature is required.")

    if time_column is not None and time_column not in data.columns:
        _abort(json_mode, console, "bad_time_column",
               f"--time-column '{time_column}' not found in CSV.")
    if time_column is not None:
        try:
            data[time_column] = pd.to_datetime(data[time_column])
        except Exception as exc:
            _abort(json_mode, console, "bad_time_column",
                   f"could not parse --time-column '{time_column}': {exc}")
        if not json_mode:
            console.print(
                f"Time-aware mode: train/test split and CV will be chronological on "
                f"'{time_column}'.",
                style="bold blue",
            )

    if target not in data.columns:
        _abort(json_mode, console, "unknown_target",
               f"--target '{target}' not found in CSV.")

    # Drop only rows whose TARGET is NA — those rows can't supervise training
    # and can't be scored. Rows with NAs only in *feature* columns are kept:
    # XGBoost handles them natively and the random_forest/ridge preprocessors
    # impute (see build_preprocessor). This replaces the old blanket
    # df.dropna() at CSV load, which silently discarded ~half the catalogue.
    rows_before_target_drop = int(data.shape[0])
    data = data[data[target].notna()]
    rows_dropped_target_na = rows_before_target_drop - int(data.shape[0])
    if data.empty:
        _abort(json_mode, console, "all_target_na",
               f"every row has a missing value in target column '{target}'.")
    if rows_dropped_target_na > 0 and not json_mode:
        console.print(
            f"Dropped {rows_dropped_target_na} row(s) with a missing "
            f"'{target}' value; {data.shape[0]} rows remain.",
            style="yellow",
        )

    data, outlier_summary = apply_outlier_policy(data, target, policy=outliers)
    if outlier_summary["n_outliers"] > 0 and not json_mode:
        pct = 100.0 * outlier_summary["n_outliers"] / max(outlier_summary["n_total"], 1)
        action_msg = {
            "keep": "kept as-is",
            "warn": "kept as-is — pass --outliers drop or winsorize to mitigate",
            "drop": "dropped",
            "winsorize": "winsorized to the IQR bounds",
        }[outliers]
        console.print(
            f"Outliers in '{target}': {outlier_summary['n_outliers']} of "
            f"{outlier_summary['n_total']} rows ({pct:.1f}%) outside "
            f"[{outlier_summary['lower']:.2f}, {outlier_summary['upper']:.2f}] — {action_msg}.",
            style="bold yellow",
        )
        print("")

    feature_outlier_candidates = [
        c for c in data.columns if c != target and c != time_column
    ]
    data, feature_outlier_summary = apply_feature_outlier_policy(
        data, feature_outlier_candidates, policy=feature_outliers
    )
    if feature_outlier_summary["n_outliers_total"] > 0 and not json_mode:
        pct = (
            100.0 * feature_outlier_summary["n_outliers_total"]
            / max(feature_outlier_summary["n_total"], 1)
        )
        feature_action_msg = {
            "keep": "kept as-is",
            "warn": "kept as-is — pass --feature-outliers drop or winsorize to mitigate",
            "drop": "rows dropped",
            "winsorize": "values winsorized per column",
        }[feature_outliers]
        affected = {
            col: stats for col, stats in feature_outlier_summary["per_column"].items()
            if stats["n_outliers"] > 0
        }
        affected_details = ", ".join(
            f"{col} ({stats['n_outliers']})" for col, stats in affected.items()
        )
        console.print(
            f"Outliers in feature columns: {feature_outlier_summary['n_outliers_total']} "
            f"of {feature_outlier_summary['n_total']} rows ({pct:.1f}%) affected "
            f"[{affected_details}] — {feature_action_msg}.",
            style="bold yellow",
        )
        print("")

    feature_data = data.drop(columns=[time_column]) if time_column else data
    high_vars = find_high_variation_features(feature_data)
    low_vars = find_no_variation_features(feature_data)

    if not json_mode:
        print("")
        console.print("Low-information features detected:")
        console.print(f"No information content: {low_vars}")
        console.print(f"High variation (potentially noisy): {high_vars}")
        print("")

    if interactive and (low_vars or high_vars):
        to_remove = questionary.checkbox(
            "Which features would you like to remove? ", choices=low_vars + high_vars
        ).ask()
        if to_remove:
            data = data.drop(to_remove, axis=1)
    elif low_vars:
        data = data.drop(low_vars, axis=1)

    feature_data = data.drop(columns=[time_column]) if time_column else data

    if not training_features:
        if expert:
            best_features_ranked = get_most_predictable_features(feature_data, target)
            if not json_mode:
                console.print("Best features detected for prediction:", style="bold white")
                print("")
                print_dataframe(best_features_ranked)

            options_list = [c for c in feature_data.columns.tolist() if c != target]
            selected_columns = questionary.checkbox(
                "Select the features for training: ", choices=options_list
            ).ask()
            if not selected_columns:
                _abort(json_mode, console, "missing_features",
                       "You must select training features.")
        else:
            ranked = get_most_predictable_features(feature_data, target, output_only_headers=True)
            n_ranked = len(ranked)
            cap = max(2, min(n_ranked, max_features))
            selected_columns = ranked.head(cap).tolist()
            if not json_mode:
                console.print(
                    f"Auto-selected features for training: {selected_columns}", style="bold blue"
                )
                if n_ranked > cap:
                    console.print(
                        f"Auto-selected {cap} of {n_ranked} features "
                        f"(use --max-features to override or pass -tf).",
                        style="italic",
                    )
                print("")
    else:
        requested = [c.strip() for c in training_features.split(",")]
        missing = [c for c in requested if c not in data.columns]
        if missing:
            _abort(json_mode, console, "unknown_features",
                   f"requested features not in CSV: {missing}")
        selected_columns = requested

    target_column = target

    if time_column is not None and time_column in selected_columns:
        selected_columns = [c for c in selected_columns if c != time_column]

    X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = prepare_data(
        data, selected_columns, target_column, time_column=time_column
    )
    time_aware = time_column is not None

    # Resolve --log-target up front so the same decision flows into auto,
    # expert, and the late-arriving expert+interactive tuning branch. The
    # 'on' safety check (y_train > 0) runs here rather than inside
    # resolve_log_target() so the CLI surfaces a friendly --json abort.
    if log_target_mode == "on":
        try:
            y_arr = y_train.to_numpy(dtype=float)
        except Exception:
            y_arr = None
        if y_arr is None or y_arr.size == 0 or (y_arr <= 0).any():
            _abort(json_mode, console, "log_target_non_positive",
                   "--log-target on requires all training targets to be strictly "
                   "positive; found non-positive values in y_train.")
    log_target_override, log_target_decision = resolve_log_target(
        y_train, mode=log_target_mode
    )

    if expert and interactive:
        if questionary.confirm("Plot histograms of the selected features?").ask():
            plotting.plot_histograms(data[selected_columns])
        if not json_mode:
            print("")

    if expert and not json_mode:
        console.print("Numerical feature analysis:", style="bold white")
        print("")
        print_feature_stats(data[list(numerical_cols)])
        print("")

    scores: dict = {}
    if expert:
        if tune is None and interactive:
            tune = questionary.confirm(
                "Run hyperparameter tuning (slower, usually higher accuracy)?"
            ).ask()
        tune = bool(tune)

        if not json_mode:
            inner_spinner = Halo(
                text=f"Training {algorithm} (tune={tune}, budget={budget})...", spinner="pong"
            )
            inner_spinner.start()
        else:
            inner_spinner = None

        model, feature_weights, log_target = start_training(
            X_train, y_train, numerical_cols, categorical_cols, algorithm,
            budget=budget, tune=tune, time_aware=time_aware,
            log_target=log_target_override,
        )
        if inner_spinner is not None:
            inner_spinner.stop()
            print_feature_weights(feature_weights)
            print("")
            if log_target:
                console.print(
                    "Note: log-target transform applied (target is positive and skewed).",
                    style="italic",
                )
    else:
        if not json_mode:
            inner_spinner = Halo(
                text=f"Auto-mode model selection (budget={budget})...", spinner="pong"
            )
            inner_spinner.start()
        else:
            inner_spinner = None

        model, algorithm, scores, log_target = auto_train(
            X_train, y_train, numerical_cols, categorical_cols,
            budget=budget, time_aware=time_aware,
            log_target=log_target_override,
        )
        if inner_spinner is not None:
            inner_spinner.stop()
            console.print(f"Selected best algorithm: [bold]{algorithm}[/bold]")
            for algo, score in scores.items():
                console.print(f"  {algo}: CV R² = {round(score, 3)}")
            if log_target:
                console.print(
                    "Note: log-target transform applied (target is positive and skewed).",
                    style="italic",
                )

    if inner_spinner is not None:
        inner_spinner.succeed("Training finished.")
        print("")

    mae, r2, p_value, rmse = evaluate_model(X_test, y_test, model)

    # Quality label, computed once and used in both the Rich and JSON paths.
    # Shared with the MCP layer via p2predict.quality (single source of truth).
    from p2predict.quality import r2_quality_label
    quality_label = r2_quality_label(r2)

    if not json_mode:
        if expert:
            console.print("Model Key Performance Metrics:", style="bold white")
            console.print(f"Model R² Score: {round(r2, 2)}")
            console.print(f"Mean Absolute Error: {round(mae, 2)}")
            console.print(f"RMSE: {round(rmse, 2)}")
            console.print(f"Residual bias p-value: {round(p_value, 4)}")
            print("")
        else:
            console.print("Model Performance Summary:", style="bold white")
            style = {"Excellent": "bold green",
                     "Good": "bold yellow",
                     "Needs Improvement": "bold red"}[quality_label]
            console.print(f"Model Quality: {quality_label}", style=style)
            console.print(f"R² Score: {round(r2 * 100, 1)}%")
            console.print(f"Mean Absolute Error: {round(mae, 2)}")
            console.print(f"RMSE: {round(rmse, 2)}")
            if p_value < 0.05:
                console.print(
                    "Residuals show systematic bias — consider expert mode for tuning.",
                    style="italic bold yellow",
                )
            if quality_label == "Needs Improvement":
                console.print(
                    "Recommendation: try expert mode with --tune, or collect more data.",
                    style="bold",
                )
            print("")

    # Fallback prompt for the legacy expert+interactive flow. Skip it if
    # the user already passed --report PATH — we'll write the report after
    # save and don't want to ask twice.
    if expert and interactive and report is None:
        if questionary.confirm("Generate the model quality PDF report?").ask():
            report = Prompt.ask("Enter PDF name (e.g., report.pdf)")

    if expert and interactive and not tune:
        if questionary.confirm(
            "Run hyperparameter tuning now to try for a better model?"
        ).ask():
            tune_spinner = Halo("Tuning...", spinner="pong")
            tune_spinner.start()
            tuned_model, tuned_score, log_target = hyper_parameter_tuning(
                X_train=X_train,
                y_train=y_train,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                algorithm=algorithm,
                budget=budget,
                time_aware=time_aware,
                log_target=log_target_override,
            )
            tune_spinner.stop()
            mae_t, r2_t, _, rmse_t = evaluate_model(X_test, y_test, tuned_model)
            console.print(
                f"Tuned R²={round(r2_t, 3)} (was {round(r2, 3)}), "
                f"MAE={round(mae_t, 2)} (was {round(mae, 2)})"
            )
            if r2_t > r2:
                console.print("Keeping tuned model.", style="bold green")
                model = tuned_model
                r2 = r2_t
            else:
                console.print("Tuned model did not improve; keeping original.", style="italic")
            print("")

    # Background sample for SHAP's LinearExplainer + conformal calibration
    # for likely-range intervals are persisted alongside the model.
    background_n = min(100, len(X_train))
    background_sample = (
        X_train.sample(n=background_n, random_state=0).reset_index(drop=True)
        if background_n > 0
        else None
    )
    calibration = compute_calibration_residuals(model, X_test, y_test)

    model_metadata = Serialize_Trained_Model(
        algorithm,
        selected_columns,
        target_column,
        model,
        r2,
        log_target=log_target,
        background_sample=background_sample,
        calibration=calibration,
    )

    # Feature importances. Extracted once, reused for both the PDF report
    # and the JSON payload. Don't fail the train CLI if extraction misbehaves
    # — surface it as missing instead.
    try:
        importances = extract_feature_importances(model, X_train)
        importances_block = [
            {"feature": k, "importance": float(v)} for k, v in importances
        ]
    except Exception:
        importances = None
        importances_block = []

    saved_model_path: str | None = None
    if interactive:
        if questionary.confirm("Save the model?").ask():
            model_name = questionary.text("Enter model name (e.g., my_model.model)").ask()
            SaveModel(model_metadata, model_name)
            saved_model_path = model_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"models/{algorithm}_{target}_{timestamp}.model"
        SaveModel(model_metadata, model_name)
        saved_model_path = model_name
        if not json_mode:
            console.print(f"Model saved to {model_name}", style="bold green")

    report_path: str | None = None
    if report:
        y_pred_test = model.predict(X_test)
        plotting.plot_results_pdf(
            y_test,
            y_pred_test,
            report,
            target_name=target,
            model_name=algorithm,
            n_train=len(X_train),
            training_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            feature_importances=importances,
        )
        report_path = report
        if not json_mode:
            console.print(f"PDF report written to {report}", style="bold green")

    if not json_mode:
        print("")
        return

    # ---- JSON path ----
    response.update({
        "input": {
            "csv_path": str(input),
            "rows_loaded": rows_loaded,
            "rows_dropped_target_na": rows_dropped_target_na,
            "rows_used": int(data.shape[0]),
            "rows_after_outlier_handling": int(data.shape[0]),
            "target": target,
        },
        "time_column": time_column,
        "outliers": {
            "target": _outlier_summary_block(outlier_summary),
            "features": _feature_outlier_summary_block(feature_outlier_summary),
        },
        "low_info_features": {
            "no_information": list(low_vars),
            "high_variation": list(high_vars),
        },
        "features_selected": list(selected_columns),
        "algorithm_selected": algorithm,
        "log_target": bool(log_target),
        "log_target_decision": log_target_decision,
        "cv_scores": {k: float(v) for k, v in scores.items()} if scores else {},
        "feature_importances": importances_block,
        "evaluation": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "residual_bias_p_value": float(p_value),
            "quality_label": quality_label,
        },
        "model_path": saved_model_path,
        "report_path": report_path,
    })
    emit(response)


if __name__ == "__main__":
    train()
