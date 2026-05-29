import datetime

from halo import Halo

spinner = Halo(text="Loading P2Predict", spinner="pong")
spinner.start()

import click
import pandas as pd
import questionary
from rich.console import Console
from rich.prompt import Prompt

from modules import plotting
from modules.cmdline_io import print_feature_stats, print_feature_weights, print_logo
from modules.hpo_training import hyper_parameter_tuning
from modules.model_evals import evaluate_model
from modules.outliers import POLICIES as OUTLIER_POLICIES, apply_outlier_policy
from modules.p2predict_feature_selection import (
    find_high_variation_features,
    find_no_variation_features,
    get_most_predictable_features,
)
from modules.prepare_data import prepare_data
from modules.trained_model_io import SaveModel, Serialize_Trained_Model, load_csv_file
from modules.training import ALGORITHMS, auto_train, start_training
from modules.ui_console import print_dataframe

console = Console()

spinner.stop()


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
@click.option("--tune/--no-tune", default=None,
              help="Expert mode only: run HPO on the chosen algorithm and save the tuned model.")
@click.option("--outliers", type=click.Choice(list(OUTLIER_POLICIES)), default="warn",
              help="How to handle outliers in the target column (Tukey IQR rule). "
                   "'warn' (default) = report only; 'drop' = remove rows; "
                   "'winsorize' = cap values; 'keep' = silent.")
@click.option("--time-column", default=None,
              help="Name of a date/time column. When given, the train/test split and CV "
                   "become chronological (TimeSeriesSplit), which prevents look-ahead bias "
                   "for time-ordered data. The column is excluded from features.")
def train(input, target, expert, algorithm, verbose, interactive, training_features,
          budget, tune, outliers, time_column):

    print("")
    print_logo()
    print("")

    mode_label = "Expert mode" if expert else "Auto mode"
    console.print(f"Welcome to P2Predict! '{mode_label}' is active.", style="bold blue")

    if interactive:
        if not input:
            input = questionary.path("Enter CSV file path").ask()
            if not input:
                console.print("Aborted: You must provide an input file.", style="bold red")
                raise SystemExit
    else:
        if not input:
            console.print(
                "Aborted: You must provide --input. Use -c for interactive mode.",
                style="bold red",
            )
            raise SystemExit
        if not target:
            console.print(
                "Aborted: You must provide --target. Use -c for interactive mode.",
                style="bold red",
            )
            raise SystemExit

    if expert:
        if interactive and not algorithm:
            algorithm = questionary.select(
                "Please choose an ML algorithm:", choices=list(ALGORITHMS)
            ).ask()
            if not algorithm:
                console.print("Aborted: You must select a training algorithm.", style="bold red")
                raise SystemExit
        elif not interactive:
            if not algorithm:
                console.print(
                    "Aborted: You must pre-select --algorithm in expert mode (or use -c).",
                    style="bold red",
                )
                raise SystemExit
            if not training_features:
                console.print(
                    "Aborted: You must provide --training_features in expert mode (or use -c).",
                    style="bold red",
                )
                raise SystemExit

    data = load_csv_file(input)
    print("")
    console.print(
        f"Training file '{input}' imported into P2Predict > "
        f"{data.shape[0]} rows  x {data.shape[1]} columns loaded."
    )
    print("")

    if not target:
        target = questionary.select("Enter target column", choices=data.columns.tolist()).ask()
        if not target:
            console.print("Aborted: A target feature is required.", style="bold red")
            raise SystemExit

    if time_column is not None and time_column not in data.columns:
        console.print(
            f"Aborted: --time-column '{time_column}' not found in CSV.", style="bold red"
        )
        raise SystemExit
    if time_column is not None:
        try:
            data[time_column] = pd.to_datetime(data[time_column])
        except Exception as exc:  # noqa: BLE001 — surface parse failures verbatim
            console.print(
                f"Aborted: could not parse --time-column '{time_column}': {exc}",
                style="bold red",
            )
            raise SystemExit
        console.print(
            f"Time-aware mode: train/test split and CV will be chronological on "
            f"'{time_column}'.",
            style="bold blue",
        )

    data, outlier_summary = apply_outlier_policy(data, target, policy=outliers)
    if outlier_summary["n_outliers"] > 0:
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

    # Exclude the time column from feature analysis — it isn't a training feature.
    feature_data = data.drop(columns=[time_column]) if time_column else data

    high_vars = find_high_variation_features(feature_data)
    low_vars = find_no_variation_features(feature_data)

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
        # Non-interactive: always drop zero-variance features (they can't help).
        data = data.drop(low_vars, axis=1)

    # Refresh after possible drops so downstream feature analysis matches.
    feature_data = data.drop(columns=[time_column]) if time_column else data

    if not training_features:
        if expert:
            best_features_ranked = get_most_predictable_features(feature_data, target)
            console.print("Best features detected for prediction:", style="bold white")
            print("")
            print_dataframe(best_features_ranked)

            options_list = [c for c in feature_data.columns.tolist() if c != target]
            selected_columns = questionary.checkbox(
                "Select the features for training: ", choices=options_list
            ).ask()
            if not selected_columns:
                console.print("Aborted: You must select training features.", style="bold red")
                raise SystemExit
        else:
            ranked = get_most_predictable_features(feature_data, target, output_only_headers=True)
            # In auto mode we let the model selector see more signal than just
            # the top two — drop only the lowest-importance tail.
            selected_columns = ranked.head(max(2, min(len(ranked), 6))).tolist()
            console.print(
                f"Auto-selected features for training: {selected_columns}", style="bold blue"
            )
            print("")
    else:
        requested = [c.strip() for c in training_features.split(",")]
        missing = [c for c in requested if c not in data.columns]
        if missing:
            console.print(
                f"Aborted: requested features not in CSV: {missing}", style="bold red"
            )
            raise SystemExit
        selected_columns = requested

    target_column = target

    if time_column is not None and time_column in selected_columns:
        selected_columns = [c for c in selected_columns if c != time_column]

    X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = prepare_data(
        data, selected_columns, target_column, time_column=time_column
    )
    time_aware = time_column is not None

    if expert and interactive:
        if questionary.confirm("Plot histograms of the selected features?").ask():
            plotting.plot_histograms(data[selected_columns])
        print("")

    if expert:
        console.print("Numerical feature analysis:", style="bold white")
        print("")
        print_feature_stats(data[list(numerical_cols)])
        print("")

    if expert:
        # Decide HPO from flag or (in interactive) prompt.
        if tune is None and interactive:
            tune = questionary.confirm(
                "Run hyperparameter tuning (slower, usually higher accuracy)?"
            ).ask()
        tune = bool(tune)

        spinner = Halo(
            text=f"Training {algorithm} (tune={tune}, budget={budget})...", spinner="pong"
        )
        spinner.start()
        model, feature_weights, log_target = start_training(
            X_train, y_train, numerical_cols, categorical_cols, algorithm,
            budget=budget, tune=tune, time_aware=time_aware,
        )
        spinner.stop()

        print_feature_weights(feature_weights)
        print("")
        if log_target:
            console.print(
                "Note: log-target transform applied (target is positive and skewed).",
                style="italic",
            )
    else:
        spinner = Halo(
            text=f"Auto-mode model selection (budget={budget})...", spinner="pong"
        )
        spinner.start()
        model, algorithm, scores, log_target = auto_train(
            X_train, y_train, numerical_cols, categorical_cols,
            budget=budget, time_aware=time_aware,
        )
        spinner.stop()
        console.print(f"Selected best algorithm: [bold]{algorithm}[/bold]")
        for algo, score in scores.items():
            console.print(f"  {algo}: CV R² = {round(score, 3)}")
        if log_target:
            console.print(
                "Note: log-target transform applied (target is positive and skewed).",
                style="italic",
            )

    spinner.succeed("Training finished.")
    print("")

    mae, r2, p_value, rmse = evaluate_model(X_test, y_test, model)
    if expert:
        console.print("Model Key Performance Metrics:", style="bold white")
        console.print(f"Model R² Score: {round(r2, 2)}")
        console.print(f"Mean Absolute Error: {round(mae, 2)}")
        console.print(f"RMSE: {round(rmse, 2)}")
        console.print(f"Residual bias p-value: {round(p_value, 4)}")
        print("")
    else:
        console.print("Model Performance Summary:", style="bold white")
        r2_score_clamped = min(max(r2, 0.0), 1.0)
        composite = r2_score_clamped * 100

        if composite > 80:
            quality, style = "Excellent", "bold green"
        elif composite > 60:
            quality, style = "Good", "bold yellow"
        else:
            quality, style = "Needs Improvement", "bold red"

        console.print(f"Model Quality: {quality}", style=style)
        console.print(f"R² Score: {round(r2 * 100, 1)}%")
        console.print(f"Mean Absolute Error: {round(mae, 2)}")
        console.print(f"RMSE: {round(rmse, 2)}")

        if p_value < 0.05:
            console.print(
                "Residuals show systematic bias — consider expert mode for tuning.",
                style="italic bold yellow",
            )
        if quality == "Needs Improvement":
            console.print(
                "Recommendation: try expert mode with --tune, or collect more data.",
                style="bold",
            )
        print("")

    if expert and interactive:
        if questionary.confirm("Generate the model quality PDF report?").ask():
            file_name = Prompt.ask("Enter PDF name (e.g., report.pdf)")
            X_plotting = pd.concat([X_train, X_test])
            y_prediction = model.predict(X_plotting)
            plotting.plot_results_pdf(data[target], y_prediction, file_name)
            print("")

    if expert and interactive and not tune:
        # Offer post-hoc HPO if the user skipped --tune up front.
        if questionary.confirm(
            "Run hyperparameter tuning now to try for a better model?"
        ).ask():
            spinner = Halo("Tuning...", spinner="pong")
            spinner.start()
            tuned_model, tuned_score, log_target = hyper_parameter_tuning(
                X_train=X_train,
                y_train=y_train,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                algorithm=algorithm,
                budget=budget,
                time_aware=time_aware,
            )
            spinner.stop()
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

    model_metadata = Serialize_Trained_Model(
        algorithm, selected_columns, target_column, model, r2, log_target=log_target
    )

    if interactive:
        if questionary.confirm("Save the model?").ask():
            model_name = questionary.text("Enter model name (e.g., my_model.model)").ask()
            SaveModel(model_metadata, model_name)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"models/{algorithm}_{target}_{timestamp}.model"
        SaveModel(model_metadata, model_name)
        console.print(f"Model saved to {model_name}", style="bold green")
    print("")


if __name__ == "__main__":
    train()
