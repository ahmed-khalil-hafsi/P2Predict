import click
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


@click.command()
@click.option("-m", "--model", type=click.Path(exists=True),
              help="Path to the trained model file (.model)")
@click.option("-p", "--predict_using",
              help='Feature values, e.g. "weight:100,color:red"')
@click.option("-i", "--predict_file", type=click.Path(exists=True),
              help="CSV file containing feature values for batch prediction")
def main(model, predict_using, predict_file):
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

    features_dict = {}
    if predict_using:
        features_dict = dict(item.split(":") for item in predict_using.split(","))
        features_df = _coerce_features(pd.DataFrame([features_dict]), feature_types)
        y = trained.predict(features_df)
        features_df["prediction"] = y
        console.print(Panel(Pretty(features_df), title="Prediction"))

    elif predict_file:
        features_df = pd.read_csv(predict_file)
        features_df = _coerce_features(features_df, feature_types)
        y = trained.predict(features_df)
        features_df[loaded["target_feature"]] = y
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

    return y


if __name__ == "__main__":
    main()
