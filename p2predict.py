#Machine learning
import pandas as pd

#UI
import art
import questionary
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint
from rich.panel import Panel
from rich.console import Group
from rich.pretty import Pretty
import click


import modules.ui_console
from modules.cmdline_io import print_logo
from modules.trained_model_io import LoadModel

# Model serialization
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



def predict(model,features):
    y = model.predict(features)
    return y

@click.command()
@click.option('-m', '--model', type=click.Path(exists=True),
              help='Path to the trained model file (.model)')
@click.option('-p', '--predict_using',
              help='Specify feature values for prediction, e.g., "weight:100,color:red"')
@click.option('-i', '--predict_file', type=click.Path(exists=True),
              help='Path to a CSV file containing feature values for batch prediction')
def main(model, predict_using, predict_file):
    console = Console()
    
    print("")
    print_logo()
    print("")

    if not model:
        model = questionary.path('Enter model file path (.model file)').ask()
        if not model:
            console.print("Aborted: Please enter the path to the trained model.", style='bold red')
            raise SystemExit

    loaded_model_metadata = LoadModel(model)
    trained_pipeline = loaded_model_metadata['model']

    if not trained_pipeline:
        console.print("Aborted: Please selected model is corrupt.", style='bold red')
        raise SystemExit

    console.print(f"'{model}' successfully loaded.", style="bold white")
    print("")
    # Get feature types and categories from the preprocessor
    preprocessor = trained_pipeline.named_steps['preprocessor']
    feature_types = {}
    all_categories = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_types.update({col: 'Numerical' for col in columns})
        elif name == 'cat':
            feature_types.update({col: 'Categorical' for col in columns})
            if 'onehot' in transformer.named_steps:
                onehot_encoder = transformer.named_steps['onehot']
                if hasattr(onehot_encoder, 'categories_'):
                    all_categories = {col: cat.tolist() for col, cat in zip(columns, onehot_encoder.categories_)}

    # Create a table for features and their types
    table = Table(title="Model Features", show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="dim", width=20)
    table.add_column("Type", justify="right")
    for feature, feature_type in feature_types.items():
        table.add_row(feature, feature_type)

    # Display the table
    console.print(table)

    # Print target feature
    console.print(f"\nTarget feature: [bold blue]'{loaded_model_metadata['target_feature']}'[/bold blue]")

    # Print categorical features and their categories
    if all_categories:
        console.print("\nCategorical Features:")
        for feature, categories in all_categories.items():
            console.print(f"[bold]{feature}[/bold]")
            for category in categories:
                console.print(f"  • {category}")
            console.print("")  # Add a blank line between features
    else:
        console.print("No categorical features to display.", style="italic")

    # Add a separator for better readability
    console.print("\n" + "="*50 + "\n")

    features_dict = {}
    if predict_using:
        features_dict = dict(item.split(":") for item in predict_using.split(","))
        features_df = pd.DataFrame([features_dict])
        y = predict(trained_pipeline,features_df)
        features_df['prediction'] = y
        console.print(Panel(Pretty(features_df),title="Prediction"))
        
    elif predict_file:
        # Read the CSV file
        features_df = pd.read_csv(predict_file)
        # Predict using the model
        y = predict(trained_pipeline, features_df)
        # Append the predictions to the DataFrame
        features_df[loaded_model_metadata['target_feature']] = y
        # Write the DataFrame back to the CSV file
        features_df.to_csv(predict_file, index=False)
        console.print(Panel(Pretty(features_df),title="Prediction"))
        
    else:     
        for feature in loaded_model_metadata['features']:
            if feature in all_categories:
                predict_using = questionary.select(
                    f'Select a value for {feature}:',
                    choices=all_categories[feature]
                ).ask()
            else:
                predict_using = questionary.text(
                    f'Enter a numeric value for {feature}:',
                ).ask()
                
            if not predict_using:
                console.print(f"Aborted: Please enter a value for {feature}.", style='bold red')
                raise SystemExit
            
            features_dict[feature] = predict_using
        
        features_df = pd.DataFrame([features_dict])

        y = predict(trained_pipeline,features_df)

        features_df['prediction'] = y

        # Create a styled table for better presentation
        table = Table(title="Prediction Results", show_header=True, header_style="bold magenta")
        
        # Add columns dynamically based on the DataFrame
        for column in features_df.columns:
            table.add_column(column, style="cyan", justify="right")
        
        # Add the row of data
        table.add_row(*[str(val) for val in features_df.iloc[0]])
        
        # Display the table in a panel without a bold border
        console.print(Panel(table, expand=False, border_style="green", padding=(1, 1)))
        
        # Display the prediction separately for emphasis
        prediction_value = features_df['prediction'].iloc[0]
        console.print(f"\n[bold]Predicted {loaded_model_metadata['target_feature']}:[/bold] [yellow]{prediction_value:.2f}[/yellow]")
    
    return y


if __name__ == '__main__':
    main()