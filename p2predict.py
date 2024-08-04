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
    console.print(f"Loaded features: {loaded_model_metadata['features']}", style="bold white")
    print("")
    console.print(f"↳ Target feature: ['{loaded_model_metadata['target_feature']}']", style="bold blue")
    print("")

    # Get the encoders for the categorical features
    preprocessor = trained_pipeline.named_steps['preprocessor']
    if 'cat' in preprocessor.named_transformers_:
        categorical_transformer = preprocessor.named_transformers_['cat']
        
        if 'onehot' in categorical_transformer.named_steps:
            onehot_encoder = categorical_transformer.named_steps['onehot']
            if hasattr(onehot_encoder, 'categories_'):
                
                trained_categories = onehot_encoder.categories_

                transformers = preprocessor.transformers_
        
                for name, transformer, feature_columns in transformers:
                    if isinstance(transformer, Pipeline) and isinstance(transformer.named_steps['onehot'], OneHotEncoder):
                        feature_columns = feature_columns.tolist()

                all_categories = {}
        
                for i, feature in enumerate(feature_columns):
                    all_categories[feature] = trained_categories[i].tolist()

                print("")
                modules.ui_console.create_tree(all_categories,"Categorical Features: ")

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
            predict_using = questionary.text(
                    f'{feature}: ',
                ).ask()
            if not predict_using:
                console.print("Aborted: Please enter a value for this feature.", style='bold red')
                raise SystemExit
            features_dict[feature] = predict_using
        
        features_df = pd.DataFrame([features_dict])

        y = predict(trained_pipeline,features_df)

        features_df['prediction'] = y

        console.print(Panel(Pretty(features_df),title="Prediction"))
    
    return y


if __name__ == '__main__':
    main()