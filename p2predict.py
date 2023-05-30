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
from P2Predict.modules.trained_model_io import LoadModel

import modules.ui_console

# Model serialization
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



def predict(model,features):
    y = model.predict(features)
    return y

@click.command()
@click.option('--model', type=click.Path(exists=True))
@click.option('--features_inline')
@click.option('--features_csv')
def main(model,features_inline,features_csv):
    console = Console()
    
    print("")
    console.print(" ____   ____   ____                   _  _        _   ",style='blue')
    console.print("|  _ \\ |___ \\ |  _ \\  _ __   ___   __| |(_)  ___ | |_ ",style='blue')
    console.print("| |_) |  __) || |_) || '__| / _ \\ / _` || | / __|| __|",style='blue')
    console.print("|  __/  / __/ |  __/ | |   |  __/| (_| || || (__ | |_ ",style='blue')
    console.print("|_|    |_____||_|    |_|    \\___| \\__,_||_| \\___| \\__|",style='blue')
    print("")

    if not model:
        model = questionary.path('Enter model file path (.model file)').ask()

    loaded_model_metadata = LoadModel(model)
    trained_pipeline = loaded_model_metadata['model']

    console.print(f"Model > { Pretty(loaded_model_metadata['model_name']) } loaded.", style="bold blue")
    print()
    console.print(f"Model features: {loaded_model_metadata['features']}", style="bold blue")
    print()
    console.print(f"Target feature: {loaded_model_metadata['target_feature']}", style="bold blue")

    # Get the encoders for the categorical features
    preprocessor = trained_pipeline.named_steps['preprocessor']
    categorical_transformer = preprocessor.named_transformers_['cat']
    onehot_encoder = categorical_transformer.named_steps['onehot']

    # Get the trained features
    trained_categories = onehot_encoder.categories_
    transformers = preprocessor.transformers_

    print(transformers)
    
    for name, transformer, feature_columns in transformers:
        if isinstance(transformer, Pipeline) and isinstance(transformer.named_steps['onehot'], OneHotEncoder):
            feature_columns = feature_columns.tolist()

    all_categories = {}
    
    for i, feature in enumerate(feature_columns):
        all_categories[feature] = trained_categories[i].tolist()

    print()
    modules.ui_console.create_tree(all_categories,"Categorical Features: ")

    features_dict = {}
    if features_inline:
        features_dict = dict(item.split(":") for item in features_inline.split(","))
    else:     
        for feature in loaded_model_metadata['features']:
            features_inline = questionary.text(
                    f'{feature}: ',
                ).ask()
            features_dict[feature] = features_inline
        
    features_df = pd.DataFrame([features_dict])

    y = predict(trained_pipeline,features_df)

    features_df['prediction'] = y

    console.print(Panel(Pretty(features_df),title="Prediction"))
    
    return y

if __name__ == '__main__':
    main()