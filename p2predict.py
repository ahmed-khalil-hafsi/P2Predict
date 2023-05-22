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
from modules.ui_console import create_tree

# Model serialization
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def load_model(model_file):
    model_metadata = joblib.load(model_file)
    return model_metadata

def predict(model,features):
    y = model.predict(features)
    return y

@click.command()
@click.option('--model', type=click.Path(exists=True))
@click.option('--features_inline')
@click.option('--features_csv')
def main(model,features_inline,features_csv):
    console = Console()

    if not model:
        model = questionary.path('Enter model file path (.model file)').ask()

    loaded_model_metadata = load_model(model)
    trained_pipeline = loaded_model_metadata['model']

    console.print(f"Model > { Pretty(loaded_model_metadata['model_name']) } loaded.", style="bold blue")
    console.print(f"Model features: {loaded_model_metadata['features']}", style="bold blue")
    console.print(f"Target feature: {loaded_model_metadata['target_feature']}", style="bold blue")

    # Get the encoders for the categorical features
    preprocessor = trained_pipeline.named_steps['preprocessor']
    categorical_transformer = preprocessor.named_transformers_['cat']
    onehot_encoder = categorical_transformer.named_steps['onehot']

    # Get the trained features
    trained_categories = onehot_encoder.categories_
    transformers = preprocessor.transformers_
    
    for name, transformer, feature_columns in transformers:
        if isinstance(transformer, Pipeline) and isinstance(transformer.named_steps['onehot'], OneHotEncoder):
            feature_columns = feature_columns.tolist()

    all_categories = {}
    
    for i, feature in enumerate(feature_columns):
        all_categories[feature] = trained_categories[i].tolist()

    create_tree(all_categories,"Loaded Features: ")

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

    console.print(Panel(Pretty(features_df),title="Model Prediction"))
    
    return y

if __name__ == '__main__':
    main()