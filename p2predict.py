#Machine learning
import pandas as pd

#UI
import art
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint
from rich.panel import Panel
from rich.console import Group
from rich.pretty import Pretty
import click

# Model serialization
import joblib

def load_model(model_file):
    model = joblib.load(model_file)
    return model

def predict(model,features):
    y = model.predict(features)
    return y

@click.command()
@click.option('--model', type=click.Path(exists=True), prompt='Enter model file path')
@click.option('--features', prompt='Enter features to be predicted (format: key:value,key:value,...)')
def main(model,features):
    console = Console()

    loaded_model = load_model(model)

    # Parse the features string into a dictionary
    features_dict = dict(item.split(":") for item in features.split(","))

    # Convert the dictionary into a DataFrame
    features_df = pd.DataFrame([features_dict])

    console.print(f'Model > {model} loaded.')
    
    console.print(Panel(Pretty(loaded_model),title='Loaded Model'))

    y = predict(loaded_model,features_df)

    features_df['prediction'] = y

    console.print(Panel(Pretty(features_df),title="Data Output"))
    
    return y

if __name__ == '__main__':
    main()