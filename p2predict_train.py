

# Machine learning libs 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Plotting Module
import plotting

# Model serialization
import joblib

#UI
import art
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint
from rich.panel import Panel
import click


def load_data(file):
    return pd.read_csv(file)

def select_columns(data, columns):
    return data[columns]

def prepare_data(data,target_column):
    # Separate the features and the target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Identify categorical and numerical columns - for now this is automated. TODO: Let user select which columns are categorical and which are numerical
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols


def train_model(X_train,y_train,numerical_cols, categorical_cols, algorithm):

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the model
    if algorithm == 'ridge':
        model = Ridge(alpha=1.0)
    elif algorithm == 'xgboost':
        model = XGBRegressor(objective ='reg:squarederror')
    elif algorithm == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=0)
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                 ])

    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)

       # Define the model
    if algorithm == 'ridge':
        importance = model.coef_
    elif algorithm == 'xgboost':
        importance = model.feature_importances_
    elif algorithm == 'random_forest':
        importance = model.feature_importances_
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}') 
    
    
    feature_names = X_train.columns.tolist()
    feature_importances = zip(feature_names, importance)
    sorted_feature_importances = sorted(feature_importances, key = lambda x: abs(x[1]), reverse=True)

    for feature, importance in sorted_feature_importances:
        console.print(f"Feature: {feature}, Importance: {importance}")

    

    #result = permutation_importance(my_pipeline, X, y, n_repeats=10, random_state=0)
    #importance_normalized = result.importances_mean / sum(result.importances_mean)
    return my_pipeline

def compute_feature_importance(X,y,model):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    importance_normalized = result.importances_mean / sum(result.importances_mean)
    return importance_normalized

def evaluate_model(X_test,y_test,model):
    # Compute predictions on the test dataset   
    predictions = model.predict(X_test)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae,r2



console = Console()

def plot_importances(feature_importances, feature_names):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature")
    table.add_column("Importance (%)")

    for i in range(len(feature_importances)):
        table.add_row(feature_names[i], str(round(feature_importances[i] * 100, 2)) + "%")

    console.print(table)

def plot_input_data_types(data):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature")
    table.add_column("Type")

    for col, dtype in data.dtypes.items():
        if dtype == 'object':
            dtype = 'text'
        elif dtype == 'int64':
            dtype = 'numerical: integer'
        elif dtype == 'float64':
            dtype = 'numerical: float'
        table.add_row(col, dtype)
    console.print(table)

@click.command()
@click.option('--input', type=click.Path(exists=True), prompt='Enter CSV file path')
@click.option('--target', prompt='Enter target column')
@click.option('--algorithm', prompt='Enter regression algorithm (ridge, xgboost, random_forest)')
def main(input, target, algorithm):
    
    console.print(art.text2art("P2Predict"), style="blue")  # print ASCII Art
    #file = Prompt.ask('Enter CSV file path: ')
    file = input
    data = load_data(file)

    console.print('Columns: ')
    plot_input_data_types(data)

    selected_columns = Prompt.ask('Which features you want to use for training?').split(',')
    target_column = target #Prompt.ask('Enter target column: ')
    data = select_columns(data, selected_columns)

    

    X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = prepare_data(data,target_column)

    #algorithm = Prompt.ask('Enter regression algorithm (ridge, xgboost, random_forest): ')
    console.print("Training the model, this may take a few minutes...", style="blue")

    model = train_model(X_train,y_train,numerical_cols,categorical_cols,algorithm)

    #feature_importances = compute_feature_importance(data,target_column,model)
    #plot_importances(feature_importances, data.drop(target_column, axis=1).columns)

    mae,r2 = evaluate_model(X_test,y_test,model)

    console.print(f'R^2 Score: {r2}', style='bold blue')
    console.print(f'Mean Absolute Error: {mae}', style='bold blue')


    y_prediction = model.predict(X_test)

    plotting.plot_results_console(y_test,y_prediction)

    save_b = Prompt.ask('Do you want to save the model? (Y/n) ')
    if save_b == 'Y':
        model_name = Prompt.ask('Enter model name: ')
        # Save the model as a pickle file
        joblib.dump(model, model_name)


def check_normalization(data):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature")
    table.add_column("Min")
    table.add_column("Max")
    table.add_column("Mean")
    table.add_column("Standard Deviation")

    for col in data.columns:
        min_val = data[col].min()
        max_val = data[col].max()
        mean_val = data[col].mean()
        std_val = data[col].std()

        table.add_row(col, str(min_val), str(max_val), str(mean_val), str(std_val))

    console.print(table)


if __name__ == '__main__':
    main()
