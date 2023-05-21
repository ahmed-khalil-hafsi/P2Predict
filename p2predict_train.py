

# Machine learning libs 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from p2predict_feature_selection import get_most_predictable_features


# Plotting Module
import plotting
import webbrowser

# Model serialization
import joblib

#UI
import art
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint
from rich.panel import Panel
from rich.pretty import Pretty
import click
import questionary
from ui_console import print_dataframe


def load_data(file):
    return pd.read_csv(file)

def select_features(data, columns):
    return data[columns]

def get_column_statistics(data,feature_columns):
    stats = {}
    for i in feature_columns:
        skewness = data[i].skew()
        kurtosis = data[i].kurt()
        stats[i] = {'skewness':skewness,'kurtosis': kurtosis}
    return stats

def prepare_data(data,selected_columns,target_column):
    # Separate the features and the target variable
    X = data[selected_columns]
    y = data[target_column]

    # Split the data into training (80%) and test sets (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Identify categorical and numerical columns - for now this is automated. TODO: Let user select which columns are categorical and which are numerical
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    column_stats = get_column_statistics(X,numerical_cols)
    console.print(Pretty(column_stats))

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
    table.add_column("Feature", overflow="fold", width=50)  # Adjust the width as necessary
    table.add_column("Importance (%)", justify="right")

    for i in range(len(feature_importances)):
        table.add_row(feature_names[i], str(round(feature_importances[i] * 100, 2)) + "%")

    console.print(table)

def output_features(data):
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
@click.option('--input', type=click.Path(exists=True), default=None, help='Dataset used for the training. This must be a CSV file.')
@click.option('--target',help='This is the column name that refers to the price in your dataset.')
@click.option('--algorithm')
@click.option('--silent', is_flag=True, default=False, help='Run in silent mode without interactive prompts.')
def main(input, target, algorithm, silent):
    
    
    console.print(art.text2art("P2Predict"), style="blue")  # print ASCII Art
    
    if not silent:
        if not input:
            input = questionary.path('Enter CSV file path').ask()
        if not target:
            target = questionary.text('Enter target column').ask()
        if not algorithm:
            algorithm = questionary.select(
                'Choose a regression algorithm:',
                choices=['ridge', 'xgboost', 'random_forest']
            ).ask()

    # Get input CSV file
    file = input
    data = load_data(file)

    # Analyze columns using a random forest estimator to determine relative importance of features 
    copy_data = data
    best_columns = get_most_predictable_features(copy_data,target)
    console.print("Best features detected for prediction: ")
    print_dataframe(best_columns)

    #Select columns to use
    selected_columns =  questionary.checkbox(
                'Which features do you want to include? ',
                choices= data.columns.tolist()
            ).ask()
    #Prompt.ask('Which columns do you want to include? (This should include also the feature to be predicted:) ').split(',')
    target_column = target

    # Prepare data for training. Split X and Y variables into a set for training and a set for testing.
    X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = prepare_data(data,selected_columns,target_column)

    # Start model training
    console.print("Training the model, this may take a few minutes...", style="blue")
    model = train_model(X_train,y_train,numerical_cols,categorical_cols,algorithm)

    #feature_importances = compute_feature_importance(data,target_column,model)
    #plot_importances(feature_importances, data.drop(target_column, axis=1).columns)

    # Calculate model accuracy
    mae,r2 = evaluate_model(X_test,y_test,model)
    console.print(f'R^2 Score: {round(r2,ndigits=2)}', style='bold blue')
    console.print(f'Mean Absolute Error: {round(mae,ndigits=2)}', style='bold blue')

    # Plot result graphs (silent mode does not produce these)
    if not silent:
        export_pdf = questionary.confirm('Do you want to generate the model quality report?').ask()
        if export_pdf:
            file_name = Prompt.ask('Enter PDF name: (.pdf) ')
            y_prediction = model.predict(X_test)
            plotting.plot_results_pdf(y_test,y_prediction,file_name)

    # Model export
    if not silent:
        save_b = questionary.confirm('Do you want to save the model?').ask()
        if save_b:
            model_name = questionary.text('Enter model name: (.model) ').ask()
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
