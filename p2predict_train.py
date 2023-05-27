

# Math, Machine learning libs 
import datetime
import random
import pandas as pd
import sklearn
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

# P2Predict
from modules.p2predict_feature_selection import get_most_predictable_features
from modules.hyper_param_opt import hyper_parameter_tuning
from modules.input_checks import check_csv_sanity

# Plotting Module
from modules import plotting
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
from modules.ui_console import print_dataframe

console = Console()

def load_data(file):

    check_csv_sanity(file)

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

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols

def start_training(X_train,y_train,numerical_cols, categorical_cols, algorithm):

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

    # Get model weights
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

    console.print("-> Training finished.",style='blue')
    for feature, importance in sorted_feature_importances:
        console.print(f"Feature: {feature}, Model Weight: {round(importance,ndigits=4)}")

    return my_pipeline



def calculate_feature_importance(X,y,model):
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

# TODO add option to name saved model after generation
# Add a hyper parameter tuning step after model is trained

@click.command()
@click.option('-i','--input', type=click.Path(exists=True), default=None, help='Dataset used for training. This must be a CSV file.')
@click.option('-t','--target',help='Feature name to be predicted. Example: If you are trying to predict a price, this should be the name of your price column')
@click.option('--expert', type=bool, is_flag=True, help="Toggle Expert Mode.", default=False)
@click.option('--algorithm', help="This is the training algorithm to be used.")
@click.option('-v', '--verbose', is_flag=True, default=True)
@click.option('--training_features', help="List of training features to be used to train the model. The list must be the headers seperate by a ':'. Example: --training_features weight:Size ")
def train(input, target, expert, algorithm, verbose,training_features):
    
    
    console.print(art.text2art("P2Predict"), style="blue")  # print ASCII Art

    
    if verbose:
        if not input:
            input = questionary.path('Enter CSV file path').ask()
            if not input:
                console.print("Aborted: You must provide an argument for the input file", style='red')
                raise SystemExit
    else:
        if not input:
            console.print("Aborted: In silent mode, you must provide an argument for the input file",style='red')
            raise SystemExit
        if not target:
            console.print("Aborted: in silent mode, you must provide an argument for the target feature",style='red')
            raise SystemExit
        
    # Expert mode needs the algorithm and the features to use    
    if expert:
        if verbose:
            if not algorithm:
                algorithm = questionary.select(
                    'Choose a regression algorithm:',
                    choices=['ridge', 'xgboost', 'random_forest']
                ).ask()
        else:   
            if not algorithm:
                console.print("Aborted: in silent expert mode, you must pre-select the training algorithm",style='red')
                raise SystemExit
            if not training_features:
                console.print("Aborted: in silent expert mode, you must provide an argument for the training features",style='red')
                raise SystemExit
    
    
        

    # Get input CSV file
    file = input
    data = load_data(file)

    # Check input file
    # 

    if not target:
        target = questionary.select('Enter target column',choices=data.columns.tolist()).ask()

    if expert:
        if not training_features:

            # Analyze columns using a random forest estimator to determine relative importance of features 
            copy_data = data
            best_columns = get_most_predictable_features(copy_data,target)
            console.print("Best features detected for prediction: ")
            print_dataframe(best_columns)

            #Select columns to use
        
            options_list = data.columns.tolist()
            options_list.remove(target) # Don't show Price as an option

            selected_columns =  questionary.checkbox(
                        'Which features do you want to include? ',
                        choices= options_list

                    ).ask()
        else:
            selected_columns = training_features.split(',')
    else: # easy mode
        pass
    
    target_column = target

    

    # Prepare data for training. Split X and Y variables into a set for training and a set for testing.
    X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = prepare_data(data,selected_columns,target_column)

   
    # plotting.plot_histograms(data)


    console.print("Feature characterization... ")
    print_feature_stats(data[numerical_cols])

    # Start model training
    console.print("Training the model, this may take a few minutes...", style='bold blue')
    model = start_training(X_train,y_train,numerical_cols,categorical_cols,algorithm)

    # Calculate model accuracy
    mae,r2 = evaluate_model(X_test,y_test,model)
    console.print("Key Performance Metrics: ")
    console.print(f'R^2 Score: {round(r2,ndigits=2)}', style='bold blue')
    console.print(f'Mean Absolute Error: {round(mae,ndigits=2)}', style='bold blue')

    # Plot result graphs (silent mode does not produce these)
    if not verbose:
        export_pdf = questionary.confirm('Do you want to generate the model quality report?').ask()
        if export_pdf:
            file_name = Prompt.ask('Enter PDF name: (.pdf) ')
            y_prediction = model.predict(X_test)
            plotting.plot_results_pdf(y_test,y_prediction,file_name)

    model_metadata = {
    'model': model,  
    'features': selected_columns,
    'target_feature': target_column,
    'model_name': algorithm,
    'R2': str(r2),
    'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'scikit_learn_version': sklearn.__version__,
    'p2predict_version': 'v0.1beta'
    }

    # Model export
    if not verbose:
        save_b = questionary.confirm('Do you want to save the model?').ask()
        if save_b:
            model_name = questionary.text('Enter model name: (.model) ').ask()
            # Save the model as a pickle file
            joblib.dump(model_metadata, model_name)
    else:
        random_int = random.randint(1, 39)
        model_name = f"models/{algorithm}_{target}_{random_int}.model"
        joblib.dump(model_metadata,model_name)
    
    


def print_feature_stats(data):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature")
    table.add_column("Min")
    table.add_column("Max")
    table.add_column("Mean")
    table.add_column("Median")
    table.add_column("Standard Deviation")
    table.add_column("Skewness")
    table.add_column("Kurtosis")

    for col in data.columns:
        min_val = data[col].min()
        max_val = data[col].max()
        mean_val = round(data[col].mean(),ndigits=4)
        median_val = round(data[col].median(),ndigits=4)
        std_val = round(data[col].std(),ndigits=4)
        skewness = round(data[col].skew(),ndigits=4)
        curt = round(data[col].kurt(),ndigits=4)

        table.add_row(col, str(min_val), str(max_val), str(mean_val), str(median_val), str(std_val), str(skewness), str(curt))

    console.print(table)


if __name__ == '__main__':
    train()
