

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

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Model serialization
import joblib

#UI
import art
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint
import click


def load_data(file):
    return pd.read_csv(file)

def select_columns(data, columns):
    return data[columns]

def perform_analysis(data, target_column, algorithm):
    # Separate the features and the target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    # Identify categorical and numerical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

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

    #Make predictions
    predictions = my_pipeline.predict(X_test)

    # Compute evaluation metric
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    console.print(f'R^2 Score: {r2}', style='bold blue')
    console.print(f'Mean Absolute Error: {mae}', style='bold blue')
    
    result = permutation_importance(my_pipeline, X, y, n_repeats=10, random_state=0)
    importance_normalized = result.importances_mean / sum(result.importances_mean)
    return importance_normalized, my_pipeline

    return my_pipeline

def plot_results(y_test, y_pred):
    # Scatter plot of predicted vs actual values
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.title('Scatter Plot: Actual vs Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

    # Histogram of residuals
    plt.subplot(1,2,2)
    residuals = y_test - y_pred
    sns.histplot(residuals)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Residuals plot
    plt.figure()
    sns.residplot(x=y_pred, y=residuals, lowess=True, color='g')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')

    # Show the plot
    plt.show()

    # Prediction Error Plot
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Price')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error vs Predicted')

    # Show the plot
    plt.show()

console = Console()

def plot_importances(feature_importances, feature_names):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature")
    table.add_column("Importance (%)")

    for i in range(len(feature_importances)):
        table.add_row(feature_names[i], str(round(feature_importances[i] * 100, 2)) + "%")

    console.print(table)

@click.command()
@click.option('--file', prompt='Enter CSV file path')
@click.option('--target', prompt='Enter target column')
@click.option('--algorithm', prompt='Enter regression algorithm (ridge, xgboost, random_forest)')
def main(file, target, algorithm):
    console.print(art.text2art("P2Predict",font="wizard"), style="bold red")  # print ASCII Art
    #file = Prompt.ask('Enter CSV file path: ')
    data = load_data(file)
    print('Columns:', data.columns)
    selected_columns = Prompt.ask('Enter relevant columns (comma-separated): ').split(',')
    target_column = target #Prompt.ask('Enter target column: ')
    data = select_columns(data, selected_columns)
    #algorithm = Prompt.ask('Enter regression algorithm (ridge, xgboost, random_forest): ')
    feature_importances, model = perform_analysis(data, target_column, algorithm)
    plot_importances(feature_importances, data.drop(target_column, axis=1).columns)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(target_column, axis=1), data[target_column], test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_results(y_test, y_pred)
    save_b = Prompt.ask('Do you want to save the model? (Y/n) ')
    if save_b == 'Y':
        model_name = Prompt.ask('Enter model name: ')
        # Save the model as a pickle file
        joblib.dump(model, model_name)


if __name__ == '__main__':
    main()
