# Math, Machine learning libs 
from halo import Halo


spinner = Halo(text='Loading P2Predict', spinner='pong')
spinner.start()

import random
import pandas as pd

# P2Predict Modules
from modules.p2predict_feature_selection import get_most_predictable_features
from modules.p2predict_feature_selection import get_most_predictable_features_RFE
from modules.hpo_training import hyper_parameter_tuning
from modules.input_checks import check_csv_sanity
from modules.trained_model_io import SaveModel, Serialize_Trained_Model, load_csv_file
from modules import plotting
from modules.ui_console import print_dataframe
from modules.cmdline_io import print_feature_weights, print_feature_stats, print_logo
from modules.model_evals import evaluate_model
from modules.prepare_data import prepare_data
from modules.training import start_training
from modules.training import auto_train
from modules.p2predict_feature_selection import find_no_variation_features
from modules.p2predict_feature_selection import find_high_variation_features

#UI
from rich.console import Console
from rich.prompt import Prompt

import click
import questionary


console = Console()

spinner.stop()

# TODO add option to name saved model after generation
# TODO Add various levels of verbosity

@click.command()
@click.option('-i', '--input', type=click.Path(exists=True), default=None, 
              help='Path to the CSV file containing the training dataset.')
@click.option('-t', '--target', 
              help='Name of the feature to be predicted (e.g., "Price" for price prediction).')
@click.option('-x', '--expert', is_flag=True, default=None,
              help='Enable Expert Mode for more control over the training process.')
@click.option('-a', '--algorithm', 
              help='Specify the machine learning algorithm to use for training (e.g., "ridge", "xgboost", "random_forest").')
@click.option('-v', '--verbose', is_flag=True, default=None,
              help='Enable verbose output for detailed information during training.')
@click.option('-c', '--interactive', is_flag=True, default=None,
              help='Enable interactive mode for guided input of training parameters.')
@click.option('-tf', '--training_features', 
              help='Comma-separated list of feature names to use for training (e.g., "Weight,Size,Color").')
def train(input, target, expert, algorithm, verbose,interactive,training_features):
    
    print("")
    print_logo()
    print("")

    if expert:
        console.print(f"Welcome to P2Predict! 'Expert mode' is active.", style='bold blue')
    else:
        console.print(f"Welcome to P2Predict! 'Auto mode' is active.", style='bold blue')
 
    if interactive:
        if not input:
            input = questionary.path('Enter CSV file path').ask()
            if not input:
                console.print("Aborted: You must provide an argument for the input file", style='bold red')
                raise SystemExit
    else:
        if not input:
            console.print("Aborted: You must provide an argument for the input file. Alternatively, switch to interactive mode by using the -c flag.",style='bold red')
            raise SystemExit
        if not target:
            console.print("Aborted: You must provide an argument for the target feature. Alternatively, switch to interactive mode by using the -c flag.",style='bold red')
            raise SystemExit
        
    # Expert mode needs the algorithm and the features to use    
    if expert:
        if interactive:
            if not algorithm:
                algorithm = questionary.select(
                    'Please choose an ML algorithm:',
                    choices=['ridge', 'xgboost', 'random_forest']
                ).ask()
                if not algorithm:
                    console.print("Aborted: You must select a training algorithm.",style='bold red')
                    raise SystemExit
        else:   
            if not algorithm:
                console.print("Aborted: You must pre-select the training algorithm. Alternatively, switch to interactive mode by using the -c flag.",style='bold red')
                raise SystemExit
            if not training_features:
                console.print("Aborted: You must provide an argument for the training features. Alternatively, switch to interactive mode by using the -c flag.",style='bold red')
                raise SystemExit
                
    # Load CSV File
    file = input
    data = load_csv_file(file)
    print("")
    console.print(f"Training file '{file}' imported into P2Predict > {data.shape[0]} rows  x {data.shape[1]} columns loaded.")
    print("")
    if not target:
        target = questionary.select('Enter target column',choices=data.columns.tolist()).ask()
        if not target:
            console.print("Aborted: A Target Feature is required for the training.", style='bold red')
            raise SystemExit
        
    
    # Find low information features
    high_vars = find_high_variation_features(data)
    low_vars = find_no_variation_features(data)

    print("")
    console.print(f"We found the following low information features: ")
    console.print(f"No information content: {low_vars}")
    console.print(f"Low information content: {high_vars}")
    print("")

    remove_no_information_features = questionary.checkbox('Which features you want to remove? ', choices=low_vars + high_vars).ask()
    if remove_no_information_features:
        data = data.drop(remove_no_information_features, axis=1)
        
    
    if not training_features:
        if expert:
            
            copy_data = data
            best_features_ranked = get_most_predictable_features(copy_data,target)
            # best_features_RFE = get_most_predictable_features_RFE(copy_data,target)
            
            console.print("Best features detected for prediction: ",style='bold white')
            print("")
            print_dataframe(best_features_ranked)

            #Select columns to use
        
            options_list = data.columns.tolist()
            options_list.remove(target) # Don't show Price as an option

            selected_columns =  questionary.checkbox(
                        'Select the features for training: ',
                        choices= options_list

                    ).ask()
            if not selected_columns:
                console.print("Aborted: You must select training features.", style='bold red')
                raise SystemExit
        else:
            selected_columns = get_most_predictable_features(data,target,output_only_headers=True)[0:2] # We will take top 2 features for the auto mode
            console.print(f"Detected best features for training: {selected_columns.to_list()}",style="bold blue")
            print("")
            
    else:
        selected_columns = training_features.split(',') # TODO must check selected features if they exist in the CSV file
    
    target_column = target

    # Prepare data for training. Split X and Y variables into a set for training and a set for testing.
    X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = prepare_data(data,selected_columns,target_column)


    if expert and interactive:
        plot_hist = questionary.confirm("Do you want to plot the histograms of the selected features?").ask()
        print("")
        if plot_hist:
            plotting.plot_histograms(data[selected_columns])

    if expert:
        console.print("Numerical Feature analysis: ",style='bold white')
        print("")
        print_feature_stats(data[numerical_cols])
        print("")

    # Start model training
   
    spinner = Halo(text='Training the model, this may take a few minutes...', spinner='pong')
    spinner.start()
    if expert:
       
        model, feature_weights = start_training(X_train,y_train,numerical_cols,categorical_cols,algorithm)
        spinner.stop()
        
        
        print_feature_weights(feature_weights)
        print("")
    else:
        # auto mode
        model = auto_train(X_train,y_train,numerical_cols,categorical_cols)
        spinner.stop()

    spinner.succeed('Training finished.')
    print("")

    mae, r2, p_value = evaluate_model(X_test, y_test, model)
    if expert:
        console.print("Model Key Performance Metrics: ", style='bold white')
        console.print(f'Model R² Score: {round(r2, ndigits=2)}', style='white')
        console.print(f'Mean Absolute Error: {round(mae, ndigits=2)}', style='white')
        console.print(f'P-value: {round(p_value, ndigits=4)}', style='white')
        print("")
    else:
        console.print("Model Performance Summary:", style='bold white')
        
        # Calculate a composite score
        r2_score = min(r2, 1.0)  # Cap R² at 1.0
        p_value_score = 1 - min(p_value, 1.0)  # Invert p-value, cap at 1.0
        mae_score = 1 / (1 + mae)  # Normalize MAE, higher is better
        
        composite_score = (0.6 * r2_score + 0.2 * p_value_score + 0.2 * mae_score) * 100
        
        if composite_score > 80:
            model_quality = "Excellent"
            style = 'bold green'
        elif composite_score > 60:
            model_quality = "Good"
            style = 'bold yellow'
        else:
            model_quality = "Needs Improvement"
            style = 'bold red'
        
        console.print(f"Model Quality: {model_quality}", style=style)
        console.print(f"Overall Score: {round(composite_score, 1)}%")
        console.print(f"R² Score: {round(r2 * 100, 1)}%")
        console.print(f"Average Prediction Error: {round(mae, 2)}")
        
        if p_value < 0.05:
            console.print("The model's predictions are statistically significant.", style='italic')
        else:
            console.print("The model's predictions are not statistically significant.", style='italic bold')
            console.print("This suggests that the model's performance may be due to chance.", style='italic')
            console.print("Consider collecting more data or refining the model.", style='italic')
        
        if model_quality == "Good" and p_value >= 0.05:
            console.print("\nNote: While the overall metrics look good, the lack of statistical significance suggests caution in interpreting the results.", style="bold yellow")
        
        if model_quality == "Needs Improvement" or p_value >= 0.05:
            console.print("\nRecommendation: Consider using expert mode to improve the model and address statistical significance.", style="bold")
        
        print("")
    
    print("")


    # Plot result graphs and create output pdf. PDF is only created in interactive mode
    if expert and interactive:
        export_pdf = questionary.confirm('Do you want to generate the model quality report?').ask()
        if export_pdf:
            file_name = Prompt.ask('Enter PDF name: (.pdf) ')
            # For plotting we want to plot the performance for all the data points
            X_plotting = pd.concat([X_train,X_test])
            y_prediction = model.predict(X_plotting)
            plotting.plot_results_pdf(data[target],y_prediction,file_name)
            print("")

    if expert and interactive:
        hyper_params = questionary.confirm('The model can try to auto-tune the hyper paramers to search for a better model. Do you want to continue? This can take significant time.').ask()
        if hyper_params:
            spinner = Halo('Searching ...',spinner='pong')
            spinner.start()
            hyper_parameter_tuning(X_train=X_train,y_train=y_train,numerical_cols=numerical_cols,categorical_cols=categorical_cols)
            spinner.stop()
            print("")

    model_metadata = Serialize_Trained_Model(algorithm, selected_columns, target_column, model, r2)

    # Model export
    if interactive:
        save_b = questionary.confirm('Do you want to save the model?').ask()
        if save_b:
            model_name = questionary.text('Enter model name: (.model) ').ask()
            # Save the model to disk
            SaveModel(model_metadata, model_name)
    else:
        random_int = random.randint(1, 39)
        model_name = f"models/{algorithm}_{target}_{random_int}.model"
        SaveModel(model_metadata, model_name)
    print("")



    

if __name__ == '__main__':
    train()