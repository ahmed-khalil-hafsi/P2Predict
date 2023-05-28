
# Math, Machine learning libs 

import random


# P2Predict Modules
from modules.p2predict_feature_selection import get_most_predictable_features
from modules.hyper_param_opt import hyper_parameter_tuning
from modules.input_checks import check_csv_sanity
from modules.trained_model_io import SaveModel, Serialize_Trained_Model, load_csv_file
from modules import plotting
from modules.ui_console import print_dataframe
from modules.cmdline_io import print_feature_importances, print_feature_stats
from modules.model_evals import evaluate_model
from modules.prepare_data import prepare_data
from modules.training import start_training
from modules.training import auto_train

#UI
from rich.console import Console
from rich.prompt import Prompt

import click
import questionary


console = Console()

# TODO add option to name saved model after generation
# TODO Add various levels of verbosity

@click.command()
@click.option('-i','--input', type=click.Path(exists=True), default=None, help='Dataset used for training. This must be a CSV file.')
@click.option('-t','--target',help='Feature name to be predicted. Example: If you are trying to predict a price, this should be the name of your price column')
@click.option('--expert', is_flag=True, help="Toggle Expert Mode.", default=None)
@click.option('--algorithm', help="This is the training algorithm to be used.")
@click.option('-v', '--verbose', is_flag=True, default=None)
@click.option('-ic', '--interactive', is_flag=True, default=True)
@click.option('--training_features', help="List of training features to be used to train the model. The list must be the headers seperate by a ':'. Example: --training_features weight:Size ")
def train(input, target, expert, algorithm, verbose,interactive,training_features):
    
    print("")
    console.print(" ____   ____   ____                   _  _        _   ",style='blue')
    console.print("|  _ \\ |___ \\ |  _ \\  _ __   ___   __| |(_)  ___ | |_ ",style='blue')
    console.print("| |_) |  __) || |_) || '__| / _ \\ / _` || | / __|| __|",style='blue')
    console.print("|  __/  / __/ |  __/ | |   |  __/| (_| || || (__ | |_ ",style='blue')
    console.print("|_|    |_____||_|    |_|    \\___| \\__,_||_| \\___| \\__|",style='blue')
    print("")

    if expert:
        console.print(f"Welcome to P2Predict! Expert mode activated.", style='bold blue')
    else:
        console.print(f"Welcome to P2Predict! Auto mode activated.", style='bold blue')
 
    if interactive:
        if not input:
            input = questionary.path('Enter CSV file path').ask()
            if not input:
                console.print("Aborted: You must provide an argument for the input file", style='red')
                raise SystemExit
    else:
        if not input:
            console.print("NON-INTERACTIVE MODE | Aborted: You must provide an argument for the input file. Alternatively, switch to interactive mode by using the -ic flag.",style='red')
            raise SystemExit
        if not target:
            console.print("NON-INTERACTIVE MODE | Aborted: You must provide an argument for the target feature. Alternatively, switch to interactive mode by using the -ic flag.",style='red')
            raise SystemExit
        
    # Expert mode needs the algorithm and the features to use    
    if expert:
        if interactive:
            if not algorithm:
                algorithm = questionary.select(
                    'EXPERT MODE > Choose an ML algorithm:',
                    choices=['ridge', 'xgboost', 'random_forest']
                ).ask()
                if not algorithm:
                    console.print("Aborted: You must select an algorithm.",style='red')
                    raise SystemExit
        else:   
            if not algorithm:
                console.print("NON-INTERACTIVE MODE | Aborted: You must pre-select the training algorithm. Alternatively, switch to interactive mode by using the -ic flag.",style='red')
                raise SystemExit
            if not training_features:
                console.print("NON-INTERACTIVE MODE | Aborted: You must provide an argument for the training features. Alternatively, switch to interactive mode by using the -ic flag.",style='red')
                raise SystemExit
                
    # Load CSV File
    file = input
    data = load_csv_file(file)
    
    if not target:
        target = questionary.select('Enter target column',choices=data.columns.tolist()).ask()
        if not target:
            console.print("Aborted: A Target Feature is required for the training.", style='red')
            raise SystemExit
        
    
    if not training_features:
        if expert:
            # Analyze columns using a random forest estimator to determine relative importance of features 
            copy_data = data
            best_columns = get_most_predictable_features(copy_data,target)
            console.print("EXPERT MODE > Best features detected for prediction: ",style='blue')
            print_dataframe(best_columns)

            #Select columns to use
        
            options_list = data.columns.tolist()
            options_list.remove(target) # Don't show Price as an option

            selected_columns =  questionary.checkbox(
                        'EXPERT MODE > Which features do you want to include? ',
                        choices= options_list

                    ).ask()
        else:
            selected_columns = get_most_predictable_features(data,target,output_only_headers=True)[0:2] # We will take top 2 features for the auto mode
            console.print(f"Detected best features for training: {selected_columns.to_list()}",style="bold blue")
            
    else:
        selected_columns = training_features.split(',') # TODO must check selected features if they exist in the CSV file
    
    target_column = target



    # Prepare data for training. Split X and Y variables into a set for training and a set for testing.
    X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = prepare_data(data,selected_columns,target_column)

    if expert:
        plot_hist = questionary.confirm("Do you want to plot the histograms of the selected features?").ask()
        if plot_hist:
            plotting.plot_histograms(data[selected_columns])

    if expert:
        console.print("EXPERT MODE > Numerical Feature characterization... ",style='blue')
        print_feature_stats(data[numerical_cols])

    # Start model training
    console.print("Training the model, this may take a few minutes...", style='bold blue')
    if expert:
        model, feature_importances = start_training(X_train,y_train,numerical_cols,categorical_cols,algorithm)
        print_feature_importances(feature_importances)
    else:
        # auto mode
        model = auto_train(X_train,y_train,numerical_cols,categorical_cols)

    
        mae,r2 = evaluate_model(X_test,y_test,model)
        if expert: # Calculate model accuracy - this is only available in expert mode  
            console.print("EXPERT MODE > Key Performance Metrics: ",style='bold green')
            console.print(f'R^2 Score: {round(r2,ndigits=2)}', style='bold blue')
            console.print(f'Mean Absolute Error: {round(mae,ndigits=2)}', style='bold blue')
        else:
            console.print("Key Performance Metric: ",style='bold green')
            if r2>0.8:
                console.print(f"Excellent model trained. The model's score is {round(r2,ndigits=2)}", style='bold green')
            elif r2>0.6:
                console.print(f"Average model trained. The model's score is {round(r2,ndigits=2)}", style='bold yellow')
            else:
                console.print(f"Poor model result. The model's score is {round(r2,ndigits=2)}. Please do not use use this model as is. Use expert mode to create a better model.", style="bold red")
                



    # Plot result graphs and create output pdf. PDF is only created in interactive mode
    if interactive:
        export_pdf = questionary.confirm('Do you want to generate the model quality report?').ask()
        if export_pdf:
            file_name = Prompt.ask('Enter PDF name: (.pdf) ')
            y_prediction = model.predict(X_test)
            plotting.plot_results_pdf(y_test,y_prediction,file_name)

    if expert:
        hyper_params = questionary.confirm('EXPERT MODE > The model can try to auto-tune the hyper paramers to search for a better model. Do you want to continue? This can take significant time.').ask()
        if hyper_params:
            hyper_parameter_tuning(X_train=X_train,y_train=y_train,numerical_cols=numerical_cols,categorical_cols=categorical_cols)

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


    
    return model_metadata       

if __name__ == '__main__':
    train()
