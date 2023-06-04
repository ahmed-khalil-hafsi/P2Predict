from rich.console import Console
from rich.table import Table
console = Console()

def print_logo():
    console.print(" ____   ____   ____                   _  _        _   ",style='bold blue')
    console.print("|  _ \\ |___ \\ |  _ \\  _ __   ___   __| |(_)  ___ | |_ ",style='bold blue')
    console.print("| |_) |  __) || |_) || '__| / _ \\ / _` || | / __|| __|",style='bold blue')
    console.print("|  __/  / __/ |  __/ | |   |  __/| (_| || || (__ | |_ ",style='bold blue')
    console.print("|_|    |_____||_|    |_|    \\___| \\__,_||_| \\___| \\__|",style='bold blue')

def plot_importances(feature_importances, feature_names):
    table = Table(show_header=True, header_style="bold blue", highlight=True)
    table.add_column("Feature", overflow="fold", width=50)  # Adjust the width as necessary
    table.add_column("Importance (%)", justify="right")

    for i in range(len(feature_importances)):
        table.add_row(feature_names[i], str(round(feature_importances[i] * 100, 2)) + "%")

    console.print(table)

def print_feature_weights(sorted_feature_importances):
    for feature, importance in sorted_feature_importances:
        console.print(f"Feature: {feature}, Model Weight: {round(importance,ndigits=4)}")

def output_features(data):
    table = Table(show_header=True, header_style="bold blue", highlight=True)
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

def print_feature_stats(data):
    console = Console()
    table = Table(show_header=True, header_style="bold blue", highlight=True)
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