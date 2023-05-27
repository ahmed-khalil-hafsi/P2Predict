import csv
import pandas as pd
from rich.console import Console

console = Console()

def check_csv_sanity(file):
    try:
        # Check if the file is empty
        if pd.read_csv(file).empty:
            console.print("Aborted: CSV file is empty", style='red')
            raise SystemExit

        with open(file, 'r') as csv_file:
            dialect = csv.Sniffer().sniff(csv_file.read(1024))
            csv_file.seek(0)
            reader = csv.reader(csv_file, dialect)
            header = next(reader)  # Read the header row

            # Check if the header contains any empty columns
            if any(cell == '' for cell in header):
                empty_columns = [i+1 for i, cell in enumerate(header) if cell == '']
                console.print(f"Aborted: CSV file contains empty column(s) at position(s): {empty_columns}", style='red')
                raise SystemExit

            # Check if any cells are empty or contain NA values
            for row_num, row in enumerate(reader, start=2):
                empty_cells = [i+1 for i, cell in enumerate(row) if cell == '']
                na_cells = [i+1 for i, cell in enumerate(row) if pd.isna(cell)]
                if empty_cells or na_cells:
                    error_msg = f"Aborted: CSV file contains empty cells or NA values in row {row_num}"
                    if empty_cells:
                        error_msg += f", empty cells at position(s): {empty_cells}"
                    if na_cells:
                        error_msg += f", NA values at position(s): {na_cells}"
                    console.print(error_msg, style='red')
                    raise SystemExit

            # Check if cells mix categorical and numerical data
            df = pd.read_csv(file)
            for col_num, col in enumerate(df.columns, start=1):
                if df[col].dtype != object and pd.api.types.is_categorical_dtype(df[col]):
                    console.print(f"Aborted: Column '{col}' contains a mix of categorical and numerical data", style='red')
                    raise SystemExit

            

    except FileNotFoundError:
        console.print(f"Aborted: File '{file}' not found", style='red')
        raise SystemExit
    except csv.Error:
        console.print(f"Aborted: Invalid CSV format in file '{file}'", style='red')
        raise SystemExit

