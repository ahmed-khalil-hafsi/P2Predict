from rich.table import Table
from rich.console import Console

def print_dataframe(df):
    console = Console()

    # create a table
    table = Table(show_header=True, header_style="bold magenta")

    # add columns
    for column in df.columns:
        table.add_column(column)

    # add rows
    for _, row in df.iterrows():
        table.add_row(*row.astype(str))

    console.print(table)