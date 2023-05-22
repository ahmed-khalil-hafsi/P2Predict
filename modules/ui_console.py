from rich.table import Table
from rich.console import Console
from rich.tree import Tree
from rich import print

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

# Create a tree in the console
def create_tree(input,tree_name):
    # Create a new Tree
    tree = Tree(tree_name, style="bold blue")

    
    for feature, categories in input.items():
        
        feature_branch = tree.add(feature, style="bold green")
        
        for category in categories:
            feature_branch.add(category, style="yellow")

    
    print(tree)