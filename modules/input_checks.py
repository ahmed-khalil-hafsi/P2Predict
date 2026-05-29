import pandas as pd
from rich.console import Console

console = Console()


def check_csv_sanity(file):
    """Load and sanity-check a CSV. Returns the cleaned DataFrame.

    Aborts on empty files, malformed CSV, or missing files. Drops rows
    with NA values (with a warning) rather than refusing to proceed.
    """
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        console.print(f"Aborted: File '{file}' not found", style="red")
        raise SystemExit(1)
    except pd.errors.ParserError as e:
        console.print(f"Aborted: Invalid CSV format in '{file}': {e}", style="red")
        raise SystemExit(1)
    except pd.errors.EmptyDataError:
        console.print("Aborted: CSV file is empty", style="red")
        raise SystemExit(1)

    if df.empty:
        console.print("Aborted: CSV file is empty", style="red")
        raise SystemExit(1)

    empty_header_positions = [
        i + 1 for i, col in enumerate(df.columns)
        if isinstance(col, str) and col.strip() == ""
    ]
    if empty_header_positions:
        console.print(
            f"Aborted: CSV file contains empty column(s) at position(s): {empty_header_positions}",
            style="red",
        )
        raise SystemExit(1)

    na_counts = df.isna().sum()
    columns_with_na = na_counts[na_counts > 0]
    if not columns_with_na.empty:
        details = ", ".join(f"{col} ({n})" for col, n in columns_with_na.items())
        console.print(
            f"Warning: CSV contains missing values in: {details}. "
            "Rows with NA will be dropped.",
            style="yellow",
        )
        df = df.dropna()
        if df.empty:
            console.print(
                "Aborted: dropping rows with missing values leaves no data.",
                style="red",
            )
            raise SystemExit(1)

    return df
