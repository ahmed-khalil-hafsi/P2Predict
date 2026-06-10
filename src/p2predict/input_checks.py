import pandas as pd
from rich.console import Console

# Route warnings to stderr so that `--json` callers get a pure-JSON stdout.
# A module-level Console() on stdout used to print the NA warning *before*
# the JSON document, making `p2predict-train --json` output unparseable.
console = Console(stderr=True)


def check_csv_sanity(file):
    """Load and sanity-check a CSV. Returns the loaded DataFrame unchanged.

    Aborts on empty files, malformed CSV, or missing files. Missing values
    are *reported* (to stderr) but no longer dropped here: dropping rows over
    all columns at load time silently discarded data — including NAs in
    columns that aren't even selected as training features. Row dropping /
    imputation is now decided downstream (in the train CLI), once the target
    and feature columns are known, so only the relevant NAs matter.
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
            "Rows are not dropped at load time — NAs in the target column are "
            "dropped at training, and NAs in feature columns are handled by "
            "the model (XGBoost natively; imputed for random_forest/ridge).",
            style="yellow",
        )

    return df
