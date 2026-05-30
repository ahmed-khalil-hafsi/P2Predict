import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(data, selected_columns, target_column, time_column=None, test_size=0.2):
    """Split into train/test. When time_column is given, the split is
    chronological (last `test_size` of rows after sorting by date is the test
    set) and the time column is excluded from features.
    """
    selected_columns = list(selected_columns)
    if time_column is not None:
        # Avoid leaking the time column into model features and ensure
        # downstream CV is also chronological.
        selected_columns = [c for c in selected_columns if c != time_column]
        data = data.sort_values(time_column).reset_index(drop=True)

    X = data[selected_columns]
    y = data[target_column]
    numerical_cols, categorical_cols = Get_Column_Types(X)

    if time_column is not None:
        n_test = max(1, int(len(data) * test_size))
        split_at = len(data) - n_test
        X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
        y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols


def select_features(data, columns):
    return data[columns]


def Get_Column_Types(X):
    numerical_cols = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns
    categorical_cols = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns
    return numerical_cols, categorical_cols
