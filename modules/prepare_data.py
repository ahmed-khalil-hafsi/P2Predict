from sklearn.model_selection import train_test_split


def prepare_data(data, selected_columns, target_column):
    X = data[selected_columns]
    y = data[target_column]
    numerical_cols, categorical_cols = Get_Column_Types(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
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
