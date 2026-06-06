from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TREE_FAMILY = {"random_forest", "xgboost"}
LINEAR_FAMILY = {"ridge", "lasso"}


def build_preprocessor(numerical_cols, categorical_cols, model_family="tree"):
    # Trees don't need scaling and prefer compact integer-coded categoricals.
    # Linear models need scaled numerics and one-hot encoded categoricals.
    if model_family in TREE_FAMILY or model_family == "tree":
        numerical_transformer = "passthrough"
        categorical_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    elif model_family in LINEAR_FAMILY or model_family == "linear":
        numerical_transformer = StandardScaler()
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, list(numerical_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
        ]
    )


def model_family_for(algorithm):
    if algorithm in TREE_FAMILY:
        return algorithm
    if algorithm in LINEAR_FAMILY:
        return algorithm
    raise ValueError(f"Unknown algorithm: {algorithm}")
