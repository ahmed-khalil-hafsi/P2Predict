from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TREE_FAMILY = {"random_forest", "xgboost"}
LINEAR_FAMILY = {"ridge", "lasso"}

# XGBoost handles NaN natively (it learns a default split direction), so we
# pass NaNs straight through for it. Every other estimator we support rejects
# NaN, so their preprocessors impute first.
_NAN_NATIVE = {"xgboost"}


def build_preprocessor(numerical_cols, categorical_cols, model_family="tree"):
    # Trees don't need scaling and prefer compact integer-coded categoricals.
    # Linear models need scaled numerics and one-hot encoded categoricals.
    #
    # NaN handling is per-family so auto mode (which compares all three
    # algorithms on the same NaN-containing data) works end to end:
    #   - xgboost: NaNs pass through untouched (native support).
    #   - random_forest: impute (sklearn forests reject NaN).
    #   - ridge/lasso: impute (linear models reject NaN).
    impute = model_family not in _NAN_NATIVE

    if model_family in TREE_FAMILY or model_family == "tree":
        if impute:
            numerical_transformer = SimpleImputer(strategy="median")
        else:
            numerical_transformer = "passthrough"
        # OrdinalEncoder propagates NaN by default (encoded_missing_value is
        # np.nan), so for the imputing families we fill the category first.
        ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        if impute:
            categorical_transformer = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", ordinal),
                ]
            )
        else:
            categorical_transformer = ordinal
    elif model_family in LINEAR_FAMILY or model_family == "linear":
        numerical_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
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
