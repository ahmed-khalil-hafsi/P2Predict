from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

TREE_FAMILY = {"random_forest", "xgboost"}
LINEAR_FAMILY = {"ridge", "lasso"}

# XGBoost handles NaN natively (it learns a default split direction), so we
# pass NaNs straight through for it. Every other estimator we support rejects
# NaN, so their preprocessors impute first.
_NAN_NATIVE = {"xgboost"}

# Target cross-fitting fold count. The encoder learns each category's mean
# target on out-of-fold rows (so the training encoding can't leak the row's
# own label). 5 folds, seeded for reproducibility — the same property the HPO
# fixes in v0.9.x restored elsewhere.
_TARGET_ENCODER_FOLDS = 5
_TARGET_ENCODER_SEED = 0


class _AdaptiveTargetEncoder(TargetEncoder):
    """TargetEncoder whose internal cross-fitting fold count shrinks to the
    sample size.

    Plain ``TargetEncoder`` uses a 5-fold split for the leakage-free training
    encoding and raises ``n_splits > n_samples`` when handed fewer than 5
    rows. That happens routinely: ``HalvingRandomSearchCV`` trains early rungs
    on tiny sub-samples, and real procurement datasets can be a few dozen
    parts. We clamp the fold count to the data at fit time (and skip
    cross-fitting entirely below 2 rows, where there is nothing to hold out),
    so the encoder is robust on small data instead of crashing model
    selection. ``transform`` is unaffected — it always uses the full-data
    category means computed by ``fit``.

    ``cv`` is set as a plain int (not a CV splitter): scikit-learn 1.5–1.8
    constrain ``TargetEncoder.cv`` to an int, and the fold shuffle is seeded
    via the encoder's own ``random_state`` for cross-version reproducibility.
    """

    def fit_transform(self, X, y=None, **fit_params):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        if n < 2:
            # Nothing to cross-fit; fall back to the plain fit+transform.
            return self.fit(X, y).transform(X)
        self.cv = min(_TARGET_ENCODER_FOLDS, n)
        return super().fit_transform(X, y, **fit_params)


def _target_encoder():
    """Smoothed, cross-fitted target encoder for tree-model categoricals.

    Replaces the previous OrdinalEncoder. Ordinal codes are *arbitrary*
    (alphabetical) integers, so a tree's threshold split on the code groups
    alphabetically-adjacent categories together — which destroys the signal
    for high-cardinality *nominal* features. The classic failure: in a
    used-vehicle model "tesla" gets the integer next to "toyota", so XGBoost
    prices a premium EV like a commodity sedan. Target encoding instead maps
    each category to its (smoothed, out-of-fold) mean target, so the code
    *orders by price* and a single tree split cleanly separates premium from
    commodity. ``smooth="auto"`` shrinks sparse categories toward the global
    mean (an empirical-Bayes prior), which is why it also *helps* small
    datasets rather than overfitting them.

    A most-frequent imputer runs first: ``TargetEncoder`` raises on a NaN
    category it never saw at fit time (it cannot match a float-nan against
    string categories), so we fill missing categoricals before encoding —
    the same guard the linear/one-hot path uses. Categories that are unseen
    at predict time are still handled natively (encoded to the target mean).
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("target", _AdaptiveTargetEncoder(
                target_type="continuous", smooth="auto",
                random_state=_TARGET_ENCODER_SEED)),
        ]
    )


def build_preprocessor(numerical_cols, categorical_cols, model_family="tree"):
    # Trees get target-encoded categoricals (one price-ordered numeric column
    # per feature); linear models get scaled numerics and one-hot categoricals.
    #
    # NaN handling is per-family so auto mode (which compares all three
    # algorithms on the same NaN-containing data) works end to end:
    #   - xgboost: numeric NaNs pass through untouched (native support).
    #   - random_forest: numeric NaNs imputed (sklearn forests reject NaN).
    #   - ridge/lasso: numeric NaNs imputed (linear models reject NaN).
    # Categorical NaNs are absorbed by the encoder in every family (TargetEncoder
    # encodes a missing/unseen category to the target mean; OneHotEncoder is
    # fed a most-frequent imputer for the linear path).
    impute = model_family not in _NAN_NATIVE

    if model_family in TREE_FAMILY or model_family == "tree":
        if impute:
            numerical_transformer = SimpleImputer(strategy="median")
        else:
            numerical_transformer = "passthrough"
        categorical_transformer = _target_encoder()
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
