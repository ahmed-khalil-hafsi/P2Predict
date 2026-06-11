import numpy as np
from scipy.stats import loguniform, randint, skew
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import HalvingRandomSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from p2predict.preprocessing import build_preprocessor, model_family_for

ALGORITHMS = ("ridge", "random_forest", "xgboost")
LOG_TARGET_SKEW_THRESHOLD = 1.0

# Smallest strictly-positive value used to clip raw-space predictions before
# taking their log in the log-space scorer. Guards against log(0)/log(<0) when
# an estimator emits a non-positive prediction during CV on the log target.
_LOG_SCORE_TINY = 1e-9


def _log_space_r2(y_true, y_pred):
    """R² computed in log space.

    The estimator's ``predict`` returns raw (price-space) values even when the
    target is log-wrapped (``TransformedTargetRegressor`` inverts the log).
    Scoring those in raw space lets a model that only nails the few large
    values win on a heavily-skewed target while being useless on the bulk of
    cheap parts. Comparing ``log(y_true)`` against ``log(y_pred)`` scores the
    candidate in the space the model is actually optimised in, which is what
    selection should reward.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return r2_score(
        np.log(np.clip(y_true, _LOG_SCORE_TINY, None)),
        np.log(np.clip(y_pred, _LOG_SCORE_TINY, None)),
    )


# Greater-is-better scorer wrapping :func:`_log_space_r2`. Reused by both
# ``_tune`` and any other place that selects candidates under the log wrap.
log_r2_scorer = make_scorer(_log_space_r2, greater_is_better=True)


def _scoring_for(log_target):
    """Pick the CV scorer so candidate selection happens in the space the
    model is trained in: log-space R² when the target is log-wrapped, plain
    R² otherwise."""
    return log_r2_scorer if log_target else "r2"


def _make_estimator(algorithm):
    if algorithm == "ridge":
        return Ridge()
    if algorithm == "random_forest":
        return RandomForestRegressor(random_state=0, n_jobs=-1)
    if algorithm == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror", n_jobs=-1, random_state=0, verbosity=0
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _search_space(algorithm, budget):
    if algorithm == "ridge":
        return {"model__alpha": loguniform(1e-3, 1e2)}
    if algorithm == "random_forest":
        n_max = 401 if budget == "fast" else 801
        return {
            "model__n_estimators": randint(100, n_max),
            "model__max_depth": [None, 6, 12, 24],
            "model__min_samples_split": randint(2, 11),
            "model__min_samples_leaf": randint(1, 6),
        }
    if algorithm == "xgboost":
        n_max = 401 if budget == "fast" else 801
        return {
            "model__n_estimators": randint(100, n_max),
            "model__max_depth": randint(3, 9),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _budget_params(budget, time_aware=False):
    if budget == "thorough":
        n_candidates, n_splits = 24, 5
    else:
        n_candidates, n_splits = 10, 3
    cv = TimeSeriesSplit(n_splits=n_splits) if time_aware else n_splits
    return {"n_candidates": n_candidates, "cv": cv}


def _prefix_params(params, log_target):
    if not log_target:
        return params
    return {f"regressor__{k}": v for k, v in params.items()}


def should_log_target(y):
    try:
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        return False
    if y_arr.size == 0 or np.any(y_arr <= 0):
        return False
    return float(skew(y_arr)) > LOG_TARGET_SKEW_THRESHOLD


def resolve_log_target(y, mode="auto"):
    """Decide whether to apply the log-target wrap and explain why.

    Returns ``(log_target: bool, decision: str)`` where ``decision`` is one
    of ``"auto:skew=<value>"``, ``"manual:on"``, or ``"manual:off"`` — the
    string is what consumers (--json output, the case studies) record so the
    choice is traceable. ``manual:on`` requires strictly positive y; raising
    that case is the caller's responsibility (so the CLI can surface a
    friendly --json error).
    """
    if mode == "off":
        return False, "manual:off"
    if mode == "on":
        return True, "manual:on"
    if mode != "auto":
        raise ValueError(f"Unknown log_target mode: {mode!r}")
    try:
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        return False, "auto:skew=nan"
    if y_arr.size == 0 or np.any(y_arr <= 0):
        return False, "auto:skew=nan"
    sk = float(skew(y_arr))
    return sk > LOG_TARGET_SKEW_THRESHOLD, f"auto:skew={sk:.2f}"


def build_pipeline(algorithm, numerical_cols, categorical_cols, log_target=False):
    preprocessor = build_preprocessor(
        numerical_cols, categorical_cols, model_family_for(algorithm)
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", _make_estimator(algorithm))]
    )
    if log_target:
        # log/exp (not log1p/expm1) so SHAP's multiplicative axiom in price
        # space is clean: per-feature factors multiply the price directly,
        # not (1 + price). Safe because should_log_target() only fires when
        # all training targets are strictly positive.
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log, inverse_func=np.exp
        )
    return pipeline


from p2predict.model_utils import inner_pipeline as _inner_pipeline


def _tune(pipeline, X_train, y_train, algorithm, budget, log_target, time_aware=False):
    bp = _budget_params(budget, time_aware=time_aware)
    params = _prefix_params(_search_space(algorithm, budget), log_target)
    search = HalvingRandomSearchCV(
        pipeline,
        param_distributions=params,
        n_candidates=bp["n_candidates"],
        cv=bp["cv"],
        # 'exhaust' makes the final (winner-deciding) rung use the full
        # training set. The default 'smallest' schedules resources from a
        # tiny floor (10 -> 30 -> 90 samples for cv=5 regression) regardless
        # of dataset size, so on a 15k-row dataset the algorithm/HP winner was
        # being chosen on 90 rows — scores were meaningless and the winning
        # algorithm flipped between identical runs.
        min_resources="exhaust",
        scoring=_scoring_for(log_target),
        random_state=0,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_score_


def auto_train(
    X_train, y_train, numerical_cols, categorical_cols, budget="fast",
    time_aware=False, log_target=None,
):
    if log_target is None:
        log_target = should_log_target(y_train)
    best_score = -np.inf
    best_model = None
    best_algorithm = None
    scores = {}
    for algorithm in ALGORITHMS:
        pipeline = build_pipeline(
            algorithm, numerical_cols, categorical_cols, log_target=log_target
        )
        model, score = _tune(
            pipeline, X_train, y_train, algorithm, budget, log_target, time_aware=time_aware
        )
        scores[algorithm] = score
        if score > best_score:
            best_score = score
            best_model = model
            best_algorithm = algorithm
    return best_model, best_algorithm, scores, log_target


def start_training(
    X_train,
    y_train,
    numerical_cols,
    categorical_cols,
    algorithm,
    budget="fast",
    tune=False,
    time_aware=False,
    log_target=None,
):
    if log_target is None:
        log_target = should_log_target(y_train)
    pipeline = build_pipeline(
        algorithm, numerical_cols, categorical_cols, log_target=log_target
    )

    if tune:
        pipeline, _ = _tune(
            pipeline, X_train, y_train, algorithm, budget, log_target,
            time_aware=time_aware,
        )
    else:
        pipeline.fit(X_train, y_train)

    importances = extract_feature_importances(pipeline, X_train)
    return pipeline, importances, log_target


def extract_feature_importances(model, X_train):
    fit_pipeline = _inner_pipeline(model)
    preprocessor = fit_pipeline.named_steps["preprocessor"]
    estimator = fit_pipeline.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_, dtype=float))
    else:
        return []

    try:
        raw_names = preprocessor.get_feature_names_out()
    except Exception:
        raw_names = [f"f{i}" for i in range(len(importances))]

    source_cols = list(X_train.columns)
    by_source = {}
    for name, imp in zip(raw_names, importances):
        rest = name.split("__", 1)[1] if "__" in name else name
        # Match the longest column-name prefix so e.g. "weight_g" doesn't get
        # grouped with "weight".
        match = None
        for col in source_cols:
            if rest == col or rest.startswith(f"{col}_"):
                if match is None or len(col) > len(match):
                    match = col
        source = match if match is not None else rest
        by_source[source] = by_source.get(source, 0.0) + float(imp)

    return sorted(by_source.items(), key=lambda kv: kv[1], reverse=True)
