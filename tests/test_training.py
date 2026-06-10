import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from p2predict.prepare_data import prepare_data
from p2predict.training import (
    ALGORITHMS,
    _log_space_r2,
    _scoring_for,
    _tune,
    auto_train,
    build_pipeline,
    extract_feature_importances,
    log_r2_scorer,
    should_log_target,
    start_training,
)


def _split(df, target="Price", time_column=None):
    features = [c for c in df.columns if c not in {target, time_column}]
    return prepare_data(df, features, target, time_column=time_column)


def test_should_log_target_detects_skew(synthetic_parts_skewed):
    assert should_log_target(synthetic_parts_skewed["Price"]) is True


def test_should_log_target_negative_values_returns_false():
    assert should_log_target([-1, 1, 2, 3]) is False


def test_should_log_target_uniform_returns_false():
    # No skew → no transform.
    assert should_log_target(np.linspace(1, 100, 200)) is False


def test_should_log_target_empty_returns_false():
    assert should_log_target([]) is False


def test_start_training_ridge_fits_and_predicts(synthetic_parts):
    X_train, X_test, y_train, y_test, num, cat = _split(synthetic_parts)
    model, importances, _ = start_training(
        X_train, y_train, num, cat, algorithm="ridge", tune=False
    )
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    assert importances and importances[0][1] >= importances[-1][1]


def test_start_training_random_forest_fits_and_predicts(synthetic_parts):
    X_train, X_test, y_train, y_test, num, cat = _split(synthetic_parts)
    model, importances, _ = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    # Weight is the strongest signal in synthetic data — it should rank top.
    assert importances[0][0] == "Weight"


def test_start_training_log_target_when_skewed(synthetic_parts_skewed):
    X_train, _, y_train, _, num, cat = _split(synthetic_parts_skewed)
    _, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    assert log_target is True


def test_auto_train_picks_best_and_returns_scores(tiny_parts):
    X_train, _, y_train, _, num, cat = _split(tiny_parts)
    model, algo, scores, log_target = auto_train(
        X_train, y_train, num, cat, budget="fast"
    )
    assert algo in ALGORITHMS
    assert set(scores.keys()) == set(ALGORITHMS)
    assert model is not None
    assert isinstance(log_target, bool)


def test_auto_train_time_aware_runs(synthetic_parts_with_date):
    X_train, X_test, y_train, y_test, num, cat = _split(
        synthetic_parts_with_date, time_column="Date"
    )
    # Time column must be excluded from the feature set used for training.
    assert "Date" not in X_train.columns
    # Chronological split → test indices come after train indices.
    assert X_test.index.min() > X_train.index.max()

    model, algo, scores, _ = auto_train(
        X_train, y_train, num, cat, budget="fast", time_aware=True
    )
    assert algo in ALGORITHMS


def test_extract_feature_importances_groups_high_cardinality_correctly(synthetic_parts):
    # 'Weight' and 'weight_extra' both contain the substring 'Weight'/'weight' —
    # the importance grouper must not collapse them.
    df = synthetic_parts.copy()
    df["weight_extra"] = df["Weight"] * 0.1
    X_train, _, y_train, _, num, cat = _split(df)
    model, _, _ = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    importances = extract_feature_importances(model, X_train)
    names = [name for name, _ in importances]
    assert "Weight" in names
    assert "weight_extra" in names
    assert len(names) == len(set(names))


def test_pipeline_handles_unseen_categories_at_predict_time(synthetic_parts):
    X_train, _, y_train, _, num, cat = _split(synthetic_parts)
    model, _, _ = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    new_row = pd.DataFrame([{
        "Weight": 10.0,
        "Region": "MARS",      # unseen
        "Supplier": "Z",       # unseen
        "Size": "Standard",
    }])
    pred = model.predict(new_row)
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])


def test_tree_prices_sparse_premium_category_via_target_encoding():
    """Regression for the categorical-encoding fix.

    A high-cardinality nominal feature with a *sparse premium* category is
    the case ordinal integer codes break: an alphabetical code puts the
    premium category next to a commodity neighbour, and the tree's threshold
    split lumps them into one leaf — so the premium part is priced like a
    commodity one (the 'XGBoost prices a Tesla like a Toyota' failure).
    Target encoding orders categories by mean price, so the premium category
    must come out clearly above the commodity baseline.
    """
    rng = np.random.default_rng(0)
    n = 3000
    # 'brand_t' is the sparse premium brand; 'brand_s' sorts right next to it
    # alphabetically (the ordinal-adjacency trap) but is a commodity brand.
    brands = np.array(["brand_a", "brand_s", "brand_t", "brand_u", "brand_z"])
    weights = np.array([0.34, 0.34, 0.01, 0.30, 0.01])  # brand_t is ~1% of rows
    brand = rng.choice(brands, n, p=weights)
    size = rng.uniform(1, 10, n)
    premium = np.where(brand == "brand_t", 5.0, 1.0)  # 5x brand premium
    price = premium * (1.0 + 0.1 * size) * np.exp(rng.normal(0, 0.1, n))
    df = pd.DataFrame({"brand": brand, "size": size, "price": price})

    X_train, _, y_train, _, num, cat = _split(df, target="price")
    model, _, _ = start_training(
        X_train, y_train, num, cat, algorithm="xgboost", tune=False,
        log_target=True,
    )
    base = {"size": 5.0}
    commodity = float(model.predict(pd.DataFrame([{**base, "brand": "brand_s"}]))[0])
    premium_pred = float(model.predict(pd.DataFrame([{**base, "brand": "brand_t"}]))[0])
    # The true premium is 5x; ordinal encoding collapses it toward ~1x. Assert
    # the model recovers a clear premium (>2.5x) for the sparse brand.
    assert premium_pred > 2.5 * commodity, (
        f"sparse premium brand priced at {premium_pred:.2f} vs commodity "
        f"{commodity:.2f} (ratio {premium_pred/commodity:.2f}; expected > 2.5)"
    )


def _synthetic_regression_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(1, 50, n)
    region = rng.choice(["EU", "CN", "SG", "US"], n)
    size = rng.choice(["Small", "Standard", "Large"], n)
    base = 0.08 * weight + np.where(region == "EU", 0.5, 0.0)
    price = np.clip(base + rng.normal(0, 0.1, n), 0.05, None)
    return pd.DataFrame(
        {"Weight": weight, "Region": region, "Size": size, "Price": price}
    )


def test_tune_decides_on_full_training_set_not_resource_floor():
    # Regression for the HalvingRandomSearchCV resource-floor bug: with the
    # default min_resources='smallest' the final (winner-deciding) rung used
    # only ~90 samples regardless of dataset size. 'exhaust' must push the
    # last rung to the full training size so selection is meaningful.
    df = _synthetic_regression_frame(n=2000)
    X_train, _, y_train, _, num, cat = _split(df)
    pipeline = build_pipeline("ridge", num, cat, log_target=False)
    _tune(pipeline, X_train, y_train, "ridge", budget="fast", log_target=False)
    # Re-run the search directly to inspect the resource schedule.
    search = HalvingRandomSearchCV(
        build_pipeline("ridge", num, cat, log_target=False),
        param_distributions={"model__alpha": [0.1, 1.0, 10.0]},
        n_candidates=6,
        cv=3,
        min_resources="exhaust",
        scoring="r2",
        random_state=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    # The largest rung must use (essentially) the full training set, not the
    # 10/30/90-row floor that min_resources='smallest' schedules. Halving's
    # integer division can shave a few rows off the top rung, hence >= 95%.
    assert max(search.n_resources_) >= 0.95 * len(X_train)
    assert min(search.n_resources_) > 100


def test_scoring_for_uses_log_scorer_only_under_log_target():
    assert _scoring_for(True) is log_r2_scorer
    assert _scoring_for(False) == "r2"


def test_log_space_r2_rewards_model_good_in_log_space():
    # A heavily skewed multiplicative target: most parts are cheap, a few are
    # very expensive. A 'log-good' model tracks the order of magnitude across
    # the whole range; a 'big-only' model nails the few large values but is
    # useless on the bulk. In raw R² the big-only model can look competitive;
    # in log space the log-good model must rank clearly higher.
    rng = np.random.default_rng(0)
    y_true = np.exp(rng.normal(0, 2.0, 500))  # log-normal, strongly skewed

    log_good = y_true * np.exp(rng.normal(0, 0.2, 500))  # small log error
    big_only = np.full_like(y_true, np.median(y_true))
    big_only[y_true > np.quantile(y_true, 0.95)] = y_true[
        y_true > np.quantile(y_true, 0.95)
    ]

    assert _log_space_r2(y_true, log_good) > _log_space_r2(y_true, big_only)


def test_log_space_scorer_handles_nonpositive_predictions():
    # The scorer must not blow up if an estimator emits a non-positive
    # prediction during CV; clipping keeps log() finite.
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.0, -1.0, 3.0, 4.0])
    score = _log_space_r2(y_true, y_pred)
    assert np.isfinite(score)


def test_build_pipeline_log_target_wrap_inverts_correctly(synthetic_parts_skewed):
    X_train, _, y_train, _, num, cat = _split(synthetic_parts_skewed)
    model = build_pipeline("random_forest", num, cat, log_target=True)
    model.fit(X_train, y_train)
    # Predictions should be in original price scale, not log scale.
    preds = model.predict(X_train.head(5))
    assert (preds > 0).all()
    assert preds.max() > 1  # sanity: skewed price ** 2 means values aren't all tiny
