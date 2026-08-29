"""Tests for the what-if comparison module.

Headline assertions are axiomatic:

  Total delta decomposition (the SHAP local-accuracy axiom applied to
  the difference of two predictions): for both non-log and log-target
  models, the per-feature deltas must sum to the inner-model delta
  exactly (modulo floating point). Without this property, the
  "drivers of the change" attribution doesn't mean anything.

  Log-target multiplicativity: the product of per-feature
  multiplicative factors equals cf_price / base_price exactly.

Without these tests the rendering looks fine but the numbers in a
design-review meeting could be wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2predict.prepare_data import prepare_data
from p2predict.training import start_training
from p2predict.intervals import compute_calibration_residuals
from p2predict.whatif import (
    INTERACTION_MATERIALITY_THRESHOLD,
    WhatIfResult,
    assess_reliability,
    compute_whatif,
    interaction_is_material,
    parse_changes,
)
from p2predict.training import extract_feature_importances
from p2predict.model_utils import whatif_to_dict


# ---------------------------------------------------------------------------
# Synthetic-data fixtures large enough that SHAP has signal to attribute.
# ---------------------------------------------------------------------------


def _generate(n=600, seed=0, skewed=False):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(1, 50, n)
    region = rng.choice(["EU", "CN", "SG", "US"], n)
    size = rng.choice(["Small", "Standard", "Large"], n)
    base = (
        0.08 * weight
        + np.where(region == "EU", 0.5, 0.0)
        + np.where(size == "Large", 0.7, 0.0)
    )
    noise = rng.normal(0, 0.15, n)
    price = np.clip(base + noise, 0.05, None)
    if skewed:
        price = np.exp(price)
    return pd.DataFrame({
        "Weight": weight, "Region": region, "Size": size, "Price": price,
    })


def _train(df, algorithm, log_target_in_data=False):
    features = [c for c in df.columns if c != "Price"]
    X_train, X_test, y_train, y_test, num, cat = prepare_data(df, features, "Price")
    model, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm=algorithm, tune=False
    )
    if log_target_in_data:
        assert log_target
    calibration = compute_calibration_residuals(model, X_test, y_test)
    return model, X_train, X_test, calibration, log_target


@pytest.fixture
def rf_setup():
    df = _generate()
    model, X_train, X_test, calibration, log_target = _train(df, "random_forest")
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "calibration": calibration,
        "log_target": log_target,
        "feature_types": {
            "Weight": "Numerical", "Region": "Categorical", "Size": "Categorical"
        },
    }


@pytest.fixture
def ridge_setup():
    df = _generate()
    model, X_train, X_test, calibration, log_target = _train(df, "ridge")
    return {
        "model": model,
        "X_train": X_train,
        "background": X_train.sample(100, random_state=0),
        "X_test": X_test,
        "calibration": calibration,
        "log_target": log_target,
        "feature_types": {
            "Weight": "Numerical", "Region": "Categorical", "Size": "Categorical"
        },
    }


@pytest.fixture
def log_target_setup():
    df = _generate(skewed=True)
    model, X_train, X_test, calibration, log_target = _train(
        df, "random_forest", log_target_in_data=True
    )
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "calibration": calibration,
        "log_target": log_target,
        "feature_types": {
            "Weight": "Numerical", "Region": "Categorical", "Size": "Categorical"
        },
    }


# ---------------------------------------------------------------------------
# parse_changes — the CLI parser.
# ---------------------------------------------------------------------------


def test_parse_changes_single():
    assert parse_changes("Region:EU") == {"Region": "EU"}


def test_parse_changes_multiple():
    assert parse_changes("Region:EU,Supplier:B,Weight:20") == {
        "Region": "EU", "Supplier": "B", "Weight": "20",
    }


def test_parse_changes_strips_whitespace():
    assert parse_changes(" Region : EU , Supplier : B ") == {
        "Region": "EU", "Supplier": "B",
    }


def test_parse_changes_rejects_malformed():
    with pytest.raises(ValueError):
        parse_changes("RegionEU")
    with pytest.raises(ValueError):
        parse_changes("Region:")  # empty value
    with pytest.raises(ValueError):
        parse_changes(":EU")  # empty key


# ---------------------------------------------------------------------------
# Decomposition axiom — the property the feature claims.
# ---------------------------------------------------------------------------


def test_decomposition_sums_to_total_delta_rf(rf_setup):
    """Sum of changed contributions + interaction contribution should
    equal the inner-model delta within floating-point tolerance."""
    base = rf_setup["X_test"].iloc[[0]]
    # Change one categorical feature.
    cf_value = "EU" if base["Region"].iloc[0] != "EU" else "CN"
    result = compute_whatif(
        rf_setup["model"], base, {"Region": cf_value},
        rf_setup["feature_types"],
    )
    decomposed = (
        sum(result.changed_contributions.values())
        + result.interaction_contribution
    )
    inner_delta = result.counterfactual_prediction - result.base_prediction
    assert decomposed == pytest.approx(inner_delta, abs=1e-6, rel=1e-4)
    assert abs(result.decomposition_residual) < 1e-5


def test_decomposition_sums_to_total_delta_ridge(ridge_setup):
    base = ridge_setup["X_test"].iloc[[0]]
    cf_value = "EU" if base["Region"].iloc[0] != "EU" else "CN"
    # For ridge, explainer needs background and we wire it via the same
    # background that lives in the model metadata at predict time. Pass
    # explicitly here.
    result = compute_whatif(
        ridge_setup["model"], base, {"Region": cf_value},
        ridge_setup["feature_types"],
        background_X=ridge_setup["background"],
    )
    decomposed = (
        sum(result.changed_contributions.values())
        + result.interaction_contribution
    )
    inner_delta = result.counterfactual_prediction - result.base_prediction
    assert decomposed == pytest.approx(inner_delta, abs=1e-6, rel=1e-4)


def test_multi_feature_change_decomposition(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    # Change two features at once.
    cf_changes = {
        "Region": "EU" if base["Region"].iloc[0] != "EU" else "CN",
        "Weight": str(float(base["Weight"].iloc[0]) + 5.0),
    }
    result = compute_whatif(
        rf_setup["model"], base, cf_changes, rf_setup["feature_types"]
    )
    decomposed = (
        sum(result.changed_contributions.values())
        + result.interaction_contribution
    )
    inner_delta = result.counterfactual_prediction - result.base_prediction
    assert decomposed == pytest.approx(inner_delta, abs=1e-4, rel=1e-3)
    # Each of the two changed features should appear in the contributions.
    assert set(result.changed_contributions.keys()) == set(cf_changes.keys())


# ---------------------------------------------------------------------------
# Log-target multiplicativity — the equivalent contract in price space.
# ---------------------------------------------------------------------------


def test_log_target_multiplicative_decomposition(log_target_setup):
    """For a log-target model the per-feature multiplicative factors
    times the interaction factor should equal cf_price / base_price."""
    base = log_target_setup["X_test"].iloc[[0]]
    cf_value = "EU" if base["Region"].iloc[0] != "EU" else "CN"
    result = compute_whatif(
        log_target_setup["model"], base, {"Region": cf_value},
        log_target_setup["feature_types"],
    )
    assert result.log_target
    assert result.multiplicative_factor is not None
    assert result.changed_multiplicative_factors is not None
    assert result.interaction_multiplicative_factor is not None

    product = float(
        np.prod(list(result.changed_multiplicative_factors.values()))
        * result.interaction_multiplicative_factor
    )
    assert product == pytest.approx(result.multiplicative_factor, abs=1e-4, rel=1e-4)


def test_log_target_factors_are_strictly_positive(log_target_setup):
    base = log_target_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        log_target_setup["model"], base, {"Region": "EU"},
        log_target_setup["feature_types"],
    )
    for factor in result.changed_multiplicative_factors.values():
        assert factor > 0
    assert result.interaction_multiplicative_factor > 0
    assert result.multiplicative_factor > 0


# ---------------------------------------------------------------------------
# Counterfactual sanity — predict on the actual modified row.
# ---------------------------------------------------------------------------


def test_counterfactual_prediction_matches_independent_predict(rf_setup):
    """compute_whatif must call predict on the *modified* row, not on the
    base row. Verify by independently building the modified row and
    predicting on it."""
    model = rf_setup["model"]
    base = rf_setup["X_test"].iloc[[0]]
    target_region = "EU" if base["Region"].iloc[0] != "EU" else "CN"

    result = compute_whatif(
        model, base, {"Region": target_region},
        rf_setup["feature_types"],
    )

    modified = base.copy()
    modified["Region"] = target_region
    independent_pred = float(model.predict(modified)[0])
    assert result.counterfactual_prediction == pytest.approx(independent_pred)


def test_base_prediction_matches_independent_predict(rf_setup):
    model = rf_setup["model"]
    base = rf_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        model, base, {"Region": "EU"}, rf_setup["feature_types"]
    )
    independent_pred = float(model.predict(base)[0])
    assert result.base_prediction == pytest.approx(independent_pred)


# ---------------------------------------------------------------------------
# Interval composition — both base and cf carry intervals when calibration
# is available.
# ---------------------------------------------------------------------------


def test_intervals_attached_when_calibration_present(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        rf_setup["model"], base, {"Region": "EU"},
        rf_setup["feature_types"],
        calibration=rf_setup["calibration"],
        coverage=0.90,
    )
    assert result.base_interval is not None
    assert result.cf_interval is not None
    assert result.base_interval.low < result.base_interval.high
    assert result.cf_interval.low < result.cf_interval.high


def test_intervals_skipped_when_calibration_missing(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        rf_setup["model"], base, {"Region": "EU"},
        rf_setup["feature_types"],
        calibration=None,
    )
    assert result.base_interval is None
    assert result.cf_interval is None


# ---------------------------------------------------------------------------
# Error paths.
# ---------------------------------------------------------------------------


def test_unknown_feature_raises_helpful(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    with pytest.raises(ValueError, match="not a training feature"):
        compute_whatif(
            rf_setup["model"], base, {"NotAColumn": "X"},
            rf_setup["feature_types"],
        )


def test_non_numeric_value_for_numeric_feature_raises(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    with pytest.raises(ValueError, match="numeric"):
        compute_whatif(
            rf_setup["model"], base, {"Weight": "heavy"},
            rf_setup["feature_types"],
        )


def test_empty_changes_raises(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    with pytest.raises(ValueError):
        compute_whatif(rf_setup["model"], base, {}, rf_setup["feature_types"])


def test_multi_row_base_raises(rf_setup):
    base = rf_setup["X_test"].iloc[:2]  # two rows
    with pytest.raises(ValueError, match="single-row"):
        compute_whatif(
            rf_setup["model"], base, {"Region": "EU"},
            rf_setup["feature_types"],
        )


# ---------------------------------------------------------------------------
# Interaction materiality helper.
# ---------------------------------------------------------------------------


def test_interaction_is_material_threshold():
    big = WhatIfResult(
        base_prediction=10.0, counterfactual_prediction=12.0,
        delta=2.0, delta_pct=20.0, changes={},
        changed_contributions={}, interaction_contribution=0.5,  # 25% of delta
    )
    assert interaction_is_material(big)

    tiny = WhatIfResult(
        base_prediction=10.0, counterfactual_prediction=12.0,
        delta=2.0, delta_pct=20.0, changes={},
        changed_contributions={}, interaction_contribution=0.02,  # 1% of delta
    )
    assert not interaction_is_material(tiny)


# ---------------------------------------------------------------------------
# Direction-reliability verdict — the regression cases are the six single-spec
# what-ifs from the battery-management-IC model (analysis/whatif_reliability_
# flag.md). Contributions are in log space, as the model returns them.
# ---------------------------------------------------------------------------


def _whatif(delta_pct, changed, interaction):
    """Minimal WhatIfResult carrying only what assess_reliability reads."""
    # Sign/magnitude of the target-space delta only has to agree with the log-
    # space direct effect; a unit base price keeps delta ≈ delta_pct/100.
    return WhatIfResult(
        base_prediction=1.0,
        counterfactual_prediction=1.0 + delta_pct / 100.0,
        delta=delta_pct / 100.0,
        delta_pct=delta_pct,
        changes={},
        changed_contributions=changed,
        interaction_contribution=interaction,
    )


def test_reliability_trust_strong_direct_dominated():
    """pins 8->16: strong signal, the change itself drives the move."""
    r = _whatif(44.6, {"package_pins": 0.3175}, 0.0509)
    verdict, reason, say = assess_reliability(r, {"package_pins": "strong"})
    assert verdict == "trust"
    assert reason == "strong_signal"
    assert say


def test_reliability_caution_interaction_dominant():
    """cells 2->4: the interaction term is larger than the change's own."""
    r = _whatif(-8.0, {"max_cells_supported": -0.0389}, -0.0440)
    verdict, reason, _ = assess_reliability(r, {"max_cells_supported": "moderate"})
    assert verdict == "caution"
    assert reason == "interaction_dominant"


def test_reliability_caution_moderate_signal():
    """max temp 85->125: direct-dominated but only a moderate driver."""
    r = _whatif(-11.9, {"op_temp_max_C": -0.1136}, -0.0132)
    verdict, reason, _ = assess_reliability(r, {"op_temp_max_C": "moderate"})
    assert verdict == "caution"
    assert reason == "moderate_signal"


def test_reliability_quote_sign_flip():
    """multi-chem: the changed spec's own effect ADDS, interactions flip the
    headline to 'saves'. Strong signal alone would miss this; the sign-flip
    rule catches it."""
    r = _whatif(-1.1, {"Battery Chemistry": 0.0144}, -0.0249)
    verdict, reason, _ = assess_reliability(r, {"Battery Chemistry": "strong"})
    assert verdict == "quote"
    assert reason == "sign_flip"


def test_reliability_trust_is_the_documented_miss():
    """add SPI to I2C: strong signal, direct-dominated, no sign flip — neither
    flag fires even though the direction is commercially backwards. This is the
    honest limit recorded in the findings note; the guidance-layer sign-check
    is what covers it, not the payload flag."""
    r = _whatif(-7.3, {"Interface": -0.0588}, -0.0174)
    verdict, _, _ = assess_reliability(r, {"Interface": "strong"})
    assert verdict == "trust"


def test_reliability_quote_weak_signal_beats_everything():
    r = _whatif(30.0, {"obscure_spec": 0.2}, 0.0)
    verdict, reason, _ = assess_reliability(r, {"obscure_spec": "weak"})
    assert verdict == "quote"
    assert reason == "weak_signal"


def test_reliability_flat_move_not_flagged_by_interaction():
    """temp floor -25->-40: a genuinely free concession (~0%). The degenerate
    0 >= 0 interaction check must NOT fire; a moderate feature still earns a
    'directional only' caution, but never a spurious sign-flip/interaction one."""
    r = _whatif(0.0, {"op_temp_min_C": 0.0}, 0.0)
    verdict, reason, _ = assess_reliability(r, {"op_temp_min_C": "moderate"})
    assert verdict == "caution"
    assert reason == "moderate_signal"  # not interaction_dominant / sign_flip


def test_reliability_flat_move_strong_is_trust():
    r = _whatif(0.2, {"Interface": 0.0}, 0.0)
    verdict, reason, _ = assess_reliability(r, {"Interface": "strong"})
    assert verdict == "trust"


def test_multiple_changed_features_take_worst_signal():
    r = _whatif(20.0, {"a": 0.1, "b": 0.1}, 0.0)
    verdict, reason, _ = assess_reliability(r, {"a": "strong", "b": "moderate"})
    assert verdict == "caution"
    assert reason == "moderate_signal"


# ---------------------------------------------------------------------------
# Reliability plumbing — populated only when importances are supplied, and it
# reaches the serialized summary.
# ---------------------------------------------------------------------------


def test_reliability_absent_without_importances(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        rf_setup["model"], base, {"Region": "EU"}, rf_setup["feature_types"]
    )
    assert result.reliability is None
    assert "reliability" not in whatif_to_dict(result)["summary"]


def test_reliability_present_with_importances(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    importances = extract_feature_importances(
        rf_setup["model"], rf_setup["X_train"]
    )
    result = compute_whatif(
        rf_setup["model"], base, {"Region": "EU"}, rf_setup["feature_types"],
        feature_importances=importances,
    )
    assert result.reliability in {"trust", "caution", "quote"}
    assert result.reliability_say_to_user
    summary = whatif_to_dict(result)["summary"]
    assert summary["reliability"] == result.reliability
    assert summary["say_to_user"] == result.reliability_say_to_user
