"""Tests for the training-domain check — 'was this part answerable at all?'

Rationale and measurements: research/out_of_domain_flag.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2predict import domain


def _domain(numeric=None, categorical=None, source="stored"):
    return {
        "numeric": numeric or {"mass_kg": {"min": 0.2, "max": 4.0},
                               "holes": {"min": 2, "max": 12}},
        "categorical": categorical or {"supplier": ["A", "B", "C"],
                                       "material": ["alu", "steel"]},
        "numeric_source": source,
    }


IN_DOMAIN_PART = {"mass_kg": 1.5, "holes": 6, "supplier": "A", "material": "alu"}


# --- the cases from the finding -------------------------------------------

def test_an_ordinary_part_is_in_domain():
    r = domain.check_part(IN_DOMAIN_PART, _domain())
    assert r["status"] == "in_domain"
    assert r["issues"] == []


def test_an_unseen_supplier_is_caught_and_named():
    """The routine version: 'what should we pay a supplier we haven't used?'
    The encoder prices it as the catalog average, which is a sound default --
    reporting that as the supplier's price is the gap."""
    r = domain.check_part({**IN_DOMAIN_PART, "supplier": "Zeta Werke"}, _domain())
    assert r["status"] == "out_of_domain"
    assert r["issues"][0]["kind"] == "unseen_category"
    assert "Zeta Werke" in r["issues"][0]["detail"]
    assert "catalog average" in r["issues"][0]["detail"]


def test_a_wildly_extrapolated_number_is_caught_with_its_scale():
    r = domain.check_part({**IN_DOMAIN_PART, "mass_kg": 900}, _domain())
    assert r["status"] == "out_of_domain"
    assert "225x" in r["issues"][0]["detail"]


def test_a_value_below_the_observed_range_is_caught():
    r = domain.check_part({**IN_DOMAIN_PART, "mass_kg": 0.001}, _domain())
    assert r["status"] == "out_of_domain"
    assert "below the smallest" in r["issues"][0]["detail"]


def test_the_impossible_part_reports_every_offending_spec():
    r = domain.check_part(
        {"mass_kg": 900, "holes": 4000, "supplier": "Zeta", "material": "Unobtainium"},
        _domain())
    assert r["status"] == "out_of_domain"
    assert {i["feature"] for i in r["issues"]} == {
        "mass_kg", "holes", "supplier", "material"}


# --- the boundary ---------------------------------------------------------

def test_the_range_edges_are_in_domain():
    for mass in (0.2, 4.0):
        assert domain.check_part(
            {**IN_DOMAIN_PART, "mass_kg": mass}, _domain())["status"] == "in_domain"


def test_a_hair_past_the_edge_is_tolerated_not_flagged():
    # A measurement a rounding-error over the max is an artifact, not an
    # out-of-domain part.
    just_over = 4.0 + 3.8 * domain.NUMERIC_TOLERANCE * 0.5
    assert domain.check_part(
        {**IN_DOMAIN_PART, "mass_kg": just_over}, _domain())["status"] == "in_domain"


def test_clearly_past_the_tolerance_is_flagged():
    well_over = 4.0 + 3.8 * domain.NUMERIC_TOLERANCE * 5
    assert domain.check_part(
        {**IN_DOMAIN_PART, "mass_kg": well_over}, _domain())["status"] == "out_of_domain"


# --- 'unknown' is not 'fine' ----------------------------------------------

def test_no_numeric_ranges_reports_unknown_not_a_clean_bill():
    r = domain.check_part(IN_DOMAIN_PART, _domain(numeric={}, source="unknown"))
    assert r["status"] == "unknown"
    assert "no way to tell" in r["say_to_user"]


def test_an_unseen_category_still_wins_over_unknown_numerics():
    r = domain.check_part({**IN_DOMAIN_PART, "supplier": "Zeta"},
                          _domain(numeric={}, source="unknown"))
    assert r["status"] == "out_of_domain"


def test_an_approximate_domain_says_so():
    r = domain.check_part(IN_DOMAIN_PART, _domain(source="approximate"))
    assert r["status"] == "in_domain"
    assert "sample of the training data" in r["say_to_user"]


# --- degenerate input ------------------------------------------------------

def test_unknown_features_and_bad_values_do_not_crash():
    d = _domain()
    assert domain.check_part({"not_a_feature": 5}, d)["status"] == "in_domain"
    assert domain.check_part({"mass_kg": None}, d)["status"] == "in_domain"
    assert domain.check_part({"mass_kg": "heavy"}, d)["status"] == "in_domain"
    assert domain.check_part({"mass_kg": float("nan")}, d)["status"] == "in_domain"
    assert domain.check_part({}, d)["status"] == "in_domain"


def test_categories_match_across_dtypes():
    # a CSV read gives '3' where the encoder holds 3
    d = _domain(categorical={"grade": [1, 2, 3]})
    assert domain.check_part({"grade": "3"}, d)["status"] == "in_domain"
    assert domain.check_part({"grade": "9"}, d)["status"] == "out_of_domain"


# --- the cap ---------------------------------------------------------------

def test_out_of_domain_forces_quote():
    for r in ("trust", "caution", "quote"):
        assert domain.cap_reliability(r, "out_of_domain") == "quote"


def test_unknown_domain_caps_trust_at_caution_but_never_upgrades():
    assert domain.cap_reliability("trust", "unknown") == "caution"
    assert domain.cap_reliability("quote", "unknown") == "quote"


def test_in_domain_leaves_the_verdict_alone():
    for r in ("trust", "caution", "quote"):
        assert domain.cap_reliability(r, "in_domain") == r


# --- reading the domain off a model ---------------------------------------

def test_stored_domain_is_preferred_and_labelled():
    d = domain.training_domain({
        "feature_domain": {"mass_kg": {"min": 0.2, "max": 4.0}},
        "background_sample": pd.DataFrame({"mass_kg": [1.0, 2.0]}),
        "model": None,
    })
    assert d["numeric_source"] == "stored"
    assert d["numeric"]["mass_kg"]["max"] == 4.0


def test_background_sample_is_the_documented_fallback_for_old_models():
    d = domain.training_domain({
        "background_sample": pd.DataFrame({"mass_kg": [1.0, 3.0], "s": ["a", "b"]}),
        "model": None,
    })
    assert d["numeric_source"] == "approximate"
    assert d["numeric"]["mass_kg"] == {"min": 1.0, "max": 3.0}
    assert "s" not in d["numeric"]          # non-numeric columns skipped


def test_a_model_with_neither_reports_unknown():
    d = domain.training_domain({"model": None})
    assert d["numeric_source"] == "unknown"
    assert d["numeric"] == {}


def test_numeric_domain_from_frame_ignores_non_numeric_and_nans():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "z"]})
    out = domain.numeric_domain_from_frame(X)
    assert out == {"a": {"min": 1.0, "max": 3.0}}


def test_say_to_user_is_jargon_free():
    banned = ("conformal", "extrapolat", "out-of-distribution", "encoder",
              "categorical", "shap", "r²", "residual")
    cases = [
        domain.check_part(IN_DOMAIN_PART, _domain()),
        domain.check_part({**IN_DOMAIN_PART, "supplier": "Zeta"}, _domain()),
        domain.check_part({**IN_DOMAIN_PART, "mass_kg": 900}, _domain()),
        domain.check_part(IN_DOMAIN_PART, _domain(numeric={}, source="unknown")),
        domain.check_part(IN_DOMAIN_PART, _domain(source="approximate")),
    ]
    for r in cases:
        low = r["say_to_user"].lower()
        for term in banned:
            assert term not in low, f"jargon {term!r} in: {r['say_to_user']!r}"
