from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from p2predict.explain import Explanation
from p2predict.whatif import WhatIfResult, interaction_is_material


def inner_pipeline(model):
    """Unwrap a TransformedTargetRegressor to get the inner Pipeline."""
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def extract_feature_info(pipeline):
    """Return (feature_types, all_categories) from a fitted preprocessor.

    Works with OneHotEncoder, OrdinalEncoder, and TargetEncoder pipelines.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_types: dict[str, str] = {}
    all_categories: dict[str, list] = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_types.update({col: "Numerical" for col in columns})
        elif name == "cat":
            feature_types.update({col: "Categorical" for col in columns})

            encoder = transformer
            if isinstance(transformer, Pipeline):
                if "onehot" in transformer.named_steps:
                    encoder = transformer.named_steps["onehot"]
                elif "target" in transformer.named_steps:
                    encoder = transformer.named_steps["target"]

            if isinstance(encoder, (OneHotEncoder, OrdinalEncoder, TargetEncoder)) and hasattr(
                encoder, "categories_"
            ):
                all_categories = {
                    col: cat.tolist()
                    for col, cat in zip(columns, encoder.categories_)
                }

    return feature_types, all_categories


def coerce_features(features_df, feature_types):
    """Coerce numerical columns to numeric dtype."""
    for col, kind in feature_types.items():
        if col in features_df.columns and kind == "Numerical":
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
    return features_df


def interval_to_dicts(intervals) -> list[dict]:
    """Serialize a list of IntervalResult to JSON-ready dicts."""
    return [
        {
            "low": float(ir.low),
            "prediction": float(ir.prediction),
            "high": float(ir.high),
            "band": ir.band,
        }
        for ir in intervals
    ]


def explanation_to_dict(explanation: Explanation) -> dict:
    """Serialize an Explanation to a JSON-ready dict.

    The dict carries TWO views of the same attribution:

      - A business view the agent can quote to a category manager verbatim:
        ``starting_point`` (the baseline price every part starts from) and
        ``price_drivers`` (each spec/supplier's effect in dollars AND percent,
        biggest mover first). This is the view to lead with.
      - The technical view (``baseline``, ``contributions``,
        ``multiplicative_factors``, ``dollar_attribution``, ``residual``) for
        callers that need the raw SHAP numbers. Do NOT surface these key names
        to a procurement user.
    """
    out = {
        "baseline": float(explanation.baseline),
        "prediction": float(explanation.prediction),
        "log_target": bool(explanation.log_target),
        "contributions": [
            {"feature": k, "value": float(v)}
            for k, v in sorted(
                explanation.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
            )
        ],
        "residual": float(explanation.residual),
    }
    if explanation.log_target and explanation.multiplicative_factors is not None:
        out["multiplicative_factors"] = [
            {"feature": k, "factor": float(v)}
            for k, v in sorted(
                explanation.multiplicative_factors.items(),
                key=lambda kv: abs(np.log(kv[1])) if kv[1] > 0 else 0.0,
                reverse=True,
            )
        ]
        out["dollar_attribution"] = (
            [
                {"feature": k, "value": float(v)}
                for k, v in sorted(
                    explanation.dollar_attribution.items(),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )
            ]
            if explanation.dollar_attribution is not None
            else None
        )
    else:
        out["multiplicative_factors"] = None
        out["dollar_attribution"] = None

    out["starting_point"] = _business_starting_point(explanation)
    out["price_drivers"] = _business_price_drivers(explanation)
    return out


def _business_starting_point(explanation: Explanation) -> float:
    """The baseline price every part 'starts from' before its specs apply, in
    dollars (price space for log-target models)."""
    if explanation.log_target and explanation.baseline_price is not None:
        return float(explanation.baseline_price)
    return float(explanation.baseline)


def _business_price_drivers(explanation: Explanation) -> list[dict]:
    """A single plain-language attribution list the agent can quote directly.

    Each entry is one spec/supplier and its effect on the price, expressed in
    BOTH dollars and percent, biggest absolute mover first:
        {"driver": "Supplier ADI", "effect_dollars": 0.72, "effect_pct": 18.0}
    Works for additive and log-target models alike, so the caller never has to
    branch on the model's internal scale.
    """
    drivers: list[dict] = []
    if explanation.log_target and explanation.multiplicative_factors is not None:
        dollars = explanation.dollar_attribution or {}
        for feature, factor in explanation.multiplicative_factors.items():
            drivers.append({
                "driver": feature,
                "effect_dollars": (
                    round(float(dollars[feature]), 4) if feature in dollars else None
                ),
                "effect_pct": round((float(factor) - 1.0) * 100.0, 1),
            })
    else:
        # Additive model: contributions are already dollars; percent is the
        # share of the baseline each driver moves the price by.
        base = float(explanation.baseline) or 1.0
        for feature, value in explanation.contributions.items():
            drivers.append({
                "driver": feature,
                "effect_dollars": round(float(value), 4),
                "effect_pct": round(float(value) / base * 100.0, 1),
            })
    drivers.sort(key=lambda d: abs(d["effect_dollars"] or 0.0), reverse=True)
    return drivers


def whatif_to_dict(result: WhatIfResult) -> dict:
    """Serialize a WhatIfResult to a JSON-ready dict.

    ``summary`` is the plain-language headline the agent can quote to a category
    manager ("Switching to Microchip saves $0.41 per part, -12%"); the
    remaining keys are the technical detail behind it. Lead with ``summary``.
    """
    delta = float(result.delta)
    direction = "no change"
    if delta > 0:
        direction = "adds"
    elif delta < 0:
        direction = "saves"
    return {
        "summary": {
            "direction": direction,  # "adds" | "saves" | "no change"
            "effect_dollars": round(abs(delta), 4),
            "effect_pct": round(abs(float(result.delta_pct)), 1),
            "new_price": round(float(result.counterfactual_prediction), 4),
            "old_price": round(float(result.base_prediction), 4),
        },
        "changes": {
            col: {"from": base_val, "to": cf_val}
            for col, (base_val, cf_val) in result.changes.items()
        },
        "base_prediction": float(result.base_prediction),
        "counterfactual_prediction": float(result.counterfactual_prediction),
        "delta": float(result.delta),
        "delta_pct": float(result.delta_pct),
        "log_target": bool(result.log_target),
        "multiplicative_factor": (
            float(result.multiplicative_factor)
            if result.multiplicative_factor is not None
            else None
        ),
        "changed_contributions": [
            {"feature": k, "value": float(v)}
            for k, v in sorted(
                result.changed_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
            )
        ],
        "interaction_contribution": float(result.interaction_contribution),
        "interaction_is_material": bool(interaction_is_material(result)),
        "base_interval": (
            {"low": float(result.base_interval.low), "high": float(result.base_interval.high)}
            if result.base_interval is not None
            else None
        ),
        "cf_interval": (
            {"low": float(result.cf_interval.low), "high": float(result.cf_interval.high)}
            if result.cf_interval is not None
            else None
        ),
    }
