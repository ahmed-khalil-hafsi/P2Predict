"""What-if analysis: compare a base scenario with a counterfactual.

What this module does
---------------------
Given a base feature row x and a set of changes (a dict of column ->
new value), return:

  • Base prediction f(x) and counterfactual prediction f(x')
  • Likely-range intervals on both (when v0.5+ calibration is available)
  • Total delta (in target units and percentage)
  • For log-target models, also the multiplicative factor (cf / base)
  • Per-feature SHAP decomposition of the delta

The procurement use case is the design-review question: "what if we
change region from CN to EU?" Today that takes two CLI runs and eyeball
arithmetic. This module gives one structured answer.

The math of the delta decomposition
-----------------------------------
SHAP's local-accuracy axiom gives, for any feature vector x,

    f(x) = phi_0 + sum_i phi_i(x)

So for the *difference* between two predictions:

    f(x') - f(x) = sum_i (phi_i(x') - phi_i(x))                  (*)

Each term phi_i(x') - phi_i(x) is feature i's contribution to the
delta. This is exact — not an approximation — and the sum is the total
delta by construction.

A subtlety worth surfacing in the CLI: when the user only changes one
or two features, the *unchanged* features can still have non-zero
deltas in (*). That's because SHAP conditions on the full feature
vector — changing region can shift the attribution to weight if there's
an interaction in the underlying model. Mathematically correct,
potentially confusing. We expose changed-feature contributions as the
"direct" effect and lump everything else into a single "interaction
effects" row when it's material (>5% of the absolute total delta).

For log-target models the math runs in log space:

    log(f(x')) - log(f(x)) = sum_i (phi_i^log(x') - phi_i^log(x))

Exponentiating gives a multiplicative decomposition in price space:

    f(x') / f(x) = prod_i exp(phi_i^log(x') - phi_i^log(x))

Each per-feature factor is exp(delta_phi) — the multiplicative
contribution of changing that feature (and the implicit interactions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from p2predict.explain import Explanation, explain_batch
from p2predict.intervals import IntervalResult, predict_interval
from p2predict.quality import feature_signal

# A SHAP delta below this absolute fraction of the total delta is rolled
# up into the "other interaction effects" bucket in the CLI rendering.
INTERACTION_MATERIALITY_THRESHOLD = 0.05

# Below this |percent change| we treat the move as effectively flat ("no
# change" / free to concede) and skip the interaction/sign-flip reliability
# checks — with a ~0 delta their arithmetic degenerates (0 ≥ 0) and would
# mislabel a genuinely free concession as unstable.
RELIABILITY_FLAT_PCT = 0.5


@dataclass
class WhatIfResult:
    """Structured comparison between a base prediction and a
    counterfactual where the user changed one or more feature values.

    All numbers are in the user's target units (e.g. price). For log-
    target models, the multiplicative_factor field is also populated so
    callers can show "x1.18 vs base" alongside "+$0.32".

    changed_contributions[k] is the SHAP-attributed contribution of the
    feature k's change to the total delta. Features the user did NOT
    explicitly change have their (potentially nonzero, due to
    interactions) contributions rolled up into interaction_contribution.
    """

    base_prediction: float
    counterfactual_prediction: float
    delta: float
    delta_pct: float
    changes: dict[str, tuple]  # column -> (base_value, cf_value)
    changed_contributions: dict[str, float]
    interaction_contribution: float
    log_target: bool = False
    multiplicative_factor: Optional[float] = None
    changed_multiplicative_factors: Optional[dict[str, float]] = None
    interaction_multiplicative_factor: Optional[float] = None
    base_interval: Optional[IntervalResult] = None
    cf_interval: Optional[IntervalResult] = None
    # SHAP residual: how far the per-feature decomposition is from the
    # total delta. Should be near 0 modulo floating-point; exposed for
    # diagnostics.
    decomposition_residual: float = 0.0
    # How far to trust the DIRECTION of this comparison, mirroring the
    # per-part interval verdict. 'trust' | 'caution' | 'quote', with a
    # short machine reason and a plain sentence for the user. Populated only
    # when feature_importances are supplied to compute_whatif (so the CLI
    # path that never passes them is unchanged); None otherwise.
    reliability: Optional[str] = None
    reliability_reason: Optional[str] = None
    reliability_say_to_user: Optional[str] = None


def _importance_signals(
    feature_importances: list[tuple[str, float]], features: set[str]
) -> dict[str, str]:
    """Map each requested feature to its 'strong'|'moderate'|'weak' signal,
    reusing the same importance-share thresholds the quality report uses so
    "strong" means one thing across the whole surface. A feature missing from
    the importances (e.g. dropped by the model) is treated as 'weak'."""
    total = sum(abs(float(v)) for _, v in feature_importances) or 1.0
    pct = {name: abs(float(v)) / total * 100.0 for name, v in feature_importances}
    return {f: feature_signal(pct.get(f, 0.0)) for f in features}


def assess_reliability(
    result: "WhatIfResult", changed_signals: dict[str, str]
) -> tuple[str, str, str]:
    """Judge how far to trust the direction of a what-if, from signals that
    are already computed — no ground-truth expected sign required.

    Returns ``(reliability, reason, say_to_user)`` where reliability is
    'trust' | 'caution' | 'quote'. Worst-case wins; rules are checked from
    most to least severe:

      1. weak-signal change  -> 'quote'  (direction rests on very few parts)
      2. sign flip           -> 'quote'  (the spec you changed pushes the
                                           other way; the headline is a
                                           side-effect, not your change)
      3. interaction-dominant-> 'caution'(most of the move is knock-on
                                           effects, not the change itself)
      4. moderate-signal     -> 'caution'(thinly sampled; directional only)
      5. otherwise           -> 'trust'

    Rules 2 and 3 depend on a non-trivial delta; on an effectively flat move
    (|delta_pct| < RELIABILITY_FLAT_PCT) they are skipped so a genuinely free
    concession isn't mislabelled.
    """
    signals = set(changed_signals.values())
    material = abs(float(result.delta_pct)) >= RELIABILITY_FLAT_PCT

    direct = sum(result.changed_contributions.values())
    interaction = float(result.interaction_contribution)
    # inner-space direct delta and target-space net delta share sign (exp is
    # monotonic for log-target; identity otherwise), so this comparison holds
    # in both cases.
    sign_flip = material and direct != 0.0 and (direct > 0) != (result.delta > 0)
    interaction_dominant = material and abs(interaction) >= abs(direct)

    if "weak" in signals:
        weak = sorted(f for f, s in changed_signals.items() if s == "weak")
        return (
            "quote",
            "weak_signal",
            f"'{', '.join(weak)}' is barely represented in this data, so the "
            "direction here is close to a guess — get a real quote rather than "
            "negotiating on this move.",
        )
    if sign_flip:
        return (
            "quote",
            "sign_flip",
            "The spec you changed actually pushes the price the other way — this "
            "result is driven by side-effects, not your change, so don't trust "
            "its direction; get a quote.",
        )
    if interaction_dominant:
        return (
            "caution",
            "interaction_dominant",
            "Treat this as directional only — most of the swing comes from "
            "knock-on effects, not the spec you changed. Sanity-check it before "
            "you hold a supplier to it.",
        )
    if "moderate" in signals:
        moderate = sorted(f for f, s in changed_signals.items() if s == "moderate")
        return (
            "caution",
            "moderate_signal",
            f"Treat this as directional only — '{', '.join(moderate)}' is only "
            "moderately represented in the data. Sanity-check it before you hold "
            "a supplier to it.",
        )
    return (
        "trust",
        "strong_signal",
        "Solid directional comparison — the spec you changed is what's driving "
        "the difference; fine to take into the negotiation.",
    )


def parse_changes(spec: str) -> dict[str, str]:
    """Parse a ``"Region:EU,Supplier:B"`` string into ``{"Region": "EU",
    "Supplier": "B"}``. Same syntax as ``-p`` for consistency.
    """
    out: dict[str, str] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid change spec '{token}'. Use 'Feature:Value', "
                f"comma-separated."
            )
        key, _, value = token.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or value == "":
            raise ValueError(f"Invalid change spec '{token}'.")
        out[key] = value
    return out


def _apply_changes(
    base: pd.DataFrame, changes: dict[str, str], feature_types: dict[str, str]
) -> pd.DataFrame:
    """Return a copy of the base row with the requested feature values
    overwritten. Numeric features are coerced from string to float so the
    caller can pass ``"15"`` for a Weight column."""
    cf = base.copy()
    for col, raw_value in changes.items():
        if col not in cf.columns:
            raise ValueError(
                f"Cannot change '{col}': not a training feature for this "
                f"model. Known features: {list(cf.columns)}."
            )
        if feature_types.get(col) == "Numerical":
            try:
                value = float(raw_value)
            except ValueError:
                raise ValueError(
                    f"Cannot change '{col}' to '{raw_value}': feature is "
                    "numeric."
                )
        else:
            value = raw_value
        cf[col] = value
    return cf


def compute_whatif(
    model,
    base_features: pd.DataFrame,
    changes: dict[str, str],
    feature_types: dict[str, str],
    background_X: Optional[pd.DataFrame] = None,
    calibration: Optional[dict] = None,
    coverage: float = 0.90,
    feature_importances: Optional[list[tuple[str, float]]] = None,
) -> WhatIfResult:
    """Run the comparison. ``base_features`` is a single-row DataFrame
    whose columns are the model's training features in order.

    ``feature_types`` maps each column to ``"Numerical"`` or
    ``"Categorical"`` (the same dict the predict CLI already builds from
    the saved preprocessor).

    ``feature_importances`` is the optional ``(name, importance)`` list from
    ``extract_feature_importances``. When supplied, the result carries a
    direction-reliability verdict (see ``assess_reliability``); when omitted,
    the reliability fields stay None and behaviour is unchanged.
    """
    if len(base_features) != 1:
        raise ValueError("compute_whatif expects a single-row base_features.")
    if not changes:
        raise ValueError("compute_whatif requires at least one change.")

    cf_features = _apply_changes(base_features, changes, feature_types)

    # Map the requested changes to (base_value, cf_value) tuples for the
    # rendered summary. Use the values from the actual DataFrames so the
    # numeric coercion is reflected.
    changes_summary: dict[str, tuple] = {}
    for col in changes:
        changes_summary[col] = (
            base_features.iloc[0][col],
            cf_features.iloc[0][col],
        )

    base_pred = float(np.asarray(model.predict(base_features)).ravel()[0])
    cf_pred = float(np.asarray(model.predict(cf_features)).ravel()[0])
    delta = cf_pred - base_pred
    delta_pct = 100.0 * delta / base_pred if base_pred != 0 else float("nan")

    # One batch call so the (expensive) SHAP explainer is built once for
    # both the base and the counterfactual row.
    base_explanation, cf_explanation = explain_batch(
        model,
        pd.concat([base_features, cf_features], ignore_index=True),
        background_X=background_X,
    )

    # Per-feature delta in SHAP attribution. For non-log models this lives
    # in target units; for log-target models it lives in log space and is
    # converted to multiplicative factors at the end.
    per_feature_delta: dict[str, float] = {}
    for col in base_explanation.contributions.keys():
        per_feature_delta[col] = (
            cf_explanation.contributions[col] - base_explanation.contributions[col]
        )

    changed_keys = set(changes.keys())
    changed_contributions = {
        col: v for col, v in per_feature_delta.items() if col in changed_keys
    }
    interaction_contribution = sum(
        v for col, v in per_feature_delta.items() if col not in changed_keys
    )

    log_target = bool(base_explanation.log_target)
    multiplicative_factor: Optional[float] = None
    changed_multiplicative_factors: Optional[dict[str, float]] = None
    interaction_multiplicative_factor: Optional[float] = None

    # Sanity check: decomposition (in inner-model output space) should equal
    # the inner-model delta. For non-log models the inner space IS the
    # target space and decomposition_residual is in target units. For
    # log-target models it's in log space.
    inner_delta = (cf_explanation.prediction - base_explanation.prediction)
    decomposition_residual = inner_delta - sum(per_feature_delta.values())

    if log_target:
        # Cf / base = product of exp(delta_phi_i). Build the multiplicative
        # factor decomposition.
        multiplicative_factor = float(cf_pred / base_pred) if base_pred > 0 else float("nan")
        changed_multiplicative_factors = {
            col: float(np.exp(v)) for col, v in changed_contributions.items()
        }
        interaction_multiplicative_factor = float(np.exp(interaction_contribution))

    base_interval: Optional[IntervalResult] = None
    cf_interval: Optional[IntervalResult] = None
    if calibration is not None:
        base_interval = predict_interval(model, base_features, calibration, coverage)[0]
        cf_interval = predict_interval(model, cf_features, calibration, coverage)[0]

    result = WhatIfResult(
        base_prediction=base_pred,
        counterfactual_prediction=cf_pred,
        delta=delta,
        delta_pct=delta_pct,
        changes=changes_summary,
        changed_contributions=changed_contributions,
        interaction_contribution=interaction_contribution,
        log_target=log_target,
        multiplicative_factor=multiplicative_factor,
        changed_multiplicative_factors=changed_multiplicative_factors,
        interaction_multiplicative_factor=interaction_multiplicative_factor,
        base_interval=base_interval,
        cf_interval=cf_interval,
        decomposition_residual=decomposition_residual,
    )

    if feature_importances is not None:
        changed_signals = _importance_signals(feature_importances, changed_keys)
        reliability, reason, say = assess_reliability(result, changed_signals)
        result.reliability = reliability
        result.reliability_reason = reason
        result.reliability_say_to_user = say

    return result


def interaction_is_material(result: WhatIfResult) -> bool:
    """True iff the interaction contribution is large enough to surface
    in the CLI rendering. Threshold: 5% of |total delta|."""
    if abs(result.delta) < 1e-12:
        return abs(result.interaction_contribution) > 1e-6
    return (
        abs(result.interaction_contribution) / abs(result.delta)
        > INTERACTION_MATERIALITY_THRESHOLD
    )
