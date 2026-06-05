"""SHAP-based per-prediction explanations for P2Predict models.

What this module computes
-------------------------
For a fitted P2Predict model and a single input row x, return the additive
decomposition

    f(x) = phi_0 + sum_i phi_i

where phi_0 is the model's baseline (its expected value over a background
population) and phi_i is feature i's Shapley value. The Shapley value is the
unique attribution satisfying efficiency (local accuracy), missingness,
symmetry, and consistency — that uniqueness is what makes the per-feature
numbers defensible in a design-review meeting rather than yet another
heuristic importance score.

Which algorithm we use, and why
-------------------------------
We pick the explainer that is *exact* for the model family and runs in
polynomial time. We do not fall back to KernelExplainer — it is slow and
Monte-Carlo approximate, and we never need it for the three model families
this project supports.

  Linear (Ridge, Lasso)     -> shap.LinearExplainer
      Closed form: phi_i = beta_i * (x_i - E[x_i]). Requires a background
      sample only to estimate E[x_i]; cost is O(F).
  Trees  (RandomForest,     -> shap.TreeExplainer with feature_perturbation=
          XGBoost)             "tree_path_dependent" (Lundberg 2018).
      Exact Shapley values in O(T L D^2), no background sample required —
      the conditional expectations are estimated from the trees' own node
      counts.

Log-target wrap (TransformedTargetRegressor with log1p / expm1)
---------------------------------------------------------------
The inner model predicts log(price). SHAP values on the inner model live in
log space and satisfy local accuracy *in log space*:

    log(pred) - log(base) = sum_i phi_i_log

Exponentiating turns the sum into a product:

    pred / base = prod_i exp(phi_i_log)

So in price space each feature becomes a *multiplicative factor*
exp(phi_i_log) -- e.g. "Region=EU multiplies the predicted price by 1.18
(+18%)". This is the axiomatically clean reading.

For procurement readability we additionally surface an "approximate dollar
attribution" obtained by proportionally rescaling the log-space contributions
to the price-space delta (pred - base). This *forces* additivity in dollars
at the cost of breaking the SHAP axioms — it is not strict SHAP, and we label
it that way in the report and in the CLI.

Source-feature roll-up
----------------------
SHAP gives one value per *transformed* feature. We sum across the columns
that came from the same source column (one-hot dummies for linear models;
ordinal-encoded categoricals for tree models, where this is a no-op).
Summing one-hot dummies' Shapley values to attribute to the source column is
standard practice and is sound under SHAP's additivity property when the
dummies are mutually exclusive (exactly one is 1 at a time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

# Local-accuracy sanity-check tolerance. Floating-point + SHAP internals can
# leave a tiny residual; anything bigger is a sign something is wrong with
# the explainer choice or the transformed-matrix shape.
_LOCAL_ACCURACY_TOL = 1e-4


@dataclass
class Explanation:
    """Per-row attribution result.

    The contract:
      contributions[col] are in the *inner model's output space* (price for
      a non-log model; log(price) for a log-target model). They satisfy
      local accuracy: baseline + sum(contributions.values()) ~= prediction
      to within _LOCAL_ACCURACY_TOL.

    For log-target models the price-space fields are populated and the
    multiplicative_factors are the only attribution form that strictly
    satisfies the SHAP axioms in price space. dollar_attribution is a
    proportional rescaling — additive but not strict SHAP.
    """

    baseline: float
    prediction: float
    contributions: dict[str, float]
    log_target: bool = False
    baseline_price: Optional[float] = None
    predicted_price: Optional[float] = None
    multiplicative_factors: Optional[dict[str, float]] = None
    dollar_attribution: Optional[dict[str, float]] = None
    residual: float = 0.0  # local-accuracy residual, for diagnostics
    # True iff product(multiplicative_factors) == predicted_price / baseline_price
    # holds strictly. Holds for the v0.4 log/exp wrap; not for an older
    # log1p/expm1 wrap, where the factors apply to (1 + price) instead.
    strict_multiplicative: bool = False


def _unwrap(model):
    """Return (inner_pipeline, is_log_target, inverse_func).

    ``inverse_func`` is read off the TransformedTargetRegressor so the
    explanation code stays correct whichever forward/inverse pair was used
    at training time (v0.4+ uses log/exp; older models may have used
    log1p/expm1). We invert via this function rather than hard-coding
    ``expm1`` so the multiplicative-axiom math only holds strictly under
    the right pairing (log/exp) but doesn't *silently lie* under the
    wrong one — we surface that case via a flag.
    """
    if isinstance(model, TransformedTargetRegressor):
        inverse = getattr(model, "inverse_func", None) or np.exp
        return model.regressor_, True, inverse
    return model, False, None


def _detect_family(estimator) -> str:
    name = type(estimator).__name__.lower()
    if any(t in name for t in ("ridge", "lasso", "linear", "elasticnet")):
        return "linear"
    if any(t in name for t in ("forest", "xgb", "gradientboost", "boost", "tree")):
        return "tree"
    return "unknown"


def _rollup_to_source(
    preprocessor, source_cols: list[str], shap_row: np.ndarray
) -> dict[str, float]:
    """Sum transformed-feature SHAP values back to source columns.

    Uses the longest-source-column-prefix match (the same logic used by
    extract_feature_importances), so source columns whose names share a
    prefix — e.g. 'weight' and 'weight_extra' — are kept separate rather
    than collapsed.
    """
    raw_names = list(preprocessor.get_feature_names_out())
    if len(raw_names) != len(shap_row):
        raise ValueError(
            f"Transformed-feature/SHAP-value length mismatch: "
            f"{len(raw_names)} names vs {len(shap_row)} values."
        )

    by_source: dict[str, float] = {col: 0.0 for col in source_cols}
    for raw_name, value in zip(raw_names, shap_row):
        rest = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
        match = None
        for col in source_cols:
            if rest == col or rest.startswith(f"{col}_"):
                if match is None or len(col) > len(match):
                    match = col
        if match is None:
            match = rest
            by_source.setdefault(match, 0.0)
        by_source[match] += float(value)
    return by_source


def _scalar_expected_value(explainer) -> float:
    """SHAP returns expected_value as either a scalar or a 1-element array
    depending on the model and version. Normalise to a Python float."""
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        ev = np.atleast_1d(ev)
        if ev.size != 1:
            # Multi-output models are not in our scope (regression only).
            raise ValueError(
                "SHAP expected_value has multiple outputs; only single-output "
                "regression is supported."
            )
        return float(ev[0])
    return float(ev)


def _shap_values(explainer, X_t):
    """Get a (n_samples, n_features) SHAP value matrix regardless of the
    library's version-dependent return shape."""
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        # Classification returns a list of per-class arrays; regression
        # returns either a 2-D array or a 1-D row. We only do regression.
        if len(sv) != 1:
            raise ValueError("Unexpected multi-output SHAP result.")
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv


def _build_explainer(estimator, family: str, background_X_t):
    """Construct the right SHAP explainer.

    Trees use the tree-path-dependent algorithm — no background needed, and
    the result is exact in O(TLD^2). Linear models use the closed-form
    LinearExplainer; that one *does* need a background to estimate E[x_i].
    """
    import shap  # imported lazily so the rest of P2Predict has no hard
                  # dependency on shap unless --explain is actually used.

    if family == "tree":
        return shap.TreeExplainer(
            estimator, feature_perturbation="tree_path_dependent"
        )
    if family == "linear":
        if background_X_t is None or len(background_X_t) == 0:
            raise ValueError(
                "Linear models require a background sample for SHAP. "
                "Re-train with v0.4 (which persists one) or pass background_X."
            )
        return shap.LinearExplainer(estimator, background_X_t)
    raise ValueError(
        f"No SHAP explainer wired for estimator '{type(estimator).__name__}'."
    )


def explain_row(
    model,
    x: pd.DataFrame,
    background_X: Optional[pd.DataFrame] = None,
) -> Explanation:
    """Compute the SHAP explanation for a single-row DataFrame x.

    Parameters
    ----------
    model
        A fitted P2Predict pipeline — either a sklearn ``Pipeline`` or a
        ``TransformedTargetRegressor`` wrapping one.
    x
        Single-row DataFrame with the same source columns the pipeline was
        trained on. Passing more than one row is supported but each row gets
        explained independently — see :func:`explain_batch`.
    background_X
        Optional background sample of raw (pre-preprocessor) feature rows.
        Required for linear models, ignored for tree models.
    """
    if len(x) != 1:
        raise ValueError("explain_row expects a single-row DataFrame.")

    inner, is_log_target, inverse_func = _unwrap(model)
    preprocessor = inner.named_steps["preprocessor"]
    estimator = inner.named_steps["model"]
    family = _detect_family(estimator)

    x_t = np.asarray(preprocessor.transform(x))
    bg_t = (
        np.asarray(preprocessor.transform(background_X))
        if background_X is not None
        else None
    )

    explainer = _build_explainer(estimator, family, bg_t)
    sv = _shap_values(explainer, x_t)[0]  # one row of SHAP values

    baseline = _scalar_expected_value(explainer)
    inner_pred = float(np.asarray(estimator.predict(x_t)).ravel()[0])

    source_cols = list(x.columns)
    contributions = _rollup_to_source(preprocessor, source_cols, sv)

    # Local-accuracy check in *inner-model* output space. This catches issues
    # like a mis-extracted preprocessor or a wrong-family explainer pick.
    residual = float(inner_pred - (baseline + sum(contributions.values())))
    if abs(residual) > _LOCAL_ACCURACY_TOL * max(1.0, abs(inner_pred)):
        # Don't raise — log via the Explanation so the CLI can surface it.
        pass

    if not is_log_target:
        return Explanation(
            baseline=baseline,
            prediction=inner_pred,
            contributions=contributions,
            residual=residual,
        )

    # Log-target post-processing.
    #
    # contributions are in inner-model output space. When the wrap is log/exp
    # (the v0.4+ default) the per-feature multiplicative factor in price
    # space is exp(contribution), and the product of factors *exactly*
    # reproduces predicted_price / baseline_price. This is the axiomatic
    # SHAP statement in price space.
    #
    # For other wraps (e.g. v0.2/v0.3 log1p/expm1) the multiplicative
    # interpretation applies on the inverse_func's *pre-shift* scale rather
    # than on price directly — for log1p that's (1 + price). We keep the
    # exp() factor (which is what SHAP gives us in log space) and let the
    # caller know via the strict_multiplicative flag.
    baseline_price = float(inverse_func(baseline))
    predicted_price = float(inverse_func(inner_pred))
    multiplicative_factors = {
        col: float(np.exp(v)) for col, v in contributions.items()
    }
    strict_multiplicative = inverse_func is np.exp

    # Approximate dollar attribution: rescale log-space contributions so they
    # sum to the price-space delta. This is *not* strict SHAP — see the
    # module docstring — but it is the form procurement readers naturally
    # want, and we label it as approximate everywhere it is shown.
    delta_price = predicted_price - baseline_price
    log_total = sum(contributions.values())
    if abs(log_total) > 1e-12:
        dollar_attribution = {
            col: float(delta_price * v / log_total)
            for col, v in contributions.items()
        }
    else:
        dollar_attribution = {col: 0.0 for col in contributions}

    return Explanation(
        baseline=baseline,
        prediction=inner_pred,
        contributions=contributions,
        log_target=True,
        baseline_price=baseline_price,
        predicted_price=predicted_price,
        multiplicative_factors=multiplicative_factors,
        dollar_attribution=dollar_attribution,
        residual=residual,
        strict_multiplicative=strict_multiplicative,
    )


def top_drivers(
    explanation: Explanation, n: int = 3, signed: bool = True
) -> list[tuple[str, float]]:
    """Return the n source features with the largest |contribution|.

    In the log-target case we rank by absolute log-space contribution (which
    is monotone with |log(multiplicative_factor)|) and report the actual
    multiplicative factor as the numeric value, since that is the
    axiomatically clean per-feature quantity in price space.
    """
    items = list(explanation.contributions.items())
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    items = items[:n]
    if explanation.log_target and explanation.multiplicative_factors is not None:
        return [
            (col, explanation.multiplicative_factors[col]) for col, _ in items
        ]
    if not signed:
        return [(col, abs(v)) for col, v in items]
    return items
