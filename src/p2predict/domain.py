"""Was this part answerable at all?

Every answer P2Predict ships is auditable — the attribution decomposes, the
interval has a coverage proof, the what-if delta sums. All three audit *how
good* an answer is. None of them asks the question that comes first: was the
part inside the data the model was built on?

Nothing in the predict path noticed. The conformal guarantee in ``intervals``
is explicitly conditional on future inputs coming from the training
distribution, and that precondition was documented but never checked — so an
out-of-domain part is simultaneously the one input where the interval means
nothing and the one input where nothing warned. Worse, the failure inverts:
because band selection keys on the *predicted* value, an extrapolated part
often prices into a better-sampled segment and comes back with a **narrower**
band than a legitimate one, reading as more trustworthy rather than less.

This module answers the prior question. It is a box check — per-feature ranges
and observed categories — which is cheap, explainable to a buyer ("900 kg is
225× the heaviest part in the data"), and catches the cases that actually
reach users. It is deliberately not a convex hull or a density estimate: a part
in range on every spec but an unobserved *combination* still passes, and that
limit is documented rather than hidden.

Rationale and measurements: research/out_of_domain_flag.md.
"""
from __future__ import annotations

import numpy as np

# How far outside the observed range still counts as the edge rather than
# extrapolation. Tree models have no split beyond their training range, so
# anything past the edge returns the edge value -- but a measurement a hair
# over the max is a rounding artifact, not an out-of-domain part.
NUMERIC_TOLERANCE = 0.02   # 2% of the observed range


def numeric_domain_from_frame(X) -> dict:
    """Exact per-column min/max over the numeric training features.

    Computed on the TRAINING split only -- the holdout must not leak into
    anything the model carries.
    """
    out: dict[str, dict] = {}
    for col in getattr(X, "columns", []):
        values = X[col]
        if not np.issubdtype(values.dtype, np.number):
            continue
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        out[col] = {"min": float(finite.min()), "max": float(finite.max())}
    return out


def training_domain(loaded: dict) -> dict:
    """The box the model was trained in, from whatever the artifact carries.

    ``numeric_source`` says how far to trust the numeric half:

    * ``stored`` — exact ranges recorded at train time.
    * ``approximate`` — derived from ``background_sample``, the ~100 training
      rows kept for SHAP. It is a *subset*, so its range is narrower than the
      true one: this can over-report an out-of-domain part, never under-report
      one. That direction is the safe one, and it is why old models still get
      a useful check with no retrain.
    * ``unknown`` — neither available. The numeric half reports nothing rather
      than a false all-clear.

    The categorical half needs no stored state at all: the fitted encoder
    already holds the categories it saw, so it works on every model on disk.
    """
    numeric: dict[str, dict] = {}
    source = "unknown"

    stored = loaded.get("feature_domain")
    if isinstance(stored, dict) and stored:
        numeric = {k: dict(v) for k, v in stored.items()}
        source = "stored"
    else:
        sample = loaded.get("background_sample")
        if sample is not None and len(sample) > 0:
            for col in sample.columns:
                values = sample[col]
                if not np.issubdtype(values.dtype, np.number):
                    continue
                finite = values[np.isfinite(values)]
                if len(finite) == 0:
                    continue
                numeric[col] = {"min": float(finite.min()), "max": float(finite.max())}
            if numeric:
                source = "approximate"

    categorical: dict[str, list] = {}
    try:
        from p2predict.model_utils import extract_feature_info, inner_pipeline

        _, categories = extract_feature_info(inner_pipeline(loaded["model"]))
        categorical = {k: list(v) for k, v in (categories or {}).items()}
    except Exception:
        # A model whose encoder can't be introspected loses the categorical
        # half only; the numeric half still reports.
        categorical = {}

    return {
        "numeric": numeric,
        "categorical": categorical,
        "numeric_source": source,
    }


def _numeric_issue(feature, value, lo, hi):
    """Plain description of how far outside the observed range a value sits."""
    span = hi - lo
    tol = abs(span) * NUMERIC_TOLERANCE
    if lo - tol <= value <= hi + tol:
        return None
    if value > hi:
        # "225x the largest" reads better than a raw delta, but only when the
        # edge is positive and non-trivial -- otherwise state it plainly.
        if hi > 0 and value / hi >= 2:
            detail = (f"{_fmt(value)} is {value / hi:.0f}x the largest value in "
                      f"the data ({_fmt(hi)})")
        else:
            detail = (f"{_fmt(value)} is above the largest value in the data "
                      f"({_fmt(hi)})")
    else:
        detail = (f"{_fmt(value)} is below the smallest value in the data "
                  f"({_fmt(lo)})")
    return {"feature": feature, "kind": "numeric_out_of_range", "detail": detail}


def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v):,}"
    return f"{v:,.4g}"


def check_part(features: dict, domain: dict) -> dict:
    """Is this part inside the model's training domain?

    Returns ``status`` ('in_domain' | 'out_of_domain' | 'unknown'), the list of
    ``issues`` naming which specs fell outside and by how much, and a
    plain-language ``say_to_user``.

    ``unknown`` means the numeric ranges could not be established — not that
    the part looked fine. The two are different claims and collapsing them is
    how a part the model has never seen earns a confident answer.
    """
    issues = []

    for feature, value in (features or {}).items():
        cats = domain.get("categorical", {}).get(feature)
        if cats is not None:
            if value is not None and not _in_categories(value, cats):
                issues.append({
                    "feature": feature,
                    "kind": "unseen_category",
                    "detail": (
                        f"'{value}' was never in the data — it is being priced "
                        f"as the catalog average, not as '{value}'"
                    ),
                })
            continue

        rng = domain.get("numeric", {}).get(feature)
        if rng is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        issue = _numeric_issue(feature, v, float(rng["min"]), float(rng["max"]))
        if issue:
            issues.append(issue)

    source = domain.get("numeric_source", "unknown")
    if issues:
        status = "out_of_domain"
    elif source == "unknown":
        status = "unknown"
    else:
        status = "in_domain"

    return {
        "status": status,
        "issues": issues,
        "numeric_source": source,
        "say_to_user": _say_to_user(status, issues, source),
    }


def _in_categories(value, cats) -> bool:
    """Match a supplied value against observed categories, tolerating dtype.

    A category read back from a CSV is a str where the encoder may hold a
    numpy scalar, so compare on the string form as well as the raw value.
    """
    if value in cats:
        return True
    return str(value) in {str(c) for c in cats}


def _say_to_user(status: str, issues: list, source: str) -> str:
    if status == "unknown":
        return (
            "This model doesn't record the range of parts it was built on, so "
            "there's no way to tell whether this part is like the ones it "
            "learned from. Treat the number as indicative and sanity-check it."
        )
    if status == "in_domain":
        base = "This part looks like the ones the model was built on."
        if source == "approximate":
            base += (
                " (Checked against a sample of the training data, so a part "
                "just outside the range may not be caught.)"
            )
        return base

    named = ", ".join(i["detail"] for i in issues[:3])
    more = "" if len(issues) <= 3 else f", and {len(issues) - 3} more"
    return (
        f"This part is outside what the model has seen: {named}{more}. "
        "The estimate and its likely-range aren't reliable here — get a quote "
        "rather than benchmarking off this number."
    )


def cap_reliability(reliability: str, status: str) -> str:
    """Hold a per-part verdict to what the domain check can support.

    The interval verdict grades the model's *price segment*, not the part —
    under a log-target the prediction cancels out of the width ratio entirely,
    so a model has at most three distinct verdicts it can ever emit, selected
    by predicted price. That machinery cannot notice an impossible part, and
    was never designed to. This is the cap that can.
    """
    order = ["trust", "caution", "quote"]
    if status == "out_of_domain":
        return "quote"
    if status == "unknown" and reliability in order:
        return max(reliability, "caution", key=order.index)
    return reliability
