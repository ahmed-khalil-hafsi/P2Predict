"""P2Predict — parametric price benchmarking for procurement and engineering.

This is the public Python API. The same functionality is also exposed via
the ``p2predict`` and ``p2predict-train`` console scripts (installed as
entry points), and via the upcoming MCP server. Pick the surface that fits
your workflow:

  • Python API (this module) — embed P2Predict in scripts, notebooks, or
    agent code.
  • CLI — interactive use, batch processing.
  • MCP — let an AI agent call P2Predict on behalf of a procurement user.

Typical Python usage::

    import pandas as pd
    from p2predict import auto_train, predict_interval, explain, load_model

    # Train (returns a fitted pipeline ready to predict / explain).
    data = pd.read_csv("purchases.csv")
    features = ["Weight", "Region", "Supplier", "Size"]
    model, info = auto_train(data, target="Price", features=features)

    # Save and reload.
    info.save("models/my_model.model")
    loaded = load_model("models/my_model.model")

    # Predict on a new part.
    new_part = pd.DataFrame([{"Weight": 15, "Region": "EU",
                              "Supplier": "A", "Size": "Standard"}])
    pred = loaded.predict(new_part)

    # Likely range + per-feature explanation, both axiomatically grounded.
    intervals = loaded.predict_interval(new_part, coverage=0.90)
    explanation = loaded.explain(new_part.iloc[[0]])
"""

from p2predict.explain import (
    Explanation,
    explain_batch,
    explain_row as explain,
    top_drivers,
)
from p2predict.intervals import IntervalResult, predict_interval
from p2predict.outliers import (
    POLICIES as OUTLIER_POLICIES,
    apply_feature_outlier_policy,
    apply_outlier_policy,
)
from p2predict.trained_model_io import (
    LoadModel as load_model,
    P2PREDICT_VERSION,
    SaveModel as save_model,
    Serialize_Trained_Model,
)
from p2predict.training import ALGORITHMS, auto_train, start_training
from p2predict.whatif import WhatIfResult, compute_whatif as what_if

__version__ = P2PREDICT_VERSION.lstrip("v")

__all__ = [
    # Training entry points.
    "auto_train",
    "start_training",
    "ALGORITHMS",
    # Prediction utilities (the fitted model itself has `.predict()`; we expose
    # the extras here).
    "predict_interval",
    "explain",
    "explain_batch",
    "top_drivers",
    "what_if",
    # Persistence.
    "save_model",
    "load_model",
    "Serialize_Trained_Model",
    # Outlier handling (also wired into the train CLI; exposed here so
    # programmatic users can run the same policy from Python).
    "apply_outlier_policy",
    "apply_feature_outlier_policy",
    "OUTLIER_POLICIES",
    # Result containers — useful for typed downstream code.
    "Explanation",
    "IntervalResult",
    "WhatIfResult",
    # Version (mirrors the persisted model metadata field).
    "P2PREDICT_VERSION",
    "__version__",
]
