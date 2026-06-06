"""Stable JSON output schema for P2Predict's CLIs.

When either CLI is invoked with ``--json``, all human-facing output is
suppressed and a single JSON document is emitted to stdout instead.
Stderr stays clean (no spinner, no logo). Exit code is 0 on success,
1 on error — including on errors, where a JSON-shaped error document
is still emitted to stdout so an agent that piped the output can parse
it instead of seeing a Rich-formatted abort message.

Schema versioning
-----------------
Every JSON document includes a top-level ``schema_version`` field
(currently "1.0"). When fields are added we can leave the version
alone; when fields are renamed or removed we bump the major number.
Tests in tests/test_json_output.py lock in the field names so this
doesn't drift accidentally.

Predict (``p2predict ... --json``)
----------------------------------
::

    {
      "schema_version": "1.0",
      "command": "predict",
      "model": {
        "path": str,
        "algorithm": str,
        "target": str,
        "version": str,            # the p2predict_version saved in the model
        "log_target": bool,
        "features": [str, ...]
      },
      "mode": "inline" | "batch" | "interactive",
      "predictions": [             # one entry per input row
        {
          "input": {feature: value, ...},
          "prediction": float
        },
        ...
      ],
      "interval": {                # present when --interval N was passed
        "coverage": float,         # e.g. 0.90
        "per_row": [
          {"low": float, "high": float, "prediction": float},
          ...
        ],
        "soft_warning": str | null # non-null when calibration is small
      },
      "explanation": [             # present when --explain was passed
        {                          # one entry per input row
          "baseline": float,
          "prediction": float,
          "log_target": bool,
          "contributions": [{"feature": str, "value": float}, ...],
          "multiplicative_factors":     # only for log-target models
              [{"feature": str, "factor": float}, ...] | null,
          "dollar_attribution":         # only for log-target models, labelled
                                        # approximate in the README
              [{"feature": str, "value": float}, ...] | null,
          "residual": float
        },
        ...
      ],
      "whatif": {                  # present when --whatif "Feature:NewVal,..." was passed
        "changes": {feature: {"from": value, "to": value}, ...},
        "base_prediction": float,
        "counterfactual_prediction": float,
        "delta": float,
        "delta_pct": float,
        "log_target": bool,
        "multiplicative_factor": float | null,
        "changed_contributions": [{"feature": str, "value": float}, ...],
        "interaction_contribution": float,
        "interaction_is_material": bool,
        "base_interval": {"low": float, "high": float} | null,
        "cf_interval": {"low": float, "high": float} | null
      },
      "batch": {                   # present in batch mode (-i)
        "csv_path": str,           # where predictions got written
        "n_rows": int
      }
    }

Train (``p2predict-train ... --json``)
--------------------------------------
::

    {
      "schema_version": "1.0",
      "command": "train",
      "input": {
        "csv_path": str,
        "rows_loaded": int,
        "rows_after_outlier_handling": int,
        "target": str
      },
      "mode": "auto" | "expert",
      "time_column": str | null,
      "outliers": {
        "target": {
          "policy": str,           # keep / warn / drop / winsorize
          "applied": str,          # the action that actually changed data
          "n_outliers": int,
          "n_total": int,
          "lower": float | null,
          "upper": float | null
        },
        "features": {
          "policy": str,
          "applied": str,
          "n_outliers_total": int,
          "per_column": {col: {"n_outliers": int,
                               "lower": float,
                               "upper": float}, ...}
        }
      },
      "low_info_features": {
        "no_information": [str, ...],
        "high_variation": [str, ...]
      },
      "features_selected": [str, ...],
      "algorithm_selected": str,
      "log_target": bool,
      "cv_scores": {algo: float, ...},   # auto-mode only
      "feature_importances": [
        {"feature": str, "importance": float},
        ...
      ],
      "evaluation": {
        "r2": float,
        "mae": float,
        "rmse": float,
        "residual_bias_p_value": float,
        "quality_label": "Excellent" | "Good" | "Needs Improvement"
      },
      "model_path": str | null,    # null if not saved (interactive declined)
      "report_path": str | null    # null unless --report PATH was passed
    }

Errors (any command, when --json is set)
----------------------------------------
::

    {
      "schema_version": "1.0",
      "command": "predict" | "train",
      "error": {
        "code": str,                # short identifier, e.g. "missing_input"
        "message": str              # human-readable description
      }
    }
"""

from __future__ import annotations

import json
import sys
from typing import Any

JSON_SCHEMA_VERSION = "1.0"


def emit(payload: dict[str, Any]) -> None:
    """Write the JSON payload to stdout. Single source of truth so the
    serialisation options stay consistent across the two CLIs."""
    json.dump(
        payload,
        sys.stdout,
        indent=2,
        default=_json_default,
        ensure_ascii=False,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_error(command: str, code: str, message: str, exit_code: int = 1) -> None:
    """Emit a JSON error document and exit non-zero.

    Used in place of ``console.print('Aborted: ...'); raise SystemExit(1)``
    when ``--json`` is active so callers piping stdout to ``jq`` (or an
    agent) still get a parseable document on failure.
    """
    emit({
        "schema_version": JSON_SCHEMA_VERSION,
        "command": command,
        "error": {"code": code, "message": message},
    })
    raise SystemExit(exit_code)


def _json_default(obj: Any) -> Any:
    """Coerce non-JSON-native types we routinely return from the model
    stack (numpy scalars, pandas/numpy timestamps, dataclasses) into
    plain Python so json.dump doesn't choke.

    Anything that survives this and still isn't serialisable will raise
    TypeError, which is what we want — better an explicit failure than a
    silent string-cast that lies about the field's type.
    """
    # numpy scalars
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover — numpy is a hard dep
        pass
    # pandas timestamps
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except ImportError:  # pragma: no cover — pandas is a hard dep
        pass
    # dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    raise TypeError(
        f"Cannot serialise object of type {type(obj).__name__} to JSON. "
        "Coerce it in the CLI before emitting."
    )
