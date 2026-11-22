import datetime
from pathlib import Path

import joblib
import sklearn

from p2predict.input_checks import check_csv_sanity

def _package_version() -> str:
    """The installed package version, from pyproject.toml via its metadata.

    This used to be a hard-coded string that fell behind the releases (it
    said v1.1.0 while pyproject said 1.3.0). An editable install only
    refreshes its metadata on reinstall, so a dev checkout can still lag
    until `pip install -e .` / `uv sync` runs again.
    """
    import importlib.metadata

    try:
        return "v" + importlib.metadata.version("p2predict")
    except importlib.metadata.PackageNotFoundError:
        return "v0.0.0+unknown"


# Stamped into every saved model as `p2predict_version`.
P2PREDICT_VERSION = _package_version()


def SaveModel(model_metadata, model_name):
    # joblib.dump doesn't create missing parent directories (e.g. a fresh
    # install's default "models/" folder), and would otherwise crash after
    # training already finished, losing the in-memory model.
    parent = Path(model_name).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_metadata, model_name)


def LoadModel(model_file):
    return joblib.load(model_file)


def Serialize_Trained_Model(
    algorithm,
    selected_columns,
    target_column,
    model,
    r2,
    log_target=False,
    background_sample=None,
    calibration=None,
    feature_domain=None,
):
    """Pack a trained model and provenance metadata.

    ``background_sample`` is a small (typically ~100-row) DataFrame of raw
    pre-preprocessor feature rows. It is required for SHAP's LinearExplainer
    on linear models (which needs it to estimate E[x_i]); it is ignored by
    TreeExplainer.

    ``calibration`` is the dict returned by
    ``p2predict.intervals.compute_calibration_residuals`` — the test-set
    residuals (in log space when log-target is active, target space
    otherwise) used by split-conformal to compute likely-range intervals.

    ``feature_domain`` is ``{column: {"min": float, "max": float}}`` over the
    numeric training features — the box the model was actually built in, used
    by ``p2predict.domain`` to tell a user when a part sits outside it. A few
    floats; no effect on prediction. Models saved without it fall back to
    approximating the range from ``background_sample``, which is a subset and
    so errs toward over-reporting rather than missing an out-of-domain part.

    All three fields are optional for backwards compatibility. Older models
    still load and predict; ``--explain`` and ``--interval`` refuse to
    run on models that lack the relevant field, with a helpful message.
    """
    return {
        "model": model,
        "features": list(selected_columns),
        "target_feature": target_column,
        "model_name": algorithm,
        "r2": str(r2),
        "log_target": bool(log_target),
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scikit_learn_version": sklearn.__version__,
        "p2predict_version": P2PREDICT_VERSION,
        "background_sample": background_sample,
        "calibration": calibration,
        "feature_domain": feature_domain,
    }


def load_csv_file(file):
    return check_csv_sanity(file)
