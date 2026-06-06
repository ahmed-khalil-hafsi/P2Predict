import datetime

import joblib
import sklearn

from modules.input_checks import check_csv_sanity

# v0.6 adds --whatif but does not change the persisted metadata schema —
# what-if uses only fields already present in v0.5+ models (background_sample
# for SHAP, calibration for likely-range intervals). Bumped purely so saved
# models reflect the runtime that produced them.
P2PREDICT_VERSION = "v0.6"


def SaveModel(model_metadata, model_name):
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
):
    """Pack a trained model and provenance metadata.

    ``background_sample`` is a small (typically ~100-row) DataFrame of raw
    pre-preprocessor feature rows. It is required for SHAP's LinearExplainer
    on linear models (which needs it to estimate E[x_i]); it is ignored by
    TreeExplainer.

    ``calibration`` is the dict returned by
    ``modules.intervals.compute_calibration_residuals`` — the test-set
    residuals (in log space when log-target is active, target space
    otherwise) used by split-conformal to compute likely-range intervals.

    Both fields are optional for backwards compatibility. Older models
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
    }


def load_csv_file(file):
    return check_csv_sanity(file)
