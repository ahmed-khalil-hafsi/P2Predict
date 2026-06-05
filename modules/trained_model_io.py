import datetime

import joblib
import sklearn

from modules.input_checks import check_csv_sanity

# Bumped for v0.4 because the metadata gains an optional ``background_sample``
# field used by SHAP's LinearExplainer (and any future model-agnostic
# explainer). v0.3 and v0.2 models still load — the extra field is optional
# and the rest of the schema is unchanged.
P2PREDICT_VERSION = "v0.4"


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
):
    """Pack a trained model and provenance metadata.

    ``background_sample`` is a small (typically ~100-row) DataFrame of raw
    pre-preprocessor feature rows. It is required for SHAP's LinearExplainer
    on linear models (which needs it to estimate E[x_i]); it is ignored by
    TreeExplainer. Optional for backwards compatibility — older models
    without it still load and predict, but ``--explain`` for linear models
    will refuse to run on them.
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
    }


def load_csv_file(file):
    return check_csv_sanity(file)
