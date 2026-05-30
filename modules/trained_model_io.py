import datetime

import joblib
import sklearn

from modules.input_checks import check_csv_sanity

P2PREDICT_VERSION = "v0.2"


def SaveModel(model_metadata, model_name):
    joblib.dump(model_metadata, model_name)


def LoadModel(model_file):
    return joblib.load(model_file)


def Serialize_Trained_Model(
    algorithm, selected_columns, target_column, model, r2, log_target=False
):
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
    }


def load_csv_file(file):
    return check_csv_sanity(file)
