import datetime
import joblib
import pandas as pd
import sklearn

from modules.input_checks import check_csv_sanity


def SaveModel(model_metadata, model_name):
    joblib.dump(model_metadata, model_name)

def LoadModel(model_file):
    model_metadata = joblib.load(model_file)
    return model_metadata

def Serialize_Trained_Model(algorithm, selected_columns, target_column, model, r2):
    model_metadata = {
    'model': model,  
    'features': selected_columns,
    'target_feature': target_column,
    'model_name': algorithm,
    'r2': str(r2),
    'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'scikit_learn_version': sklearn.__version__,
    'p2predict_version': 'v0.1beta'
    }
    return model_metadata

def load_csv_file(file):

    check_csv_sanity(file)

    return pd.read_csv(file)