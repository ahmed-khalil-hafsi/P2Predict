
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector
from xgboost import XGBRegressor

def get_available_models():
    all_models = [
  ('ridge', Ridge(alpha=1.0)),
  ('xgboost', XGBRegressor(objective='reg:squarederror')),
  ('random_forest', RandomForestRegressor(n_estimators=100, random_state=0))
    ]

    return all_models

def get_available_models_map():
    all_models_map = {
    'ridge': Ridge(alpha=1.0),
    'xgboost': XGBRegressor(objective='reg:squarederror'),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=0)
    }
    return all_models_map
    

def auto_train(X_train,y_train,numerical_cols,categorical_cols):
    # add a heuristic to select features for auto-training

    preprocessor = Preprocess_data(numerical_cols, categorical_cols)

    algorithm = 'random_forest'

    model = get_available_models_map(algorithm)

    p2predict_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                 ])
    
    p2predict_pipeline.fit(X_train, y_train)
    
    return p2predict_pipeline

def start_training(X_train,y_train,numerical_cols, categorical_cols, algorithm):

    # Preprocessing for numerical data
    preprocessor = Preprocess_data(numerical_cols, categorical_cols)

    # Define the model
    if algorithm == 'ridge':
        model = Ridge(alpha=1.0)
    elif algorithm == 'xgboost':
        model = XGBRegressor(objective ='reg:squarederror')
    elif algorithm == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=0)
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

    # Bundle preprocessing and modeling code in a pipeline
    p2predict_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                 ])

    # Train the pipeline
    p2predict_pipeline.fit(X_train, y_train)

    # Get model weights
    importance = get_weights(algorithm, model)     
    sorted_feature_importances = map_features_to_importances(X_train, importance)

    return p2predict_pipeline, sorted_feature_importances

def Preprocess_data(numerical_cols, categorical_cols):
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
    return preprocessor

def map_features_to_importances(X_train, importance):
    feature_names = X_train.columns.tolist()
    feature_importances = zip(feature_names, importance)
    sorted_feature_importances = sorted(feature_importances, key = lambda x: abs(x[1]), reverse=True)
    return sorted_feature_importances

def get_weights(algorithm, model):
    if algorithm == 'ridge':
        importance = model.coef_
    elif algorithm == 'xgboost':
        importance = model.feature_importances_
    elif algorithm == 'random_forest':
        importance = model.feature_importances_
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    return importance

