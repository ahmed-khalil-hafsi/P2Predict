
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector
from xgboost import XGBRegressor

def start_training(X_train,y_train,numerical_cols, categorical_cols, algorithm):

    # Preprocessing for numerical data
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
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                 ])

    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)

    # Get model weights
    if algorithm == 'ridge':
        importance = model.coef_
    elif algorithm == 'xgboost':
        importance = model.feature_importances_
    elif algorithm == 'random_forest':
        importance = model.feature_importances_
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}') 
    
    feature_names = X_train.columns.tolist()
    feature_importances = zip(feature_names, importance)
    sorted_feature_importances = sorted(feature_importances, key = lambda x: abs(x[1]), reverse=True)

    return my_pipeline, sorted_feature_importances

