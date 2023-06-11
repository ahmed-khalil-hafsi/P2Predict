
# ML
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# UI
from rich.console import Console

from modules.training import get_available_models

console = Console()

# TODO finish the hyper param tuning algo
def hyper_parameter_tuning(X_train,y_train,numerical_cols,categorical_cols):
        
  models = get_available_models()
  grid_param = get_hyper_parameters()
  
      # Preprocessing for numerical data
  numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
  categorical_transformer = Pipeline(steps=[
      ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))
  ])

    # Bundle preprocessing for numerical and categorical data
  preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        
  for i in range(len(models)):
      model_name, model = models[i]

      # Bundle preprocessing and modeling code in a pipeline
      my_pipeline = Pipeline(steps=[
          ('preprocessor', preprocessor),
          ('model', model)
      ])

      gd_sr = GridSearchCV(estimator=my_pipeline,
                           param_grid=grid_param[i],
                           scoring='r2',
                           cv=3,
                           n_jobs=-1,
                          )

      gd_sr.fit(X_train, y_train)

      best_parameters = gd_sr.best_params_
      best_score = gd_sr.best_score_

      console.print(f"Model: {model_name} --> Best R^2: {round(best_score,2)}")

def get_hyper_parameters():
    grid_param = [
    {
      'model': [Ridge()],
      'model__alpha': [0.5, 1, 2]
    },
    {
      'model': [XGBRegressor(objective='reg:squarederror')],
      'model__n_estimators': [100, 200]
    },
    {
      'model': [RandomForestRegressor(random_state=0)],
      'model__n_estimators': [100, 200,500]
    }
    ]
      
    return grid_param



