# TODO
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def hyper_parameter_tuning(my_pipeline,models,X_train,y_train):
        grid_param = [
            [{"model": [Ridge()],
              "model__alpha": [0.5, 1, 2],
              # Add other parameters here
              }],
            [{"model": [XGBRegressor(objective ='reg:squarederror')],
              "model__n_estimators": [100, 200],
              # Add other parameters here
              }],
            [{"model": [RandomForestRegressor(random_state=0)],
              "model__n_estimators": [100, 200],
              # Add other parameters here
              }]
    ]
        
        for i in range(len(models)):
        # Bundle preprocessing and modeling code in a pipeline
            my_pipeline = Pipeline(steps=[('preprocessor', preprocessing),
                                          ('model', models[i][1])
                                       ])

            gd_sr = GridSearchCV(estimator=my_pipeline,
                                 param_grid=grid_param[i],
                                 scoring='accuracy',
                                 cv=5,
                                 n_jobs=-1)

            gd_sr.fit(X_train, y_train)

            best_parameters = gd_sr.best_params_
            best_result = gd_sr.best_score_

            #console.print(f"Model: {models[i][0]}, Best Parameters: {best_parameters}, Best Score: {best_result}")
