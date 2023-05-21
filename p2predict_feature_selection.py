
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def get_most_predictable_features_lasso(data, target_column):
    
    transformer = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        remainder='passthrough'
    )

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    transformer.fit(X)
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    X_transformed = transformer.transform(X)

    feature_selection = Lasso()
    feature_selection.fit(X_transformed, y)

    selected_features_mask = (feature_selection.coef_!=0).any(axis=0)

    onehot_features = transformer.named_transformers_['onehotencoder'].get_feature_names_out(categorical_columns)
    original_features = list(X.columns)

    # Build a mapping from transformed feature indices to original feature names
    feature_mapping = []
    for feature in original_features:
        if feature in categorical_columns:
            n_categories = sum([f.startswith(f'{feature}_') for f in onehot_features])
            feature_mapping.extend([feature]*n_categories)
        else:
            feature_mapping.append(feature)
            
    selected_features = [feature_mapping[i] for i in range(len(feature_mapping)) if selected_features_mask[i]]

    # removing duplicates
    selected_features = list(set(selected_features))

    return selected_features

def get_most_predictable_features(data, target_column):
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Preprocessing for numerical data: standard scaling
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data: one-hot encoding
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Define the model
    model = RandomForestRegressor(random_state=0)

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)])

    # Preprocessing of training data, fit model 
    my_pipeline.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_

    # Get the list of names from the one-hot encoder
    one_hot_features = my_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols)
    all_features = np.concatenate([numerical_cols, one_hot_features])

    # Create a dataframe of features and importances
    feature_importances = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances})

    # Sum the importances of one-hot encoded features and map them back to the original feature name
    feature_importances['OriginalFeature'] = feature_importances['Feature'].apply(lambda x: x.split('_')[0])
    feature_importances = feature_importances.groupby('OriginalFeature').sum()

    # Sort features by importance
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

    # Convert to percentages
    feature_importances['Importance'] = round(feature_importances['Importance'] / feature_importances['Importance'].sum() * 100)

    # Get the most important features
    selected_features = feature_importances.index.tolist()
    #print(feature_importances)

    return feature_importances
