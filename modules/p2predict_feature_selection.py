
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

def calculate_entropy(series):
    counts = series.value_counts(normalize=True)
    return -np.sum(counts*np.log2(counts))

def find_high_variation_features(df):
    # For numerical data
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Coefficient of variation = standard deviation / mean
    coefficients_of_variation = (numeric_df.std() / numeric_df.mean()).dropna()

    # Features with high variation might be considered those with CV > 1 
    high_variation_numeric = coefficients_of_variation[coefficients_of_variation > 1].index.tolist()

    # For categorical data
    categorical_df = df.select_dtypes(include=['object', 'bool', 'category'])

    # Calculate entropy for each column
    unique_ratio = categorical_df.apply(lambda x: x.nunique() / len(x))
    print(unique_ratio)

    # Consider high variation columns with very high unique values
    high_variation_categorical = unique_ratio[unique_ratio > 0.9].index.tolist()
    
    
    # Combine both lists
    high_variation_features = high_variation_numeric + high_variation_categorical
    
    return high_variation_features


def find_no_variation_features(df):
    
    unique_counts = df.nunique()
    no_variation_features = unique_counts[unique_counts == 1].index.tolist()

    return no_variation_features

def get_most_predictable_features_lasso(data, target_column):

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

    X_transformed = preprocessor.fit_transform(X)

    feature_selection = Lasso()
    feature_selection.fit(X_transformed, y)

    selected_features_mask = feature_selection.coef_ != 0
    print(selected_features_mask)

    onehot_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    original_features = list(X.columns)

    # Build a mapping from transformed feature indices to original feature names
    feature_mapping = []
    for feature in original_features:
        if feature in categorical_cols:
            n_categories = sum([f.startswith(feature) for f in onehot_features])
            feature_mapping.extend([feature] * n_categories)
        else:
            feature_mapping.append(feature)

    selected_features = [feature_mapping[i] for i in range(len(feature_mapping)) if selected_features_mask[i]]
    

    # Removing duplicates
    selected_features = list(set(selected_features))

    return selected_features


def get_most_predictable_features(data, target_column,output_only_headers=False):
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

    # Use a shallow random forest to estimate feature importance
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

    # Check if there are more features than importances
    if len(all_features) > len(importances):
        print("Warning: Number of features does not match the number of importances")
        return None

    # Create a dataframe of features and importances
    feature_importances = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances})

    
    # Sum the importances of one-hot encoded features and map them back to the original feature name
    feature_map = feature_importances.groupby(feature_importances['Feature'].apply(lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else x))['Importance'].sum().to_dict()


    # Map summed importances back to the original feature name
    feature_importances['OriginalFeature'] = feature_importances['Feature'].apply(lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else x)
    feature_importances['Importance'] = feature_importances['OriginalFeature'].map(feature_map)

    # Sort features by importance
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

    feature_importances.drop_duplicates(subset=['OriginalFeature', 'Importance'], keep='first', inplace=True)
    feature_importances.drop('Feature', axis=1, inplace=True)
    feature_importances.rename(columns={'OriginalFeature': 'Feature'}, inplace=True)

    if output_only_headers:
        return feature_importances['Feature']
    else:
        feature_importances['Importance'] = round(feature_importances['Importance'] / feature_importances['Importance'].sum() * 100, 2)
        feature_importances = feature_importances[['Feature','Importance']]
        feature_importances.rename(columns={'Importance':'Importance (%)'}, inplace=True)
        return feature_importances


# TODO fix the RFE
def get_most_predictable_features_RFE(data, target_column, n_features_to_select=10):
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

    # Use a shallow random forest to estimate feature importance
    model = RandomForestRegressor(random_state=0)

    # Define the RFE
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)

    # Bundle preprocessing and feature selection in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('feature_selection', rfe)])

    # Preprocessing of training data, fit model 
    my_pipeline.fit(X, y)

    selected = my_pipeline.named_steps['feature_selection'].support_

    selected_num_features = numerical_cols[selected[:len(numerical_cols)]]
    selected_cat_features = categorical_cols[selected[len(numerical_cols):]]
    selected_features = np.concatenate([selected_num_features, selected_cat_features])
    original_feature_names = [feature.split('_')[0] if '_' in feature else feature for feature in selected_features]

    return original_feature_names
    
