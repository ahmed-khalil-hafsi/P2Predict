from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


def evaluate_model(X_test, y_test, model):
    from scipy import stats

    # Compute predictions on the test dataset   
    predictions = model.predict(X_test)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Calculate p-value
    _, p_value = stats.ttest_ind(y_test, predictions)
    
    return mae, r2, p_value

def get_column_statistics(data,feature_columns):
    stats = {}
    for i in feature_columns:
        skewness = data[i].skew()
        kurtosis = data[i].kurt()
        stats[i] = {'skewness':skewness,'kurtosis': kurtosis}
    return stats

def calculate_feature_importance(X,y,model):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    importance_normalized = result.importances_mean / sum(result.importances_mean)
    return importance_normalized