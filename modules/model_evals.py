import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_model(X_test, y_test, model):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Replace the prior (and incorrect) two-sample t-test with a
    # residual-mean test: under a well-calibrated model, residuals should be
    # centered on zero. Reject => systematic bias.
    residuals = np.asarray(y_test) - np.asarray(predictions)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    n = len(residuals)
    if n > 1 and residuals.std(ddof=1) > 0:
        from scipy import stats
        _, p_value = stats.ttest_1samp(residuals, 0.0)
    else:
        p_value = 1.0

    return mae, r2, p_value, rmse


def get_column_statistics(data, feature_columns):
    return {
        col: {"skewness": data[col].skew(), "kurtosis": data[col].kurt()}
        for col in feature_columns
    }


def calculate_feature_importance(X, y, model):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=0, n_jobs=-1)
    total = sum(result.importances_mean) or 1.0
    return result.importances_mean / total
