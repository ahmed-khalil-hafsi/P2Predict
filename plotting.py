# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3


def plot_results_html(y_test, y_pred):
    # Scatter plot of predicted vs actual values
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.title('Scatter Plot: Actual vs Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

    # Histogram of residuals
    plt.subplot(1,2,2)
    residuals = y_test - y_pred
    sns.histplot(residuals)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')

    # Show the plots
    plt.tight_layout()

    # Residuals plot
    plt.figure()
    sns.residplot(x=y_pred, y=residuals, lowess=True, color='g')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')

    # Prediction Error Plot
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Price')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error vs Predicted')

    # Convert current figure to HTML
    plot_html = mpld3.fig_to_html(plt.gcf())

    # Save to a file
    with open("plots.html", "w") as f:
        f.write(plot_html)

    plt.close('all')  # close all open plots

def plot_results_console(y_test, y_pred):
    # Scatter plot of predicted vs actual values
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.title('Scatter Plot: Actual vs Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

    # Histogram of residuals
    plt.subplot(1,2,2)
    residuals = y_test - y_pred
    sns.histplot(residuals)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Residuals plot
    plt.figure()
    sns.residplot(x=y_pred, y=residuals, lowess=True, color='g')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')

    # Show the plot
    plt.show()

    # Prediction Error Plot
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Price')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error vs Predicted')

    # Show the plot
    plt.show()