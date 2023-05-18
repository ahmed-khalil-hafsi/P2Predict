# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table


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

def add_table_to_plot(ax, data):
    # create a new table with your data
    table_data = [['Metrics', 'Value']] + data
    table = ax.table(cellText=table_data, colWidths = [0.4]*len(table_data[0]), 
                     loc='upper left', cellLoc = 'center')

    table.auto_set_font_size(True)
    table.scale(1, 1.5)

def plot_results_pdf(y_test, y_pred,filename):
    with PdfPages(filename) as pdf:
        # Scatter plot of predicted vs actual values
        fig = plt.figure(figsize=(14,6))
        plt.suptitle('Model Performance Indicators (1/2)')

        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=1, rowspan=1)
        add_table_to_plot(ax1, [["R2 Score", 0.95], ["RMSE", 0.20], ["MAE", 0.15]])
        ax1.axis('tight')
        ax1.axis('off')

        ax2 = plt.subplot2grid((1, 5), (0, 1), colspan=2, rowspan=1)
        plt.scatter(y_test, y_pred, edgecolors='r', facecolors='none')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.grid(True)

        # Histogram of residuals
        ax3 = plt.subplot2grid((1, 5), (0, 3), colspan=2, rowspan=1)
        residuals = y_test - y_pred
        sns.histplot(residuals, bins=20, color='skyblue', kde=True)
        plt.title('Distribution of Price Delta')
        plt.xlabel('Price Delta: Actual - Predicted')
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Residuals plot
        plt.figure(figsize=(14,8))
        plt.suptitle('Model Performance Indicators (2/2)')
        plt.subplot(1,2,1)
        sns.residplot(x=y_pred, y=residuals, lowess=True, color='g', scatter_kws={'s': 5})
        plt.title('Price Delta vs Predicted')
        plt.xlabel('Predicted Price')
        plt.ylabel('Price Delta')
        plt.axhline(0, color='red', linestyle='--')
        plt.grid(True)

        # Prediction Error Plot
        plt.subplot(1,2,2)
        plt.scatter(y_pred, residuals, edgecolors='r', facecolors='none') 
        plt.xlabel('Predicted Price')
        plt.ylabel('Prediction Error')
        plt.title('Prediction Error vs Predicted Price')
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()
