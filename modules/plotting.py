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

def plot_histograms(df):
    n = len(df.columns)
    fig, axs = plt.subplots(1, n, figsize=(n*5, 4), constrained_layout=True)

    sns.set_style('dark')
    for i, column in enumerate(df.columns):
        sns.histplot(df[column], color='darkblue', bins=30, ax=axs[i])
        # sns.kdeplot(df[column], color="darkblue", lw=2, ax=axs[i])  # plot kde on the i-th subplot
        axs[i].set_title('Distribution of ' + column, fontsize=12) 
        axs[i].set_xlabel(column, fontsize=10)  # set x-label
        axs[i].set_ylabel('Frequency', fontsize=10)  # set y-label

    
    plt.show()


def plot_results_pdf(y_test, y_pred,filename):
    with PdfPages(filename) as pdf:
        # Scatter plot of predicted vs actual values
        fig = plt.figure(figsize=(14,6))
        plt.suptitle('Procurement Price Prediction Performance (1/2)')
        plt.subplot(1,2,1)

        plt.scatter(y_test, y_pred, edgecolors='r', facecolors='none')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue')
        plt.title('Predicted vs Actual Procurement Prices')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.grid(True)

        # Histogram of residuals
        plt.subplot(1,2,2)
        residuals = y_test - y_pred
        sns.histplot(residuals, bins=20, color='skyblue', kde=True)
        plt.title('Price Prediction Deviations')
        plt.xlabel('Deviation: Actual Price - Predicted Price')
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Residuals plot
        plt.figure(figsize=(14,6))
        plt.suptitle('Procurement Price Prediction Performance (2/2)')
        plt.subplot(1,2,1)
        sns.residplot(x=y_pred, y=residuals, lowess=True, color='g', scatter_kws={'s': 5})
        plt.title('Deviation vs Predicted Procurement Price')
        plt.xlabel('Predicted Procurement Price')
        plt.ylabel('Price Deviation')
        plt.axhline(0, color='red', linestyle='--')
        plt.grid(True)

        # Prediction Error Plot
        plt.subplot(1,2,2)
        plt.scatter(y_pred, residuals, edgecolors='r', facecolors='none') 
        plt.xlabel('Predicted Procurement Price')
        plt.ylabel('Prediction Deviation')
        plt.title('Prediction Deviation vs Predicted Procurement Price')
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()
