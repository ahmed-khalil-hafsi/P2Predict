# P2Predict
P2Predict is An open-source Python package for advanced procurement price prediction. It employs sophisticated regression analysis techniques to provide reliable and actionable insights into price trends, aiding in strategic decision-making in procurement.
The project is in active development. Contributions are welcome!

## Features
- Import data from a CSV file
- Select relevant columns for analysis
- Perform feature/impact analysis for the target variable (default: price)
- Train a regressor on the selected features to predict the target variable
- Display a scatter plot of predicted vs actual prices
- Support for Ridge, XGBoost, and Random Forest regression algorithms
- Display feature importances
- Model serialization for later use

## How to use
To use P2Predict, simply clone the repository and run the training script with the desired options. 

```Python
python p2predict_train.py --file <CSV file path> --target <target column name> --algorithm <ridge, xgboost, or random_forest>
```

## Dependencies
- pandas
- sklearn
- xgboost
- matplotlib
- seaborn
- joblib
- art
- rich
- click

## Example
```Python
python main.py --file data.csv --target price --algorithm ridge
```

This command will load data from `data.csv`, use `price` as the target column, and apply Ridge regression.

## Contributing
We welcome contributions to P2Predict! Please feel free to open an issue or submit a pull request if you have a feature you'd like to add, or if you've found a bug.


