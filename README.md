# P2Predict
P2Predict is an open-source Python program for advanced procurement price prediction. It employs machine learning techniques to provide reliable and actionable insights into price trends, aiding in strategic decision-making in procurement.
The project is in heavy active development - Contributions are welcome!

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
To use P2Predict, you need the following steps:

1) Prepre the data for training
  - Make sure your data is in a CSV format
  - Make sure you don't have any blanks or gaps in the data (empty columns, empty cells, ...)
  - Make sure you don't have any erros in the data (#NAs and such)
  - Make sure you don't have any text in columns where a number is expected
  
2) Train a machine learning model
To train a new model, you have the use the tool `p2predict_train.py`. The tool accepts the following arguments:
 
Usage: p2predict_train.py [OPTIONS]

Options:
  --input PATH This is the path to your input CSV file
  --target TEXT This is the name of the feature you need to predict (i.e. Price)
  --algorithm TEXT Choose the machine learning algorithm to be used: <ridge, xgboost, or random_forest>
  --help            Show this message and exit.
  
Example:

```Python
python3 p2predict_train.py --input dummy/example.csv --target Price --algorithm ridge
```
This would train a model based on data found in `dummy/example.csv`, using the `ridge` algorithm and taking `Price` as a target feature (i.e. the model will predict prices)

3) Use the model to predict prices
To predict a new price using a model already trained, you have to use the tool `p2predict.py`. The tool accepts the following arguments:

Usage: p2predict.py [OPTIONS]

Options:
  --model PATH this is the path to the trained model
  --features TEXT this is a comma separated key:value list that has the input features 
  --help           Show this message and exit.


Example:

```Python
python3 p2predict.py --model models/ridge_weight_region_price.model --features weight_g:25,region:5
```

This would use the model saved in `models/ridge_weight_region_price.model` in order to predict the Price for an object with a weight_g of 25 and is in region 5.
Make sure that the model accepts exactly the same features in the right order. The model in this example has been trained using p2predict_train using weight_g and region as training features.

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


## Data
For data examples, check `dummy/example.csv`

## Contributing
We welcome contributions to P2Predict! Please feel free to open an issue or submit a pull request if you have a feature you'd like to add, or if you've found a bug.


