# P2Predict

P2Predict is an open-source Python comand-line program for advanced procurement price prediction. It employs machine learning techniques to provide reliable and actionable insights into price trends, aiding in strategic decision-making in procurement. 

The project is in heavy active development - Contributions are welcome!

This project is not released for production yet.

## Features

#### Prediction
- Predict prices (or any other target feature) based on a trained model

#### Model Training
- Import training data from a CSV file
- Perform feature/impact analysis
- Train a machine learning model on the selected features to predict the price (or any other target)
- Show predicted vs actual prices
- Support for Ridge, XGBoost, and Random Forest ML algorithms
- Model Export & Import
- Calculation of evaluation metrics (mean absolute error and R^2 scores are supported)

## How to Use

To use P2Predict, follow these steps:

### 0. Install dependencies
   - Install the required dependencies listed in the `Dependencies` section.
   
### 1. Prepare the data for training
   - Ensure your data is in a CSV format.
   - Remove any blanks or gaps in the data (empty columns, empty cells, etc.).
   - Address any errors in the data (e.g., #NAs).
   - Verify that numeric columns do not contain text.

### 2. Train your model
   - Use the `p2predict_train.py` tool to train a new model.
   - The tool accepts the following arguments:

     ```bash
     python3 p2predict_train.py --input PATH --target TEXT --algorithm TEXT
     ```

     - `--input PATH`: Path to your input CSV file.
     - `--target TEXT`: Name of the feature to predict (e.g., "Price").
     - `--algorithm TEXT`: Choose the machine learning algorithm to be used: "ridge", "xgboost", or "random_forest".

     Example:

     ```bash
     python3 p2predict_train.py --input dummy/example.csv --target Price --algorithm ridge
     ```

     This command trains a model using data from `dummy/example.csv`, the `ridge` algorithm, and `Price` as the target feature.

### 3. Use the model to predict prices
   - Use the `p2predict.py` tool to predict a new price using a trained model.
   - The tool accepts the following arguments:

     ```bash
     python3 p2predict.py --model PATH --features TEXT
     ```

     - `--model PATH`: Path to the trained model.
     - `--features TEXT`: Comma-separated key:value list of input features.

     Example:

     ```bash
     python3 p2predict.py --model models/ridge_weight_region_price.model --features weight_g:25,region:5
     ```

     This command uses the model saved in `models/ridge_weight_region_price.model` to predict the price for an object with a weight of 25g and located in region 5. Make sure the model accepts the same features in the correct order. The model in this example was trained using `p2predict_train`, using `weight_g` and `region` as training features.

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

For data examples, check `dummy/example.csv`.

## Contributing

We welcome contributions to P2Predict! If you have a feature you'd like to add or if you've found a bug, please feel free to open an issue or submit a pull request.
