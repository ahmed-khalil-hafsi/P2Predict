# P2Predict Documentation

P2Predict is an open-source Python command-line program for advanced procurement price prediction. It uses machine learning techniques to provide reliable and actionable insights into price trends, aiding in strategic decision-making in procurement.

## Table of Contents
- [1. Installation](#1-installation)
- [2. Getting Started](#2-getting-started)
- [3. Usage](#3-usage)
- [4. License](#4-license)

## 1. Installation

Clone the P2Predict repository to your local machine. Since P2Predict is written in Python, you need to have Python installed to run the program. Please note that the P2Predict project is under heavy active development, and it is not yet released for production.

## 2. Getting Started

P2Predict comprises two main files:

1. `p2predict_train.py`: This file is used to train your machine learning model. The file accepts a CSV file containing your data and uses several machine learning algorithms to train a model.

2. `p2predict.py`: This file is used to make predictions with your trained model. The file accepts a trained model file and a CSV file containing new data to predict.

## 3. Usage

The sections below demonstrate how to train your model and make predictions with it.

### 3.1 Training Your Model

To train your model, run `p2predict_train.py` from the command line. 

You can run it with or without arguments. If you run the script without arguments, the program will guide you through the process interactively. If you run it with arguments, it will perform the task automatically. Below is the format to run the script with arguments:

```bash
python p2predict_train.py --input <CSV_FILE_PATH> --target <TARGET_FEATURE> --algorithm <TRAINING_ALGORITHM> --silent --training_features <TRAINING_FEATURES>
```

`--input`: Path to the CSV file used for training.
`--target`: The feature to predict.
`--algorithm`: The training algorithm to use. Possible values are 'ridge', 'xgboost', and 'random_forest'.
`--silent`: Run in silent mode without interactive prompts.
`--training_features`: The features used for training. List these as headers separated by a ':'.

Example:

```bash
python p2predict_train.py --input data.csv --target price --algorithm xgboost --silent --training_features weight:size
```
### 3.2 Making Predictions

To make predictions with your trained model, run `p2predict.py` from the command line.

Like `p2predict_train.py`, you can run `p2predict.py` with or without arguments.

```bash
python p2predict.py --model <MODEL_FILE_PATH> --features_inline <FEATURES_INLINE> --features_csv <CSV_FILE_PATH>
```
`--model`: Path to the trained model file (.model file).
`--features_inline`: List of features for the prediction. Format is 'feature1:value1,feature2:value2'.
`--features_csv`: Path to the CSV file that contains the features for the prediction.

Example:

```bash
python p2predict.py --model my_model.model --features_inline 'weight:150,size:30' --features_csv new_data.csv
```
### 4. License

P2Predict is released under the MIT license. See **LICENSE** for the license details.