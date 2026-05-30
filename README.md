# P2Predict
     ____   ____   ____                   _  _        _   
    |  _ \ |___ \ |  _ \  _ __   ___   __| |(_)  ___ | |_ 
    | |_) |  __) || |_) || '__| / _ \ / _` || | / __|| __|
    |  __/  / __/ |  __/ | |   |  __/| (_| || || (__ | |_ 
    |_|    |_____||_|    |_|    \___| \__,_||_| \___| \__|


[![P2Predict_train](https://github.com/ahmed-khalil-hafsi/P2Predict/actions/workflows/p2predict_train.yml/badge.svg)](https://github.com/ahmed-khalil-hafsi/P2Predict/actions/workflows/p2predict_train.yml)

**P2Predict benchmarks what similar parts have historically cost, to inform design and sourcing decisions.**

You feed it a CSV of past purchases (technical features → price). It trains a model and lets you ask, *"given these features, what have parts like this cost?"* — turning historical data into a reality check engineering and procurement teams can use to discuss feature value, scope creep, and design trade-offs.

![User Experience Expert Mode](./documentation/p2predict_train.gif)

### What it is (and isn't)

P2Predict is a **parametric, data-driven cost-prediction tool** — the kind of model NASA, ICEAA, and the cost-estimating bodies call *parametric estimating*. It learns `features → price` from your historical data.

It is **not bottom-up should-costing.** It does not decompose parts into material cost + labor minutes + machine time + overhead. Tools like aPriori or Siemens Teamcenter PCM do that, from first principles. P2Predict answers a complementary question: *"what has the market actually charged us for parts like this?"* The two approaches work well together — one tells you what a part ought to cost, the other tells you what similar parts have cost.

### Who it's for

Procurement, sourcing, and engineering teams that want a shared, data-grounded view of cost when reviewing designs. Typical use cases:

- **Design reviews / VE/VA workshops** — "if we add this feature, what have similar parts with it cost historically?"
- **Scope and tech-debt discussions** — quantify the cost of features that are nice-to-have vs. essential.
- **Supplier benchmarking** — compare quoted prices against the model's prediction for the same spec.
- **RFQ sanity checks** — flag quotes that are far from what similar parts have cost.

This is a command-line tool aimed at fairly technical users (procurement engineers, commodity managers, cost engineers). It is not yet polished for non-technical business users.

This software is released under the MIT license. See `LICENSE` for the license details.

## How it works in one minute

1. **Bring your history.** A CSV of past purchases — one row per part, with technical features (weight, material, region, supplier, size, …) and the price you paid.
2. **Train a model.** P2Predict fits a regression model (Ridge / Random Forest / XGBoost), cross-validates them against each other, and keeps the best one.
3. **Ask "what would similar parts cost?"** Feed it the technical features of a new or proposed part and it returns a benchmark price grounded in your historical data.

The model learns from your data — so the benchmark reflects your supply base and your buying patterns, not a vendor's reference catalog.

## Features

#### Benchmarking / prediction
- Predict the benchmark price (or any other numerical target) for a part given its technical features
- Batch-predict an entire CSV of candidate parts in one call
- Robust to unseen categorical values at prediction time (a new supplier code or region won't crash the model)

#### Model Training
- Import training data from a CSV file
- **Auto-mode now performs cross-validated model selection** across Ridge, Random Forest and XGBoost using `HalvingRandomSearchCV` — it picks the best model *and* hyperparameters for your data, not just a default Random Forest
- **Automatic log-target transform** for positive, skewed targets (typical of price data) — often cuts MAE substantially
- Auto-detection of the most predictive features using a Random Forest baseline
- Auto-detection of low/no information features that might bias the model
- Tree models use `OrdinalEncoder` (fast, handles high-cardinality categoricals); linear models use `OneHotEncoder + StandardScaler`
- Expert mode lets you pick the algorithm and optionally run hyperparameter tuning — the tuned model is the one that gets saved
- Configurable HPO search budget via `--budget {fast,thorough}`
- Models can be easily saved and loaded
- Evaluation metrics: R², MAE, RMSE, and a residual-bias check

### Plotting
- Create a PDF file with model performance indicators (predicted vs actual price, distribution of prediction errors, ...)

![alt text](./documentation/model_perf_plot.png)

## Quick Start

To use P2Predict, follow these steps:

### 0. Install dependencies
   - Install the required dependencies by invoking `pip install -r requirements.txt`
   
### 1. Prepare the data for training
   - Ensure your data is in a CSV format.
   - Remove any blanks or gaps in the data (empty columns, empty cells, etc.).
   - Address any errors in the data (e.g., #NAs).
   - Verify that numeric columns do not contain text.

### 2. Train your model
   
   - Use the `p2predict_train.py` tool to train a new model.
   - The tool accepts the following arguments:

     ```bash
     python3 p2predict_train.py [OPTIONS]
     ```

     - `--input`, `-i`: Path to your input CSV file. This dataset is used for training.
     - `--target`, `-t`: Name of the feature to predict (e.g., "Price").
     - `--expert`, `-x`: Toggle Expert Mode. In Expert mode, you have more control over the training process.
     - `--algorithm`, `-a`: Choose the algorithm in expert mode: `ridge`, `xgboost`, or `random_forest`.
     - `--interactive`, `-c`: Activate interactive mode. If not set, you must specify all required options.
     - `--verbose`, `-v`: Increase output verbosity.
     - `--training_features`, `-tf`: List of training features to be used, separated by commas.
     - `--budget`, `-b`: HPO search budget — `fast` (default) or `thorough`. Used by auto-mode model selection and by expert-mode `--tune`.
     - `--tune / --no-tune`: Expert mode only. Run hyperparameter tuning and save the tuned model.

     For a complete list of options, run `python3 p2predict_train.py --help`.

     ### Examples:

     #### Example 1: Interactive Auto-Mode

     ```bash
     python3 p2predict_train.py --interactive
     ```

     Launches P2Predict in Interactive Auto-Mode. The program guides you through the process and automatically performs CV-based model selection across Ridge, Random Forest, and XGBoost.

     #### Example 2: Non-Interactive Auto-Mode

     ```bash
     python3 p2predict_train.py --input examples/example.csv --target Price
     ```

     Auto-mode picks the best algorithm and hyperparameters for you. Output names the winning algorithm and lists CV R² for each candidate.

     #### Example 3: Auto-Mode with Thorough HPO

     ```bash
     python3 p2predict_train.py --input data/sales.csv --target Revenue --budget thorough
     ```

     Same as Example 2 but with a wider hyperparameter search (slower, usually higher accuracy).

     #### Example 4: Interactive Expert Mode
     
     ```bash
     python3 p2predict_train.py --expert --interactive
     ```

     Interactive Expert Mode gives you control over feature selection, algorithm choice, and HPO. You're prompted before running hyperparameter tuning, and the tuned model is what gets saved.

     #### Example 5: Non-Interactive Expert Mode with Tuned XGBoost
     ```bash
     python3 p2predict_train.py --expert --input examples/example.csv --algorithm xgboost --target Price --training_features Weight,Size,Region --tune --budget fast
     ```

     Trains an XGBoost model on the chosen features, runs `HalvingRandomSearchCV`, and saves the tuned model.

     #### Example 6: Specifying Multiple Training Features
     ```bash
     python3 p2predict_train.py --expert --input data/housing.csv --target Price --algorithm ridge --training_features Area,Bedrooms,Location,YearBuilt
     ```

     Trains a Ridge regression model with the specified features.

   - After training, the model will be saved and can be used for predictions with `p2predict.py`.

      
### 3. Use the model to predict prices
   - Use the `p2predict.py` tool to predict a new price using a trained model.
   - The tool accepts the following arguments:

     ```bash
     python3 p2predict.py -m MODEL_PATH [-p PREDICT_USING] [-i PREDICT_FILE]
     ```

     - `-m, --model MODEL_PATH`: Path to the trained model file (.model file).
     - `-p, --predict_using TEXT`: Inline prediction feature/value pair to be fed to the trained model.
     - `-i, --predict_file FILE`: A CSV file that contains prediction features and values. This file will be fed to the trained model to generate predictions.

     Examples:

     1. Using inline prediction:
     ```bash
     python3 p2predict.py -m models/my_trained_model.model -p "weight_g:25,region:5"
     ```

     This command uses the model saved in `models/my_trained_model.model` to predict the price for an object with a weight of 25g and located in region 5.

     2. Using a prediction file:
     ```bash
     python3 p2predict.py -m models/my_trained_model.model -i prediction_data.csv
     ```

     This command uses the model saved in `models/my_trained_model.model` to generate predictions for all the entries in the `prediction_data.csv` file.

   Note: Make sure the features you provide (either inline or in the CSV file) match the features the model was trained on. The model in these examples was trained using `p2predict_train.py`, and the prediction features should correspond to the training features used.

## What's new in v0.2

- **Auto-mode does real model selection.** Previously it always used a default Random Forest and threw away the tuning step. It now runs `HalvingRandomSearchCV` over Ridge, Random Forest and XGBoost and picks the best.
- **Hyperparameter tuning in expert mode now replaces the saved model** instead of just printing scores.
- **Automatic log-target transform** when the target is positive and skewed (typical for price data).
- Tree models use `OrdinalEncoder` (no more one-hot blow-up on high-cardinality categoricals).
- Fixed feature-importance grouping bug (previously misgrouped names like `weight_g`).
- Fixed broken statistical check in evaluation (`evaluate_model` used to run the wrong t-test). Now reports R², MAE, RMSE and a residual-bias check.
- Saved models include a timestamp in the filename; no more random suffix collisions or `None_*.model`.
- Faster CSV sanity check; warns on NA values and drops them instead of aborting.
- `--budget {fast,thorough}` flag controls HPO search size.

> **Breaking change:** v0.2 changes the pipeline structure (new preprocessor and optional `TransformedTargetRegressor` wrap). Models trained on v0.1 will not load — retrain them. The metadata format now also includes a `log_target` field.

## Dependencies

Check `requirements.txt` for exact versions. Install with `pip install -r requirements.txt`
- click
- joblib
- matplotlib
- mpld3
- numpy
- pandas
- rich
- scikit-learn (≥ 1.5)
- scipy
- seaborn
- xgboost
- halo
- questionary

## Data

A small example dataset is in [`examples/example.csv`](examples/example.csv). The columns are the typical shape P2Predict expects: a few technical features per row (weight, region, supplier, size, …) and a price column.

## Contributing

I warmly welcome contributions to this project! If you have an exciting feature idea or have discovered a bug, I'd be delighted to hear from you. Please don't hesitate to open an issue or submit a pull request.

I'm particularly keen on expanding my collection of datasets for various direct and indirect commodities. This includes, but is not limited to, Integrated Circuits (ICs), Passive Components, Plastic Parts, and Mechanical Parts. If you're aware of any open datasets in these areas, or if your organization would like to contribute a dataset, I'd be incredibly grateful. Please reach out to me - your input could be invaluable in enhancing the project's capabilities and benefiting the wider procurement community.

## Become a sponsor!

I work on open source projects during my free time. If you think this projects adds value to the procurement community, please consider sponsoring a donation!
