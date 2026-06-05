"""End-to-end demonstration of the P2Predict Python API.

This is the surface an agent (or an embedded application) calls when it
wants to ask "what should this part cost?" without shelling out to the
CLI. Every function shown here has axiomatic tests in tests/test_*.py
that lock in the property it promises.

Run with::

    pip install -e .
    python examples/python_api.py
"""
from __future__ import annotations

import pandas as pd

import p2predict
from p2predict import (
    auto_train,
    explain,
    load_model,
    predict_interval,
    save_model,
    Serialize_Trained_Model,
    what_if,
)
from p2predict.intervals import compute_calibration_residuals
from p2predict.model_evals import evaluate_model
from p2predict.prepare_data import prepare_data


def main() -> None:
    print(f"P2Predict {p2predict.__version__}\n")

    # 1. Load training data. Any CSV with technical features + a price column.
    data = pd.read_csv("examples/example.csv")
    # The example dataset has a zero-variance column we don't want to train on.
    data = data.drop(columns=["Supplier"])
    features = ["Weight", "Region", "CPN", "Size"]

    # 2. Split, train. auto_train runs CV-based model selection across
    #    Ridge / RandomForest / XGBoost via HalvingRandomSearchCV.
    X_train, X_test, y_train, y_test, num, cat = prepare_data(data, features, "Price")
    model, algorithm, scores, log_target = auto_train(
        X_train, y_train, num, cat, budget="fast"
    )
    mae, r2, _, rmse = evaluate_model(X_test, y_test, model)
    print(f"Selected: {algorithm}  R²={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    print(f"All candidates: {dict((k, round(v, 3)) for k, v in scores.items())}\n")

    # 3. Compute conformal calibration on the test residuals so likely-range
    #    intervals are available at inference time.
    calibration = compute_calibration_residuals(model, X_test, y_test)

    # 4. Persist with full metadata. Background sample (for SHAP's
    #    LinearExplainer on linear models) and calibration both ride along.
    background = X_train.sample(n=min(100, len(X_train)), random_state=0)
    metadata = Serialize_Trained_Model(
        algorithm, features, "Price", model, r2,
        log_target=log_target,
        background_sample=background,
        calibration=calibration,
    )
    save_model(metadata, "/tmp/demo.model")

    # 5. Reload and inference. This is what an agent does mid-conversation.
    loaded = load_model("/tmp/demo.model")
    trained = loaded["model"]
    new_part = pd.DataFrame([{
        "Weight": 15, "Region": "EU", "CPN": "CP15-EXAMPLE", "Size": "Standard",
    }])
    point_estimate = float(trained.predict(new_part)[0])
    print(f"Predicted price: {point_estimate:.2f}")

    # 6. Likely range. Coverage is mathematically guaranteed under
    #    exchangeability — see modules/intervals.py for the proof sketch.
    [interval] = predict_interval(trained, new_part, loaded["calibration"], coverage=0.90)
    print(f"Likely range (90%): {interval.low:.2f} – {interval.high:.2f}")
    print("→ Quotes outside this range are unusual and worth questioning.\n")

    # 7. Per-feature SHAP attribution. Exact (TreeExplainer for trees,
    #    LinearExplainer for linear models; never KernelExplainer). Local
    #    accuracy axiom (baseline + sum(contribs) == prediction) holds.
    explanation = explain(trained, new_part, background_X=loaded["background_sample"])
    print("Per-feature attribution (SHAP):")
    print(f"  Baseline: {explanation.baseline:+.2f}")
    for feature, contribution in sorted(
        explanation.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
    ):
        print(f"  {feature:<10} {contribution:+.2f}")
    print(f"  Prediction (should ≈ sum above): {explanation.prediction:+.2f}\n")

    # 8. What-if counterfactual: change Region from EU to CN and see the
    #    impact on price + how the change attributes back to features.
    feature_types = {"Weight": "Numerical", "Region": "Categorical",
                     "CPN": "Categorical", "Size": "Categorical"}
    comparison = what_if(
        trained, new_part, {"Region": "CN"}, feature_types,
        background_X=loaded["background_sample"],
        calibration=loaded["calibration"],
        coverage=0.90,
    )
    print(f"What-if: change Region EU → CN")
    print(f"  Base price:           {comparison.base_prediction:.2f}")
    print(f"  Counterfactual:       {comparison.counterfactual_prediction:.2f}")
    print(f"  Delta:                {comparison.delta:+.2f} ({comparison.delta_pct:+.1f}%)")
    print(f"  Likely range shifts from "
          f"[{comparison.base_interval.low:.2f}, {comparison.base_interval.high:.2f}] to "
          f"[{comparison.cf_interval.low:.2f}, {comparison.cf_interval.high:.2f}]")


if __name__ == "__main__":
    main()
