"""Independent verification of the README's noise-floor findings.

Re-derives every load-bearing number in the case-study README and runs the
experiments that test whether the modest model quality is truly a property of
the data (the README's claim) or partly a property of how P2Predict trains.

Experiments
-----------
E1  Verify the saved README model's holdout metrics (raw R2, log R2,
    median %err, MAE, RMSE) on the reconstructed split.
E2  Complete-case XGBoost (what P2Predict effectively trains on after
    check_csv_sanity drops every row with any NA) vs the same XGBoost fed
    ALL rows with native-NaN handling — same 10 features, same holdout.
E3  Hyperparameter/model selection scored in raw price space (current core
    behaviour: scoring='r2' on the TransformedTargetRegressor) vs scored in
    log space. Same search space, same split.
E4  Extended features: rebuild the dataset from the raw PUB LOG extract with
    8 additional well-covered characteristics + ITEM_NAME, retrain, and
    re-measure BOTH the model and the noise ceiling on the wider signature.
E5  Per-signature median aggregation (the README Part 2b claim).
E6  Leave-one-out group-mean predictor on duplicate-signature rows — a
    model-free reality check on the 0.60 ceiling.

Run:  python verify_findings.py            (all experiments)
      python verify_findings.py --quick    (skip E3/E4 searches)
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA = HERE / "data"
sys.path.insert(0, str(REPO_ROOT / "src"))

TARGET = "unit_price_each_usd"
BASE_FEATURES = [
    "material", "head_style", "thread_diameter_in", "length_in",
    "thread_class", "thread_series", "finish", "tensile_strength_psi",
    "threads_per_inch", "width_across_flats_in",
]

# Extra PUB LOG characteristics (REQUIREMENTS_STATEMENT -> column, kind).
# Chosen purely by coverage among FSC-5306 NIINs; nothing exotic.
EXTRA_SPECS = {
    "THREAD LENGTH":                        ("thread_length_in", "num"),
    "HEAD HEIGHT":                          ("head_height_in", "num"),
    "GRIP LENGTH":                          ("grip_length_in", "num"),
    "GRIP DIAMETER":                        ("grip_diameter_in", "num"),
    "HEAD DIAMETER":                        ("head_diameter_in", "num"),
    "THREAD DIRECTION":                     ("thread_direction", "cat"),
    "HARDNESS RATING":                      ("hardness", "cat"),
    "MATERIAL DOCUMENT AND CLASSIFICATION": ("material_doc", "cat"),
    "SPECIFICATION/STANDARD DATA":          ("spec_standard", "cat"),
}

XGB_FIXED = dict(
    objective="reg:squarederror", tree_method="hist", n_estimators=600,
    max_depth=8, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
    min_child_weight=5, n_jobs=-1, random_state=0, verbosity=0,
)


def _load_case_prepare():
    spec = importlib.util.spec_from_file_location("case_prepare", HERE / "prepare_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pct = np.abs(y_pred - y_true) / y_true
    out = {
        "raw_R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "median_%err": float(np.median(pct) * 100),
        "P90_%err": float(np.quantile(pct, 0.90) * 100),
    }
    if (y_pred > 0).all():
        out["log_R2"] = r2_score(np.log(y_true), np.log(y_pred))
    else:
        clipped = np.clip(y_pred, 1e-6, None)
        out["log_R2"] = r2_score(np.log(y_true), np.log(clipped))
    return out


def show(label: str, m: dict) -> None:
    print(f"  {label:<58} logR2 {m['log_R2']:.3f}  rawR2 {m['raw_R2']:.3f}  "
          f"med%err {m['median_%err']:.1f}  MAE ${m['MAE']:.2f}")


def encode_for_xgb(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str]):
    """Ordinal-encode categoricals the way p2predict does for trees (NaN-safe)."""
    train = train.copy()
    test = test.copy()
    for c in cat_cols:
        levels = pd.Index(train[c].astype("string").unique()).dropna()
        mapping = {v: i for i, v in enumerate(levels)}
        train[c] = train[c].astype("string").map(mapping).astype(float)
        test[c] = test[c].astype("string").map(mapping).astype(float)  # unseen -> NaN
    return train, test


def fit_xgb_log(X_train, y_train, X_test, **overrides):
    params = {**XGB_FIXED, **overrides}
    model = XGBRegressor(**params)
    model.fit(X_train, np.log(y_train))
    return np.exp(model.predict(X_test))


# --------------------------------------------------------------------------- #
def e1_verify_saved_model(df_all: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("E1 — saved README model on the reconstructed holdout")
    print("=" * 78)
    from p2predict import load_model

    cc = df_all.dropna().reset_index(drop=True)  # check_csv_sanity equivalent
    X = cc[BASE_FEATURES]
    y = cc[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"  complete-case rows {len(cc):,}  (train {len(X_tr):,} / test {len(X_te):,})"
          f"  — full cleaned file has {len(df_all):,} rows; "
          f"{(1 - len(cc)/len(df_all))*100:.0f}% dropped by the NA policy")

    for path in sorted((REPO_ROOT / "models").glob("*_unit_price_each_usd_*.model")):
        loaded = load_model(path)
        preds = loaded["model"].predict(X_te)
        show(path.name[:56], metrics(y_te, preds))


def e2_nan_handling(df_all: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("E2 — complete-case (current core behaviour) vs native-NaN, all rows")
    print("=" * 78)
    cat = [c for c in BASE_FEATURES if df_all[c].dtype == object]

    df = df_all.reset_index(drop=True)
    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=0)
    train, test = df.loc[tr_idx], df.loc[te_idx]
    train_cc = train.dropna()
    test_cc = test.dropna()

    Xtr_all, Xte_all = encode_for_xgb(train[BASE_FEATURES], test[BASE_FEATURES], cat)
    Xtr_cc, Xte_cc = encode_for_xgb(train_cc[BASE_FEATURES], test_cc[BASE_FEATURES], cat)
    _, Xte_cc_from_all = encode_for_xgb(train[BASE_FEATURES], test_cc[BASE_FEATURES], cat)

    print(f"  train rows: complete-case {len(train_cc):,} vs all {len(train):,}")
    p = fit_xgb_log(Xtr_cc, train_cc[TARGET], Xte_cc)
    show("trained on complete cases  -> complete-case test", metrics(test_cc[TARGET], p))
    p = fit_xgb_log(Xtr_all, train[TARGET], Xte_cc_from_all)
    show("trained on ALL rows (NaN)  -> complete-case test", metrics(test_cc[TARGET], p))
    p = fit_xgb_log(Xtr_all, train[TARGET], Xte_all)
    show("trained on ALL rows (NaN)  -> all-rows test", metrics(test[TARGET], p))
    # the rows the current pipeline can't even score:
    test_na = test[test[BASE_FEATURES].isna().any(axis=1)]
    if len(test_na):
        _, Xte_na = encode_for_xgb(train[BASE_FEATURES], test_na[BASE_FEATURES], cat)
        p = fit_xgb_log(Xtr_all, train[TARGET], Xte_na)
        show(f"trained on ALL rows (NaN)  -> NA-only test ({len(test_na):,} rows)",
             metrics(test_na[TARGET], p))


def e3_scoring_space(df_all: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("E3 — HPO scored in raw price space (current core) vs log space")
    print("=" * 78)
    from scipy.stats import loguniform, randint
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV

    cc = df_all.dropna().reset_index(drop=True)
    cat = [c for c in BASE_FEATURES if cc[c].dtype == object]
    X_tr, X_te, y_tr, y_te = train_test_split(
        cc[BASE_FEATURES], cc[TARGET], test_size=0.2, random_state=0)
    Xtr, Xte = encode_for_xgb(X_tr, X_te, cat)

    space = {
        "n_estimators": randint(100, 801),
        "max_depth": randint(3, 9),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }

    for label, y_fit, inv in (
        ("scored in RAW price space (core behaviour)", y_tr.to_numpy(), None),
        ("scored in LOG space", np.log(y_tr.to_numpy()), np.exp),
    ):
        search = HalvingRandomSearchCV(
            XGBRegressor(**{**XGB_FIXED, "n_estimators": 300}),
            param_distributions=space, n_candidates=24, cv=5,
            scoring="r2", random_state=0, n_jobs=-1, refit=True)
        t0 = time.time()
        search.fit(Xtr, y_fit)
        preds = search.best_estimator_.predict(Xte)
        if inv is not None:
            preds = inv(preds)
        m = metrics(y_te, preds)
        show(f"{label}  [cv best {search.best_score_:.3f}, {time.time()-t0:.0f}s]", m)


def build_extended_dataset(case_prep) -> pd.DataFrame:
    char = pd.read_csv(DATA / "characteristics_5306.csv", dtype=str, keep_default_na=False)
    ident = pd.read_csv(DATA / "identification_5306.csv", dtype=str, keep_default_na=False)
    mgmt = pd.read_csv(DATA / "management_5306.csv", dtype=str, keep_default_na=False)

    keep_stmts = set(case_prep.SPEC_COLUMNS) | set(EXTRA_SPECS)
    keep = char[char["REQUIREMENTS_STATEMENT"].isin(keep_stmts)]
    wide = keep.pivot_table(index="NIIN", columns="REQUIREMENTS_STATEMENT",
                            values="CLEAR_TEXT_REPLY", aggfunc="first")
    renames = dict(case_prep.SPEC_COLUMNS)
    renames.update({k: v[0] for k, v in EXTRA_SPECS.items()})
    wide = wide.rename(columns=renames).reset_index()
    wide.columns.name = None

    price = case_prep._one_price_per_niin(mgmt)
    df = wide.merge(price, on="NIIN", how="inner")
    df = df.merge(ident[["NIIN", "ITEM_NAME"]].drop_duplicates("NIIN"), on="NIIN", how="left")

    for col in ("head_style", "thread_series", "thread_direction"):
        if col in df.columns:
            df[col] = df[col].apply(case_prep._dedup_doubled)
    df["material"] = df["material"].apply(case_prep._material_grade)
    df["finish"] = df["finish"].apply(case_prep._finish_group)
    df["thread_class"] = df["thread_class"].apply(case_prep._thread_class)

    num_cols = case_prep.NUMERIC_SPECS + [v[0] for v in EXTRA_SPECS.values() if v[1] == "num"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].apply(case_prep._leading_float)

    # spec_standard: keep the leading document token ("NAS6204", "MS90726", ...)
    if "spec_standard" in df.columns:
        df["spec_standard"] = df["spec_standard"].map(
            lambda s: re.split(r"[\s,]", str(s).strip())[0][:12] if s and str(s).strip() else np.nan)
    if "material_doc" in df.columns:
        df["material_doc"] = df["material_doc"].map(
            lambda s: str(s).strip()[:40] if s and str(s).strip() else np.nan)
    if "hardness" in df.columns:
        df["hardness"] = df["hardness"].map(
            lambda s: str(s).strip()[:30] if s and str(s).strip() else np.nan)

    df = df[df[TARGET].between(case_prep.MIN_PRICE_EACH, case_prep.MAX_PRICE_EACH)]
    df = df.dropna(subset=case_prep.CORE_SPECS)
    df = df[df["material"] != "unknown"]
    for col in df.columns:
        if df[col].dtype == object and col not in ("NIIN",):
            df[col] = df[col].replace("", np.nan)
    return df.drop(columns=["NIIN"]).reset_index(drop=True)


def e4_extended_features(df_ext: pd.DataFrame, noise_report) -> None:
    print("\n" + "=" * 78)
    print("E4 — extended features (8 extra PUB LOG specs + ITEM_NAME)")
    print("=" * 78)
    ext_features = [c for c in df_ext.columns if c != TARGET]
    cat = [c for c in ext_features if df_ext[c].dtype == object]
    print(f"  rows {len(df_ext):,}   features {len(ext_features)} "
          f"(vs {len(BASE_FEATURES)} in the study)")

    # noise ceiling on the WIDER signature
    sig_all = ext_features
    rep = noise_report(df_ext.assign(**{TARGET: df_ext[TARGET]}), sig_all, TARGET)
    print(f"  extended-signature diagnosis: singletons {rep['singleton_frac']*100:.0f}%  "
          f"ceiling {rep['r2_ceiling']:.2f}  band {rep['median_price_band']:.1f}x  "
          f"(dup rows {rep['dup_rows']:,})")

    tr_idx, te_idx = train_test_split(df_ext.index, test_size=0.2, random_state=0)
    train, test = df_ext.loc[tr_idx], df_ext.loc[te_idx]
    Xtr, Xte = encode_for_xgb(train[ext_features], test[ext_features], cat)
    p = fit_xgb_log(Xtr, train[TARGET], Xte)
    show("XGB all-rows native-NaN, EXTENDED features -> all-rows test",
         metrics(test[TARGET], p))

    Xtr_b, Xte_b = encode_for_xgb(
        train[BASE_FEATURES], test[BASE_FEATURES],
        [c for c in BASE_FEATURES if df_ext[c].dtype == object])
    p = fit_xgb_log(Xtr_b, train[TARGET], Xte_b)
    show("XGB all-rows native-NaN, BASE features (same split)",
         metrics(test[TARGET], p))

    # which extras carry the lift
    model = XGBRegressor(**XGB_FIXED)
    model.fit(Xtr, np.log(train[TARGET]))
    imp = sorted(zip(ext_features, model.feature_importances_),
                 key=lambda kv: -kv[1])[:12]
    print("  top gain importances:", ", ".join(f"{k} {v:.2f}" for k, v in imp))


def e5_signature_medians(df_all: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("E5 — per-signature median aggregation (README Part 2b)")
    print("=" * 78)
    cat = [c for c in BASE_FEATURES if df_all[c].dtype == object]
    agg = (df_all.groupby(BASE_FEATURES, dropna=False)[TARGET]
           .median().reset_index())
    print(f"  {len(agg):,} signatures (from {len(df_all):,} rows)")
    tr_idx, te_idx = train_test_split(agg.index, test_size=0.2, random_state=0)
    train, test = agg.loc[tr_idx], agg.loc[te_idx]
    Xtr, Xte = encode_for_xgb(train[BASE_FEATURES], test[BASE_FEATURES], cat)
    p = fit_xgb_log(Xtr, train[TARGET], Xte)
    show("XGB on per-signature median prices", metrics(test[TARGET], p))


def e6_group_mean_ceiling(df_all: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("E6 — leave-one-out group-mean on duplicate-signature rows")
    print("=" * 78)
    key = df_all[BASE_FEATURES].astype("string").fillna("<NA>").agg("|".join, axis=1)
    size = df_all.groupby(key)[TARGET].transform("size")
    d = df_all[size >= 2]
    k = key[size >= 2]
    logt = np.log(d[TARGET])
    gsum = logt.groupby(k).transform("sum")
    gn = logt.groupby(k).transform("size")
    loo_mean = (gsum - logt) / (gn - 1)
    r2 = r2_score(logt, loo_mean)
    print(f"  duplicate rows {len(d):,}: a pure lookup (mean of the OTHER rows with the")
    print(f"  identical spec) achieves log R2 = {r2:.3f} — the practical ceiling for")
    print(f"  perfectly-known specs, vs the in-sample variance ceiling of ~0.60.")


def e7_combined_best(df_ext: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("E7 — combined best effort: all rows + native NaN + extended features"
          " + log-space HPO")
    print("=" * 78)
    from scipy.stats import loguniform, randint
    from sklearn.model_selection import RandomizedSearchCV

    ext_features = [c for c in df_ext.columns if c != TARGET]
    cat = [c for c in ext_features if df_ext[c].dtype == object]
    tr_idx, te_idx = train_test_split(df_ext.index, test_size=0.2, random_state=0)
    train, test = df_ext.loc[tr_idx], df_ext.loc[te_idx]
    Xtr, Xte = encode_for_xgb(train[ext_features], test[ext_features], cat)

    space = {
        "n_estimators": randint(300, 1200),
        "max_depth": randint(4, 11),
        "learning_rate": loguniform(0.01, 0.2),
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": randint(1, 20),
        "reg_lambda": loguniform(0.1, 30),
    }
    search = RandomizedSearchCV(
        XGBRegressor(**XGB_FIXED), space, n_iter=30, cv=4,
        scoring="r2", random_state=0, n_jobs=-1, refit=True)
    t0 = time.time()
    search.fit(Xtr, np.log(train[TARGET]))
    preds = np.exp(search.best_estimator_.predict(Xte))
    m = metrics(test[TARGET], preds)
    show(f"tuned (cv logR2 {search.best_score_:.3f}, {time.time()-t0:.0f}s)", m)
    print(f"  best params: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()} }")


def e8_banded_intervals(df_all: pd.DataFrame) -> None:
    """Global conformal q_hat (current core) vs price-band (Mondrian) q_hat."""
    print("\n" + "=" * 78)
    print("E8 — likely-range width: one global q_hat vs banded calibration")
    print("=" * 78)
    cat = [c for c in BASE_FEATURES if df_all[c].dtype == object]
    tr_idx, te_idx = train_test_split(df_all.index, test_size=0.2, random_state=0)
    train, test = df_all.loc[tr_idx], df_all.loc[te_idx]
    Xtr, Xte = encode_for_xgb(train[BASE_FEATURES], test[BASE_FEATURES], cat)
    model = XGBRegressor(**XGB_FIXED)
    model.fit(Xtr, np.log(train[TARGET]))

    # split the holdout into calibration / evaluation halves
    Xte = Xte.reset_index(drop=True)
    y_te = test[TARGET].reset_index(drop=True)
    cal_idx, ev_idx = train_test_split(Xte.index, test_size=0.5, random_state=1)
    pred_cal = np.exp(model.predict(Xte.loc[cal_idx]))
    pred_ev = np.exp(model.predict(Xte.loc[ev_idx]))
    res_cal = np.abs(np.log(y_te.loc[cal_idx].to_numpy()) - np.log(pred_cal))
    res_ev = np.abs(np.log(y_te.loc[ev_idx].to_numpy()) - np.log(pred_ev))

    def q90(r):
        n = len(r)
        lvl = min(1.0, np.ceil((n + 1) * 0.9) / n)
        return float(np.quantile(r, lvl, method="higher"))

    q_global = q90(res_cal)
    print(f"  global q_hat: every part gets a x{np.exp(2*q_global):.0f} wide 90% band "
          f"(coverage on eval: {np.mean(res_ev <= q_global)*100:.1f}%)")

    bands = [0, 5, 155, np.inf]
    labels = ["< $5", "$5-$155", "> $155"]
    band_cal = pd.cut(pred_cal, bands, labels=labels)
    band_ev = pd.cut(pred_ev, bands, labels=labels)
    for lab in labels:
        r_c = res_cal[band_cal == lab]
        r_e = res_ev[band_ev == lab]
        if len(r_c) < 20 or len(r_e) == 0:
            continue
        q_b = q90(r_c)
        print(f"  band {lab:<9} (n_cal={len(r_c):>4}): banded q_hat -> "
              f"x{np.exp(2*q_b):>5.0f} wide band  "
              f"(vs global x{np.exp(2*q_global):.0f}; coverage {np.mean(r_e <= q_b)*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    case_prep = _load_case_prepare()
    sys.path.insert(0, str(HERE))
    from diagnose_noise import noise_report

    df_all = pd.read_csv(DATA / "bolts_clean.csv")
    print(f"bolts_clean.csv: {len(df_all):,} rows")

    e1_verify_saved_model(df_all)
    e2_nan_handling(df_all)
    if not args.quick:
        e3_scoring_space(df_all)
        df_ext = build_extended_dataset(case_prep)
        e4_extended_features(df_ext, noise_report)
        e7_combined_best(df_ext)
    e5_signature_medians(df_all)
    e6_group_mean_ceiling(df_all)
    e8_banded_intervals(df_all)


if __name__ == "__main__":
    main()
