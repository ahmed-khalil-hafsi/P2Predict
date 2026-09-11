"""Operating characteristics of the bias gate: the point-null test vs an equivalence test.

Addendum evidence for `research/bias_gate_materiality.md`. Read-only: it
simulates holdout residuals and refits throwaway models in memory. It never
writes a model and never touches core.

The materiality floor can't be calibrated on our three case studies -- users
price anything from castings to cloud contracts. So instead of asking "what is
bias like in the wild" (unknowable), this measures the *operating
characteristics of the test itself* on a synthetic grid where the true bias is
known by construction. Those curves are properties of the gate, not of anyone's
dataset, so they generalise. The case studies then serve as spot-checks.

Three parts:
  A. false-flag / miss / n-sensitivity curves for both gates, over
     n x true_bias x noise x price-spread, for several candidate floors
  B. the resolution floor -- the smallest bias each holdout size can actually
     resolve, which is what sets the honest 'unknown' region
  C. spot-checks on the three case-study datasets

Run from the repo root:  python research/bias_gate_equivalence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom, t as student_t

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SEED = 11
DRAWS = 400          # independent holdouts simulated per grid cell
CI_LEVEL = 0.95      # for the equivalence test's median interval
ALPHA_T = 0.05       # UNBIASED_P in quality.py

# --- the grid -------------------------------------------------------------
# n: spans 'insufficient_data' (15) to the case-study holdouts (16k).
N_GRID = [15, 20, 30, 50, 100, 250, 500, 1000, 2500, 8000, 16000]
# true median bias: the model reads this fraction high on the typical part.
BIAS_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.20]
# log-scale noise sd -> roughly the typical % error a buyer would see.
NOISE_GRID = [0.15, 0.30, 0.50]
# spread of the price catalogue itself (sd of log price). Irrelevant to a
# relative test, but it is what makes the dollar-space t-test misbehave.
SPREAD_GRID = [0.5, 1.0, 1.5]
# candidate materiality bands to compare.
FLOOR_GRID = [0.02, 0.03, 0.05, 0.07, 0.10]
# the floor the per-n breakdown is printed at (see the addendum's derivation).
REPORT_FLOOR = 0.05

# What counts as immaterial / clearly material when scoring the gates. Stated
# up front so the summary numbers aren't tuned after the fact.
IMMATERIAL_MAX = 0.01
MATERIAL_MIN = 0.10


# --- the two gates --------------------------------------------------------

def flags_today(y, yhat):
    """Current gate: one-sample t-test of DOLLAR residuals against zero.

    Mirrors quality.py:333 + quality.py:200, vectorised over draws (rows).
    Returns the boolean 'flagged unreliable' per draw.
    """
    resid = y - yhat
    n = resid.shape[1]
    mean = resid.mean(axis=1)
    sd = resid.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tstat = mean / (sd / np.sqrt(n))
    p = 2.0 * student_t.sf(np.abs(tstat), df=n - 1)
    return p <= ALPHA_T, p


def median_ci(rel, level=CI_LEVEL):
    """Distribution-free CI for the median, from order statistics.

    No bootstrap: the interval is the (k_lo, k_hi) order statistics whose
    binomial tail mass is <= (1-level)/2 either side. Exact, assumption-free,
    and -- the reason it belongs in core rather than a bootstrap -- fully
    deterministic, so a verdict never depends on an RNG seed.

    `rel` is (draws, n). Returns (lo, hi) arrays, NaN where n is too small for
    any interval to exist at this level.
    """
    n = rel.shape[1]
    alpha = 1.0 - level
    k_lo = int(binom.ppf(alpha / 2.0, n, 0.5)) - 1       # 0-indexed
    k_hi = int(binom.isf(alpha / 2.0, n, 0.5))
    if k_lo < 0 or k_hi > n - 1:
        nan = np.full(rel.shape[0], np.nan)
        return nan, nan
    srt = np.sort(rel, axis=1)
    return srt[:, k_lo], srt[:, k_hi]


def verdict_proposed(rel, floor):
    """Four states, matching quality.bias_assessment (vectorised over draws).

    Collapsed to the three ACTIONABLE outcomes the gate is scored on:
      flag    = 'material'                            -> verdict 'unreliable'
      passed  = 'immaterial' or 'likely_immaterial'   -> trustworthy / usable
      unknown = 'likely_material' or 'unmeasured'     -> verdict 'unknown'

    'likely_material' is deliberately NOT scored as a flag: at a 20-60 part
    holdout the point estimate alone is too noisy to condemn a model on, so it
    carries a directional warning under 'unknown' rather than a verdict.
    """
    lo, hi = median_ci(rel)
    med = np.median(rel, axis=1)
    resolvable = ~np.isnan(lo)
    material = resolvable & ((lo > floor) | (hi < -floor))
    immaterial = resolvable & (lo > -floor) & (hi < floor)
    straddle = resolvable & ~material & ~immaterial
    likely_immaterial = straddle & (np.abs(med) < floor)
    likely_material = straddle & ~likely_immaterial

    flag = material
    passed = immaterial | likely_immaterial
    unknown = likely_material | ~resolvable
    return flag, unknown, passed


# --- part A: the sweep ----------------------------------------------------

def simulate(rng, n, bias, noise, spread, draws=DRAWS):
    """Holdout residuals with a known true median bias.

    log y = mu + e, mu ~ N(0, spread^2), e ~ N(0, noise^2). A model that
    estimates E[log y | x] perfectly predicts exp(mu); multiplying by
    (1 + bias) makes it read `bias` high on the typical part. Note the mean
    residual is off by exp(noise^2/2) even at bias=0 -- the retransformation
    effect falls out of the construction rather than being injected.
    """
    mu = rng.normal(0.0, spread, size=(draws, n))
    e = rng.normal(0.0, noise, size=(draws, n))
    y = np.exp(mu + e)
    yhat = np.exp(mu) * (1.0 + bias)
    return y, yhat


def sweep():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in N_GRID:
        for bias in BIAS_GRID:
            for noise in NOISE_GRID:
                for spread in SPREAD_GRID:
                    y, yhat = simulate(rng, n, bias, noise, spread)
                    rel = y / yhat - 1.0
                    today, _ = flags_today(y, yhat)
                    row = {
                        "n": n, "true_bias": bias, "noise": noise,
                        "spread": spread,
                        "median_rel_resid": float(np.median(rel)),
                        "flag_rate_today": float(today.mean()),
                        "pass_rate_today": float(1.0 - today.mean()),
                    }
                    for floor in FLOOR_GRID:
                        f, u, p = verdict_proposed(rel, floor)
                        key = f"{int(floor * 100):02d}"
                        row[f"flag_rate_m{key}"] = float(f.mean())
                        row[f"unknown_rate_m{key}"] = float(u.mean())
                        row[f"pass_rate_m{key}"] = float(p.mean())
                    rows.append(row)
    return rows


def score(rows):
    """Collapse the grid into the three numbers that decide the design."""
    imm = [r for r in rows if r["true_bias"] <= IMMATERIAL_MAX]
    mat = [r for r in rows if r["true_bias"] >= MATERIAL_MIN]
    out = {
        "today": {
            "false_flag_rate": float(np.mean([r["flag_rate_today"] for r in imm])),
            "miss_rate": float(np.mean([1 - r["flag_rate_today"] for r in mat])),
            "verdict_inversion": verdict_inversion(
                rows, "pass_rate_today", "flag_rate_today"),
        }
    }
    for floor in FLOOR_GRID:
        key = f"{int(floor * 100):02d}"
        out[f"m={floor:.2f}"] = {
            # a 'false flag' is only an outright flag; 'unknown' is honest,
            # not wrong, so it is reported separately.
            "false_flag_rate": float(np.mean([r[f"flag_rate_m{key}"] for r in imm])),
            "unknown_rate_immaterial": float(
                np.mean([r[f"unknown_rate_m{key}"] for r in imm])),
            "miss_rate": float(np.mean([r[f"pass_rate_m{key}"] for r in mat])),
            "unknown_rate_material": float(
                np.mean([r[f"unknown_rate_m{key}"] for r in mat])),
            "verdict_inversion": verdict_inversion(
                rows, f"pass_rate_m{key}", f"flag_rate_m{key}"),
        }
    return out


def verdict_inversion(rows, pass_field, flag_field):
    """Does the gate give OPPOSITE actionable answers for the same model?

    The original complaint, in one number. Hold the model fixed (bias, noise,
    spread) and vary only n: if the gate both *passes* it at one holdout size
    and calls it *unreliable* at another, it is measuring n, not bias. Scored
    as min(max_n pass_rate, max_n flag_rate) -- high only when both happen.

    An 'unknown' at small n is not an inversion: it is the gate declining to
    answer, which is the honest response and is scored separately.
    """
    worst = 0.0
    combos = {(r["true_bias"], r["noise"], r["spread"]) for r in rows}
    for bias, noise, spread in combos:
        sel = [r for r in rows
               if (r["true_bias"], r["noise"], r["spread"]) == (bias, noise, spread)]
        max_pass = max(r[pass_field] for r in sel)
        max_flag = max(r[flag_field] for r in sel)
        worst = max(worst, min(max_pass, max_flag))
    return float(worst)


def by_n(rows, floor, bias):
    """Per-holdout-size verdict mix at one floor and one true bias.

    Averaged over noise and price spread. This is the usability read: where
    does the honest answer stop being 'unknown'?
    """
    key = f"{int(floor * 100):02d}"
    out = []
    for n in N_GRID:
        sel = [r for r in rows if r["n"] == n and r["true_bias"] == bias]
        out.append({
            "n": n,
            "today_flag": float(np.mean([r["flag_rate_today"] for r in sel])),
            "pass": float(np.mean([r[f"pass_rate_m{key}"] for r in sel])),
            "unknown": float(np.mean([r[f"unknown_rate_m{key}"] for r in sel])),
            "flag": float(np.mean([r[f"flag_rate_m{key}"] for r in sel])),
        })
    return out


# --- part B: what can a holdout of size n actually resolve? ---------------

def resolution(rng):
    """Smallest true bias each n can call material, and the CI half-width.

    The half-width is the honest limit: no floor below it can ever be resolved
    at that holdout size, so a tighter floor buys 'unknown', not safety.
    """
    out = []
    for n in N_GRID:
        for noise in NOISE_GRID:
            y, yhat = simulate(rng, n, 0.0, noise, 1.0, draws=DRAWS)
            rel = y / yhat - 1.0
            lo, hi = median_ci(rel)
            half = float(np.nanmedian((hi - lo) / 2.0)) if not np.all(np.isnan(lo)) else float("nan")
            smallest = None
            for bias in BIAS_GRID:
                if bias == 0.0:
                    continue
                yb, yhb = simulate(rng, n, bias, noise, 1.0, draws=DRAWS)
                relb = yb / yhb - 1.0
                # detected when the gate flags it at a floor of half that bias
                f, _, _ = verdict_proposed(relb, bias / 2.0)
                if f.mean() >= 0.80:
                    smallest = bias
                    break
            out.append({
                "n": n, "noise": noise,
                "ci_half_width": half,
                "smallest_resolvable_bias": smallest,
            })
    return out



# --- part C: spot-checks on the three case-study datasets ----------------
# Not calibration -- three datasets cannot calibrate a threshold for every
# category a user might price. These only check that the rule chosen on the
# synthetic operating characteristics gives a sane answer on real models.

DATASETS = [
    ("used cars",
     "case-studies/used-cars/data/vehicles_training.csv",
     "case-studies/used-cars/data-sample/vehicles_sample.csv",
     "price",
     ["year", "odometer", "manufacturer", "condition", "fuel",
      "transmission", "drive", "type", "state", "paint_color"]),
    ("heavy equipment",
     "case-studies/heavy-equipment-sales/data/bulldozers_training.csv",
     "case-studies/heavy-equipment-sales/data-sample/bulldozers_sample.csv",
     "sale_price_usd",
     ["age_at_sale", "sale_year", "product_group", "product_size",
      "enclosure", "state"]),
    ("battery mgmt ICs",
     "case-studies/battery-management-ics/data/bmics_clean.csv",
     "case-studies/battery-management-ics/data-sample/bmics_sample.csv",
     "unit_price_at_1_usd",
     ["manufacturer", "Battery Chemistry", "Interface", "max_cells_supported",
      "op_temp_min_C", "op_temp_max_C", "package_pins", "is_multi_cell"]),
]


def _fit(df, target, features):
    """Same throwaway fit as bias_gate_materiality.py, so numbers compare."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from p2predict.training import start_training

    df = df.dropna(subset=[target])
    df = df[df[target] > 0]
    X, y = df[features], df[target]
    numeric = [f for f in features if pd.api.types.is_numeric_dtype(X[f])]
    categorical = [f for f in features if f not in numeric]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    model, _, log_target = start_training(
        X_tr, y_tr, numeric, categorical, "xgboost", log_target=None)
    return model, X_te, y_te, log_target


def spot_checks(floor=REPORT_FLOOR):
    import pandas as pd
    from sklearn.metrics import r2_score
    from p2predict.quality import assess_model, residual_bias_p

    out = []
    for label, csv, sample, target, features in DATASETS:
        path = REPO / csv
        if not path.exists():
            path = REPO / sample
        if not path.exists():
            print(f"skip {label}: no data", file=sys.stderr)
            continue
        df = pd.read_csv(path, low_memory=False)
        model, X_te, y_te, log_target = _fit(df, target, features)
        y_true = np.asarray(y_te, float)
        y_pred = np.asarray(model.predict(X_te), float)
        n = len(y_true)
        r2 = float(r2_score(y_true, y_pred))
        rel = (y_true / y_pred - 1.0)[None, :]

        today = assess_model(r2, residual_bias_p(y_true, y_pred), n)["verdict"]
        lo, hi = median_ci(rel)
        f, u, pa = verdict_proposed(rel, floor)
        proposed = "unreliable" if f[0] else ("unknown" if u[0] else "pass")

        # the inversion test on a real model: same model, only n varies.
        rng = np.random.default_rng(SEED)
        ladder = []
        for k in sorted({k for k in [30, 50, 100, 250, 500, 1000, 2500, n]
                         if k <= n}):
            draws = 1 if k == n else 200
            t_flag = p_flag = p_unk = 0
            for _ in range(draws):
                idx = (rng.choice(n, size=k, replace=False) if k < n
                       else np.arange(n))
                t_flag += assess_model(
                    r2, residual_bias_p(y_true[idx], y_pred[idx]), k
                )["verdict"] == "unreliable"
                rk = (y_true[idx] / y_pred[idx] - 1.0)[None, :]
                fk, uk, _ = verdict_proposed(rk, floor)
                p_flag += bool(fk[0]); p_unk += bool(uk[0])
            ladder.append({
                "n": int(k),
                "today_pct_unreliable": 100.0 * t_flag / draws,
                "proposed_pct_unreliable": 100.0 * p_flag / draws,
                "proposed_pct_unknown": 100.0 * p_unk / draws,
            })

        rec = {
            "dataset": label, "n_holdout": n, "r2": r2,
            "log_target": bool(log_target),
            "mean_residual_pct": float(np.mean(rel) * 100),
            "median_residual_pct": float(np.median(rel) * 100),
            "median_ci_pct": [float(lo[0] * 100), float(hi[0] * 100)],
            "verdict_today": today, "verdict_proposed": proposed,
            "ladder": ladder,
        }
        out.append(rec)

        print(f"\n=== {label}  (n={n:,}, R2 {r2:.3f}, log_target={log_target})")
        print(f"    mean residual {rec['mean_residual_pct']:+.1f}%   "
              f"median residual {rec['median_residual_pct']:+.1f}%   "
              f"median CI [{lo[0]*100:+.1f}%, {hi[0]*100:+.1f}%]")
        print(f"    today: '{today}'   proposed (m={floor:.0%}): '{proposed}'")
        print(f"    {'n':>8} {'today unrel':>12} {'prop unrel':>11} {'prop unk':>9}")
        for r in ladder:
            print(f"    {r['n']:>8,} {r['today_pct_unreliable']:>11.0f}% "
                  f"{r['proposed_pct_unreliable']:>10.0f}% "
                  f"{r['proposed_pct_unknown']:>8.0f}%")
    return out




def _dump(obj):
    """Pretty JSON, but with each numeric array collapsed onto one line."""
    import re
    txt = json.dumps(obj, indent=2)
    return re.sub(
        r"\[\s+-?[\d.][^\[\]{}\"a-zA-Z]*?\]",
        lambda m: re.sub(r"\s+", " ", m.group(0)).replace("[ ", "[").replace(" ]", "]"),
        txt,
    )


def _columnar(rows, ndigits=4):
    """Grid rows as {columns, rows} instead of 891 key-repeating objects.

    Same information, ~5x smaller on disk -- this folder keeps results files
    small enough to read. Rates come from DRAWS draws, so 4 decimals is
    already past their resolution.
    """
    cols = list(rows[0])
    return {
        "columns": cols,
        "rows": [[round(r[c], ndigits) if isinstance(r[c], float) else r[c]
                  for c in cols] for r in rows],
    }


def main():
    rng = np.random.default_rng(SEED + 1)
    rows = sweep()
    result = {
        "config": {
            "seed": SEED, "draws_per_cell": DRAWS, "ci_level": CI_LEVEL,
            "alpha_t": ALPHA_T, "immaterial_max": IMMATERIAL_MAX,
            "material_min": MATERIAL_MIN,
        },
        "summary": score(rows),
        "resolution": resolution(rng),
        "spot_checks": None,  # filled below so part A prints first
        "grid": _columnar(rows),
    }
    out = REPO / "research" / "bias_gate_equivalence_results.json"
    out.write_text(_dump(result))

    print(f"grid cells: {len(rows)}  draws/cell: {DRAWS}\n")
    print("=== A. operating characteristics ===")
    print(f"{'gate':10s} {'false-flag':>11s} {'miss':>7s} {'inversion':>10s} "
          f"{'unk(imm)':>9s} {'unk(mat)':>9s}")
    s = result["summary"]
    print(f"{'today':10s} {s['today']['false_flag_rate']:>10.1%} "
          f"{s['today']['miss_rate']:>7.1%} "
          f"{s['today']['verdict_inversion']:>10.2f} {'-':>9s} {'-':>9s}")
    for floor in FLOOR_GRID:
        k = f"m={floor:.2f}"
        print(f"{k:10s} {s[k]['false_flag_rate']:>10.1%} {s[k]['miss_rate']:>7.1%} "
              f"{s[k]['verdict_inversion']:>10.2f} "
              f"{s[k]['unknown_rate_immaterial']:>9.1%} "
              f"{s[k]['unknown_rate_material']:>9.1%}")

    for bias in (0.0, 0.10):
        print(f"\n=== A2. verdict mix by holdout size  (m={REPORT_FLOOR:.0%}, "
              f"true bias {bias:.0%}) ===")
        print(f"{'n':>6s} {'today':>18s} {'|':>3s} {'pass':>8s} {'unknown':>9s} "
              f"{'unreliable':>11s}")
        for r in by_n(rows, REPORT_FLOOR, bias):
            today = "unreliable" if r["today_flag"] > 0.5 else "trustworthy"
            today_col = f"{today} ({r['today_flag']:.0%})"
            print(f"{r['n']:>6d} {today_col:>18s} {'|':>3s} "
                  f"{r['pass']:>8.0%} {r['unknown']:>9.0%} {r['flag']:>11.0%}")

    print("\n=== B. what each holdout size can resolve (median CI half-width) ===")
    print(f"{'n':>6s} " + " ".join(f"{'noise=' + str(nz):>13s}" for nz in NOISE_GRID))
    for n in N_GRID:
        cells = {r["noise"]: r for r in result["resolution"] if r["n"] == n}
        parts = []
        for nz in NOISE_GRID:
            r = cells[nz]
            hw = r["ci_half_width"]
            parts.append("n/a" .rjust(13) if np.isnan(hw) else f"+/-{hw:>9.1%}")
        print(f"{n:>6d} " + " ".join(parts))
    print("\n=== C. spot-checks on the case-study datasets ===")
    result["spot_checks"] = spot_checks()
    out.write_text(_dump(result))
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
