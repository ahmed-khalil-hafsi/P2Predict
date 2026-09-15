# Finding: the agent never learns a feature has outliers, so it never fixes one

**Status: open.** **No core code changed.** The capability that fixes this
already exists and already works; the gap is that its trigger signal is
computed and then discarded before the agent can see it. Nothing here proposes
a new algorithm, a new dependency or a new policy.

Evidence: `feature_outlier_visibility.py`, results in
`feature_outlier_visibility_results.txt`. Surfaced by the
[TDC ADMET validation](../evals/tdc_admet_validation.md) (section 3.2), where
RDKit's `Ipc` descriptor reaches ~10^31 and Ridge's cross-validated R^2
correspondingly reached ~-10^32, removing it from selection on every run.

## TL;DR

One pathological cell in one numeric column silently costs you the better
model. Price here is *generated* as a linear function of two specs, so Ridge is
the correct answer and anything else is the tool being misled:

| Feature-outlier policy | Selected | Ridge CV R^2 | Holdout MAE |
|---|---|---|---|
| none (clean data, for reference) | **ridge** | 0.997 | **1.49** |
| `warn` — the current default | xgboost | **-4.6e53** | **3.64** |
| `drop` | **ridge** | 0.997 | **1.49** |
| `winsorize` | **ridge** | 0.997 | **1.48** |

The delivered model is **2.4x worse** on holdout, and nothing tells the user
why. `drop` and `winsorize` both restore it completely. **The feature works.
It is simply never invoked, because the agent is never told there is anything
to invoke it for.**

## Why it happens

Two independent mechanisms compose, and neither is a coding error.

1. **Linear extrapolation.** The linear path imputes and applies
   `StandardScaler` (`preprocessing.py`), which is correct. But the scaler fits
   on the training fold. A validation-fold row carrying a value orders of
   magnitude larger gets an astronomical z-score, and a linear model
   extrapolates linearly from it. This is textbook behaviour for linear models
   on unbounded features, not a defect.
2. **Selection by CV R^2.** `auto_train` crowns the highest cross-validated
   score. A single fold scoring -1e53 is unrecoverable, so Ridge loses
   regardless of how well it fits the other 99% of the data. The failure is
   silent: the user receives a working tree model and no indication that a
   better one was disqualified by one cell.

## The actual gap: the signal is computed, then thrown away

`apply_feature_outlier_policy` already returns a full per-column summary
(counts, bounds). Its own docstring says `warn` "surfaces a message at the
caller". The CLI does exactly that:

```
src/p2predict/cli/train.py:292   data, feature_outlier_summary = apply_feature_outlier_policy(...)
src/p2predict/cli/train.py:295   if feature_outlier_summary["n_outliers_total"] > 0 and not json_mode:
```

The MCP path does not:

```
src/p2predict/mcp/server.py:224  data, _ = apply_feature_outlier_policy(...)   # propose_training_plan
src/p2predict/mcp/server.py:994  data, _ = apply_feature_outlier_policy(...)   # train
```

Both MCP tools accept `feature_outlier_policy` and both default it to `warn`,
which changes nothing. So on the agent path the detection runs, produces
`n_outliers_total=1, Wide_Range_Spec n_outliers=1`, and that result is bound to
`_`.

This matters more than a CLI-vs-MCP asymmetry normally would, because the MCP
server is the primary interface: end users reach P2Predict through an agent,
and the CLI is a developer surface. A warning only the CLI prints is a warning
almost no user will ever see.

It also answers the question that prompted this finding, which was why agents
driving P2Predict never set `feature_outlier_policy` and lean entirely on
`auto_train`. They are not ignoring the control. They have no evidence it is
needed, and `propose_training_plan` — the call they are instructed to make
first, precisely so the human can decide the data questions — is silent on it.

## What to change

Preference order, cheapest first. All are plumbing, none touches the estimator.

1. **Return the summary from `propose_training_plan`.** It already builds a
   `warnings` list for exactly this kind of thing. Adding the per-column
   outlier counts puts the decision in front of the agent at the one moment it
   is required to consult the user, with the existing policy names as the
   options. This alone likely closes the gap.
2. **Return it from `train` too**, so a caller that skipped the plan still gets
   told what the run saw.
3. **Only then consider defaults.** Changing the default from `warn` to
   `winsorize` would fix the measured case silently, but it also mutates a
   user's data without asking, which cuts against the tool's stance of
   explaining rather than deciding. Not proposed here. Surfacing first is
   reversible; a changed default is a behaviour change on every existing
   workflow.

## What would kill this

* If the per-column summary turns out to be noisy on real procurement data
  (Tukey IQR flags a large fraction of rows on ordinary skewed spend), then
  surfacing it becomes an alarm nobody reads and the finding is a no-op. This
  is the first thing to measure, on the case-study datasets, before any change.
* If agents given the summary still do nothing with it, the problem is the
  guidance rather than the plumbing, and the fix belongs in the MCP tool
  descriptions instead.

## Scope note

This is not the out-of-domain case. A part whose specs sit far outside training
range at *predict* time is already capped by the `in_domain` block shipped in
#40. This finding is about training-time model selection, where no such guard
exists, and the failure direction is a quietly weaker model rather than a
quietly wrong price.
