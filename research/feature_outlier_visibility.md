# Finding: the agent never learns a feature has outliers, so it never fixes one

**Status: fix implemented, open for review.** The measurements below changed
the fix twice, so read the *What to change* section as the record of what the
evidence actually supports, not as the original pitch. The capability that fixes this
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

The MCP path did not. **Correction to an earlier draft of this document,
which cited a line number in `propose_training_plan`: that tool never called
`apply_feature_outlier_policy` at all.** It checks leakage, ID-like columns and
the log-target rule, and was silent on feature magnitude by omission. The only
MCP call site was in `train`:

```
src/p2predict/mcp/server.py:994  data, _ = apply_feature_outlier_policy(...)   # train
```

So on the agent path the detection either never ran (`propose_training_plan`)
or ran and had its result bound to `_` (`train`). Either way nothing reached
the agent.

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

The first two versions of this proposal did not survive measurement. Recorded
in order, because the dead ends are the useful part.

**Rejected: surface the raw IQR summary.** Measured on the checked-in
case-study samples, with the target correctly excluded from the feature list:

| dataset | rows | rows flagged by Tukey IQR |
|---|---:|---:|
| `bolts_sample.csv` | 60 | 25.0% |
| `bmics_sample.csv` | 30 | 20.0% |
| `bulldozers_sample.csv` | 5000 | 4.0% |
| `vehicles_sample.csv` | 5000 | 2.2% |

On the 50-300 row datasets P2Predict is built for, plain IQR touches a fifth to
a quarter of all rows. Surfacing that as an alarm produces an alarm nobody
reads, which is exactly the kill criterion this document set for itself.

**Rejected: an IQR-based magnitude threshold.** A robust z-score,
`(max|x| - median) / IQR`, separated the synthetic cases cleanly, with collapse
between z=97 (Ridge still selected) and z=195 (Ridge disqualified). It does not
transfer. `threads_per_inch` in the aerospace-fasteners sample sits at z=502,
well past the synthetic boundary, and is *fine*: rerunning that study with
`winsorize` moves Ridge's CV from -0.0866 to -0.0784 and changes neither the
selected model nor the error. One real column past a synthetic threshold and
unaffected is enough to disqualify the threshold. The statistic also reports
infinity for any near-constant column (zero IQR), which false-fired on
`op_temp_min_C` in the battery-IC sample.

**Implemented: a magnitude ratio, with the threshold set from data.**
`max|x| / median(|x|)` over non-zero values, which is immune to the zero-IQR
case. Measured:

| column | magnitude ratio |
|---|---:|
| `threads_per_inch` (bolts, genuine 4040 among 10-32) | 168 |
| `max_cells_supported` (bmics) | 16 |
| `age_at_sale` (bulldozers) | 6.1 |
| `odometer` (vehicles) | 4.9 |
| `Weight` (examples) | 2.3 |
| **the RDKit `Ipc`-shaped column that started this** | **6.4e29** |

`EXTREME_MAGNITUDE_RATIO = 1e6`, roughly 5,900x above the highest ratio in any
real column measured and far inside the range where the collapse is certain
rather than borderline. It is silent on all five case-study samples.

Wired into both agent-facing tools:

* **`propose_training_plan`** now inspects the specs it would actually train on
  and, when a column trips the threshold, appends a plain-language question to
  `questions_for_the_user`. That is the call the agent is already required to
  make before training, so the decision lands in front of the user at the one
  moment it is supposed to.
* **`train`** captures the summary instead of discarding it, returns it as
  `feature_data_quality`, and adds a line to `warnings` naming the column and
  the policy that fixes it.

Both also return the ordinary per-column IQR counts as structured data, which
is context an agent can reason over without it being shouted.

**Not changed: the default.** `feature_outlier_policy` stays `warn`. Switching
it to `winsorize` would silently mutate a user's data without asking, which
contradicts the tool's stance of explaining rather than deciding. Surfacing is
reversible; a changed default is a behaviour change on every existing workflow.

## What would kill this

* ~~If the per-column summary turns out to be noisy on real procurement
  data...~~ **This fired.** It is why the raw summary is returned as structured
  context rather than as the alarm, and why the alarm keys on magnitude.
* If agents given the flag still do nothing with it, the problem is the
  guidance rather than the plumbing, and the fix belongs in the MCP tool
  descriptions instead. Not yet observed either way.
* If a real customer dataset legitimately carries a ratio above 1e6 (a column
  genuinely spanning six orders of magnitude, with no error in it), the
  threshold is wrong rather than the idea. Nothing in the case studies comes
  within three orders of magnitude of it, but the case studies are five
  datasets.

## Scope note

This is not the out-of-domain case. A part whose specs sit far outside training
range at *predict* time is already capped by the `in_domain` block shipped in
#40. This finding is about training-time model selection, where no such guard
exists, and the failure direction is a quietly weaker model rather than a
quietly wrong price.
