# External validation: P2Predict on the TDC ADMET benchmarks

**Status: validation, not a proposal. No core code changed, and none is
proposed here.**

Everything in `research/` is a finding that argues for a change to core. This
document is a different genre and is filed here for that reason: it is a
*measurement* of P2Predict 1.1.0 against a benchmark nobody involved with the
project controls, and it stays true (as a record of that version) whether or
not anything is ever done about it. It is meant to be re-run on each release —
see [`README.md`](README.md).

It did generate two proposals. Those live in `research/`, where proposals
belong:

* [`../research/algorithm_shelf.md`](../research/algorithm_shelf.md) — what the
  29-of-30 XGBoost result does and does not say about `auto_train`. Decided
  2026-09-15: CatBoost/LightGBM and blending declined, cross-fitting the
  conformal calibration carried forward.
* [`../research/agent_as_featurizer.md`](../research/agent_as_featurizer.md) —
  the transferable idea: extracting compositional features from the free-text
  description column that ERP exports already carry.

**Run:** P2Predict **1.1.0**, 2026-09-15, local, CPU only, one MacBook.
**Scripts:** `featurize.py`, `run_eval.py`, `run_all.sh`.
**Raw numbers:** `results/1.1.0/*.json` — every figure below is quoted from
those files or from TDC's own `group.evaluate_many()` output inside them.

## TL;DR

P2Predict was pointed at three public pharma drug-property leaderboards with no
chemistry knowledge, no per-task tuning, and no code changes. It placed **7 of
24** and **7 of 20** on the two smaller boards and **~16 of 22** on the largest.
On Caco-2 it finished ahead of the Chemprop, Chemprop-RDKit, AttentiveFP and GCN
entries, though behind the MapLight + GNN hybrid (0.287) and MolMapNet-D (0.287);
on PPBR three graph-based entries (MapLight + GNN, MiniMol, Chemprop) finish
ahead of it. The blanket claim "beats every graph neural network" is not true on
any of the three boards and is not made here. Three things came out of
it that matter to the product:

1. **Competitiveness declines monotonically with dataset size.** It wins where
   data is scarce, which is the regime real users occupy.
2. **The `thorough` HPO budget is not reliably worth its cost.** It helped on
   one endpoint of three and cost 4.9–6.6× the compute every time. On the
   smallest dataset it *hurt* stability.
3. **Conformal coverage tracks calibration-set size exactly as theory
   predicts** — evidence the implementation is correct, and a concrete
   statement about what users at 10–60 calibration rows should expect
   (conservative, wider-than-necessary ranges).

Plus: the log-target guardrail declined the wrap on all three endpoints, for
two different correct reasons, with zero configuration.

**This is a trust artifact, not a capability claim.** The claim it supports is
"the engine is sound", evidenced by a third party with no interest in
flattering it — not "our ML is special", which invites a comparison to
AutoGluon that P2Predict would lose. The differentiation is the verdict layer,
the dollar translation, the agent interface and opinionated defaults for one
data shape, none of which this benchmark scores.

## Why this benchmark

Every P2Predict case study asks the reader to take our word for it. They cannot
audit the battery-IC study — they do not have the data, the suppliers, or the
specs. That is the structural weakness of vendor-published benchmarks: the
vendor picks the dataset, picks the metric, and scores itself.

Therapeutics Data Commons inverts all three. TDC defines the train/test split,
TDC defines the metric, and twenty-plus other teams — several of whom do
nothing but molecular ML — are scored on the same board the same way.

The question this answers is not "is P2Predict good at chemistry". It is
**"is the engine sound when it has no idea what the columns mean"**.

## What it was asked to predict

Three ADMET endpoints — properties that determine whether a drug candidate is
viable:

| Endpoint | The question | Units |
|---|---|---|
| **Caco-2** | How fast does this drug cross the gut wall into the bloodstream? | log10 cm/s |
| **PPBR** | What % of this drug binds to plasma proteins (and is therefore inactive)? | percent |
| **Lipophilicity** | How does this drug partition between oil and water? | logD |

Each molecule was flattened into ~210 RDKit 2D descriptors — molecular weight,
ring counts, topological indices, partial charges — which is structurally
identical to a parts CSV: one row per entity, one column per spec. The
featurisation step (`featurize.py`) is the only domain-specific code in the
pipeline, and it lives *outside* P2Predict.

## Results

All numbers are the official `group.evaluate_many()` output under TDC's own
protocol (scaffold split, 5 seeds, mean ± std). Best budget per endpoint shown.

| Endpoint | P2Predict MAE | Rank | Field | Beats |
|---|---|---|---|---|
| **Caco-2** | **0.289 ± 0.004** | **7 of 24** | top 29% | Chemprop, Chemprop-RDKit, AttentiveFP, GCN, DeepMol |
| **PPBR** | **7.948 ± 0.108** | **7 of 20** | top 35% | Chemprop-RDKit, AttentiveFP, GCN |
| **Lipophilicity** | **0.580 ± 0.008** | **~16 of 22** | bottom 27% | 6 of 21 |

Leaderboard positions are as published on 2026-09-15 and will move.

### Caco-2 — rank 7 of 24 (`results/1.1.0/caco2_wang_all-features_fast.json`)

| Rank | Model | MAE |
|---|---|---|
| 1 | CaliciBoost | 0.256 ± 0.006 |
| 2 | XG Boost | 0.274 ± 0.004 |
| 3 | MapLight | 0.276 ± 0.005 |
| 4 | BaseBoosting | 0.285 ± 0.005 |
| 5 | MolMapNet-D | 0.287 ± 0.005 |
| 6 | MapLight + GNN | 0.287 ± 0.005 |
| 7 | XGBoost | 0.289 ± 0.011 |
| **7=** | **P2Predict** | **0.289 ± 0.004** |
| 8 | DeepMol (AutoML) | 0.297 ± 0.008 |
| 11 | Chemprop-RDKit | 0.330 ± 0.024 |
| 14 | Chemprop | 0.344 ± 0.015 |
| 17 | AttentiveFP | 0.401 ± 0.032 |
| 22 | GCN | 0.599 ± 0.104 |

Ties rank 7 to three decimals, with a *tighter* across-seed spread than that
entry (±0.004 vs ±0.011).

### PPBR — rank 7 of 20 (`results/1.1.0/ppbr_az_all-features_thorough.json`)

| Rank | Model | MAE |
|---|---|---|
| 1 | Gradient Boost | 7.440 ± 0.024 |
| 5 | Chemprop | 7.788 ± 0.210 |
| 6 | BaseBoosting | 7.914 ± 0.096 |
| **7=** | **P2Predict** | **7.948 ± 0.108** |
| 7 | DeepMol (AutoML) | 7.990 ± 0.104 |
| 9 | Chemprop-RDKit | 8.288 ± 0.173 |
| 13 | AttentiveFP | 9.373 ± 0.335 |
| 18 | GCN | 10.194 ± 0.373 |

### Lipophilicity — rank ~16 of 22 (`results/1.1.0/lipophilicity_astrazeneca_all-features_thorough.json`)

| Rank | Model | MAE |
|---|---|---|
| 1 | MiniMol | 0.456 ± 0.008 |
| 2 | Chemprop-RDKit | 0.467 ± 0.006 |
| 3 | Chemprop | 0.470 ± 0.009 |
| 11 | GCN | 0.541 ± 0.011 |
| 14 | AttentiveFP | 0.572 ± 0.007 |
| 15 | RDKit2D + MLP | 0.574 ± 0.017 |
| **~16** | **P2Predict** | **0.580 ± 0.008** |
| 16 | Basic ML | 0.617 ± 0.003 |
| 19 | DeepMol (AutoML) | 0.656 ± 0.012 |

This is the endpoint where it loses, and it is reported for the same reason the
[aerospace-fasteners case study](../case-studies/aerospace-fasteners/) exists:
a benchmark that only shows wins is not a benchmark.

## Finding 1 — competitiveness declines monotonically with dataset size

| Endpoint | Train rows | Rank | Percentile |
|---|---:|---|---|
| Caco-2 | 637 | 7 / 24 | top 29% |
| PPBR | 1,952 | 7 / 20 | top 35% |
| Lipophilicity | 2,940 | ~16 / 22 | bottom 27% |

And Lipophilicity is the one board where graph models hold ranks 1, 2 and 3.

**Read:** P2Predict is competitive where data is scarce and falls behind once
deep models have enough rows to learn their own representations. That is not a
flaw to fix — it is the correct trade-off for a tool whose users bring 50–300
rows of purchase history. The regime where it wins is the regime it ships into.

Three points is three points: the trend is consistent with the tabular-vs-deep
literature but is not established by this run alone.

## Finding 2 — the wider HPO search is not reliably worth its cost

Times are the summed model-fitting wall clock across the 5 seeds
(`train_seconds` in the result files), on one machine, with the usual
run-to-run variance of a laptop.

| Endpoint | Train rows | `fast` | `thorough` | Verdict |
|---|---:|---|---|---|
| Caco-2 | 637 | **0.289 ± 0.004** (2.7 min) | 0.289 ± 0.012 (17.9 min) | fast wins — same mean, ⅓ the variance, 6.6× cheaper |
| PPBR | 1,952 | 8.124 ± 0.160 (7.2 min) | **7.948 ± 0.108** (35.1 min) | thorough wins — 2.2% better, tighter |
| Lipophilicity | 2,940 | 0.581 ± 0.008 (11.1 min) | **0.580 ± 0.008** (54.1 min) | tie — 0.001 for 4.9× the compute |

The tidy story ("bigger data justifies a wider search") **does not hold** — the
largest dataset showed no benefit at all. `thorough` paid off on exactly one
endpoint of three and cost 4.9–6.6× the compute every time.

**Read:** the `fast` default is well chosen. `thorough` is worth *trying*, not
assuming. On the smallest dataset it actively hurt stability — the wider search
overfits the cross-validation when there is little data to cross-validate
against.

### Model selection: XGBoost won 29 of 30 runs

Across all six sweeps (3 endpoints × 2 budgets × 5 seeds), `auto_train` crowned
**XGBoost 29 times**; one PPBR `fast` seed picked random forest. **Ridge never
won.**

Two caveats that the raw count hides, and that matter for how far this can be
carried:

* **Ridge did not merely lose, it diverged.** Its CV R² on PPBR and
  Lipophilicity reaches values like −2.4e32 (see `cv_scores` in the result
  files). The cause is in the descriptors, not the model: RDKit's `Ipc`
  descriptor reaches 3.6e31 on the Lipophilicity training set, and a scaled
  linear model extrapolating past that range produces astronomically wrong
  predictions. So "Ridge never won" is a fact about this featurisation, not a
  general statement about linear models.
* **On P2Predict's own case studies the shelf is not one algorithm deep.**
  Ridge won the 150-part battery-management-IC study; random forest won the
  80k-row used-cars study; XGBoost won aerospace fasteners and heavy equipment.
  The 29-of-30 result describes 210-column descriptor data at 637–2,940 rows.

What this does and does not imply for adding CatBoost/LightGBM or for blending
the runners-up is worked through in
[`../research/algorithm_shelf.md`](../research/algorithm_shelf.md) — including
why most of it does *not* clear the bar that prior decisions already set.

## Finding 3 — conformal coverage tracks calibration-set size, exactly as theory predicts

Nominal coverage was 90% on every run. Observed on the TDC test set, across
seeds and both budgets:

| Endpoint | Calibration rows | Test rows | Observed 90% coverage |
|---|---:|---:|---|
| Caco-2 | 91 | 182 | 94.5% – 100.0% |
| PPBR | 279 | 559 | 91.2% – 95.7% |
| Lipophilicity | 420 | 840 | 86.3% – 90.1% |

A clean monotonic convergence toward nominal as the calibration set grows. With
91 calibration points the empirical quantile is coarse and the intervals come
out **conservative and noisy**; with 420 they land close to the promise, dipping
slightly under where scaffold-split distribution shift bites hardest — which is
itself the expected behaviour when exchangeability is strained, and the
scaffold split strains it deliberately.

This is textbook split-conformal behaviour and is evidence the implementation
is correct.

**Why it matters for procurement:** at the typical 50–300-row dataset the
calibration set is 10–60 rows — below even the Caco-2 regime. Users should
expect ranges that are **wider than strictly necessary, erring safe**. That is
the right failure direction for a negotiation tool, and it is worth saying out
loud rather than letting people discover it.

**Caveat:** no leaderboard entry reports calibrated intervals at all, so there
is nothing to compare these numbers against. They are reported as a property of
our output, not as a competitive result.

## The guardrails fired correctly, unprompted

Worth recording separately, because it is the strongest evidence here about
engineering quality. Every run left `log_target` on `auto`:

* **Caco-2** — target is log10 permeability, entirely negative. The log-target
  wrap was declined by the `y > 0` safety check (`auto:skew=nan`). Applying it
  would have crashed or produced nonsense.
* **PPBR** — percentages bunched near 100, i.e. *left*-skewed
  (`auto:skew=-2.00`). The rule correctly declined the wrap for a completely
  different reason.
* **Lipophilicity** — logD spans negative values; declined again
  (`auto:skew=nan`).

Three domains, three correct refusals, zero configuration. The same rule that
protects a skewed price column protected a chemical property it has never seen.

## Methodology

The protocol is TDC's, not ours — that is the entire point.

1. TDC supplies the scaffold-split `train_val` / `test` partition. We never
   choose a split, so we cannot choose a favourable one.
2. For each of 5 seeds, TDC hands us a `train` / `valid` partition of
   `train_val`.
3. We fit on `train` **only**. `valid` is used for conformal calibration — its
   intended purpose — and never touches model fitting.
4. We predict `test` and hand the predictions to `group.evaluate_many()`.

**Leakage controls:** descriptors are computed per molecule and independently.
Column drops (mostly-missing/constant) are decided on **train only**.
Imputation uses **train medians only**. No feature selection — all ~210
descriptors offered, since the CLI's 6-feature cap is a CLI default, not an
engine limit.

**Conservative by construction:** we train on `train` only (e.g. 637 rows on
Caco-2), where some leaderboard entries likely use all of `train_val` (728).
If anything P2Predict ran with ~14% less data than the competition.

**Honest caveat:** TDC is an **honour-system leaderboard** — the test labels
ship with the data. Our protocol is clean, but "we scored ourselves" belongs in
any public writeup of these numbers.

## What is NOT novel here

State this early and plainly in anything public. The headline finding —
*descriptor-based gradient boosting matches or beats graph neural networks on
molecular property prediction* — is **already established**, not a discovery:

* Jiang et al. (2021, *J. Cheminformatics*) ran essentially this comparison and
  reached the same conclusion.
* Grinsztajn et al. (2022) and Shwartz-Ziv & Armon, on why tree models still
  beat deep learning on tabular data generally.
* **The leaderboard itself already says it** — ranks 1, 2, 3, 4 and 7 on Caco-2
  are all boosting/descriptor methods. The TDC community knows.

Adjacent claims are also already occupied: DeepMol (AutoML) is on both boards,
so "generic AutoML is competitive" is not new either; and conformal prediction
for QSAR has its own literature (Norinder, Carlsson, Alvarsson and others).

*(Citations recalled, not verified — confirm before publishing anywhere public.)*

**What this run legitimately demonstrates** is narrower and still worth having:
a domain-agnostic engine, with no per-task tuning and no domain knowledge,
reaching competitive performance on a third-party-scored benchmark — with its
guardrails firing correctly in a domain nobody configured them for.

## Limits of this validation

* **Three endpoints, all regression.** The ADMET group also contains
  classification tasks, which P2Predict does not do at all.
* **The ranks are a snapshot** of the 2026-09-15 leaderboard, against entries
  submitted under the honour system described above.
* **Timings are one laptop, one run each.** Treat the compute ratios as
  order-of-magnitude, consistent with the variance already documented in
  [`../research/large_data_scalability.md`](../research/large_data_scalability.md).
* **Nothing here measures the product.** The verdict layer, the dollar
  translation, the what-if decomposition, the MCP surface — the parts a
  category manager actually touches — are not scored by any leaderboard. This
  measures the engine underneath them.
* **The fastest route up the board is the one not to take.** Morgan/ECFP
  fingerprints are almost certainly where the top entries' edge comes from, and
  are pure chemistry feature engineering *outside* P2Predict: they would move
  the rank and leave the product exactly as good as it is today. What is worth
  taking from them is the idea, not the implementation — see
  [`../research/agent_as_featurizer.md`](../research/agent_as_featurizer.md).

## Reproducing

```bash
cd evals
./.venv/bin/python run_eval.py --benchmark Caco2_Wang --budget fast --seeds 1 2 3 4 5
./run_all.sh          # all three endpoints, both budgets
```

Setup, the install order this stack requires, and the re-run-on-each-release
protocol are in [`README.md`](README.md) and
[`requirements.txt`](requirements.txt). **These dependencies must never be
added to the root `requirements.txt` or `pyproject.toml`** — they are benchmark
tooling, not product dependencies.
