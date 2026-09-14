# Finding: at 150 parts, auto-mode's model choice is a coin flip — but a cheap one

> **Status: open.** Per the project's working agreement this note **proposes and
> changes nothing in core**. Reproduce everything below with
> `.venv/bin/python research/selection_noise.py`.
>
> **Read the conclusion before the numbers.** The instability here is real and
> larger than expected. The *cost* of it is small. This note argues for
> **surfacing** the uncertainty, not for changing the selection rule — the
> evidence does not clear the project's bar for touching core.

`auto_train` scores ridge, random forest and XGBoost by cross-validation and
keeps the argmax:

```python
if score > best_score:                      # training.py:203
    best_score, best_model, best_algorithm = score, model, algorithm
```

Nothing in that comparison knows how *precise* the scores are. And on the
default `fast` budget the CV is **3-fold**
([`training.py:93`](../src/p2predict/training.py)), so on a 150-part catalog
each family's score is an average over three folds of fifty rows. The winner
is then reported to the user as the chosen model, and its CV score as
evidence.

## Method

Three case-study datasets subsampled into synthetic catalogs of 75 / 150 /
300 parts, 40 resamples each. Each draw is graded on a **disjoint 3,000-row
evaluation sample**: let the rule choose, then score every candidate family
on rows it has never seen.

Three rules on identical draws:

| rule | behaviour |
|---|---|
| `argmax` | what ships today |
| `one_se` | among families within 1 SE of the best mean CV score, take the simplest (ridge < random forest < XGBoost) |
| `ridge` | never choose at all — the do-nothing baseline |

Two things are measured that the shipped path cannot see:

- **Selection stability** — re-run the same CV on the same rows with a
  different fold shuffle. Any change in the winner is selection noise and
  nothing else.
- **Regret** — median APE of the chosen family minus median APE of the family
  that actually generalised best, in percentage points. This is what the
  noise *costs*.

Two deliberate choices. Tuning is **off** — every family gets library
defaults — which isolates the selection question and makes the result a lower
bound, since the shipped path also picks hyperparameters from the same small
CV. And the **grader is median APE, not R²**: price-space R² is unbounded
below, and on the heavy-tailed fasteners catalog a single exploded prediction
drives a family's score to −10⁶⁰ and makes every average meaningless. That
explosion rate is reported separately below — it is a finding of its own.

## Measured

Full results: [`selection_noise_results.json`](selection_noise_results.json),
which also carries per-trial detail so a re-analysis never needs a re-run.

| dataset | n | margin inside 1 SE | winner changes on reshuffle | best-to-worst family gap | `argmax` picked best | `argmax` regret | `one_se` picked best | `one_se` regret | `ridge` picked best | `ridge` regret |
|---|---|---|---|---|---|---|---|---|---|---|
| used cars | 75 | 65% | **45%** | 4.7pp | 52% | 1.1pp | 60% | 1.0pp | 38% | 3.5pp |
| used cars | 150 | 62% | 28% | 4.4pp | 78% | 0.5pp | 80% | 0.6pp | 8% | 4.0pp |
| used cars | 300 | 52% | 22% | 3.4pp | 80% | 0.4pp | 82% | 0.5pp | 0% | 3.4pp |
| heavy equipment | 75 | 32% | 22% | 4.2pp | 80% | 0.7pp | 82% | 0.5pp | 100% | 0.0pp |
| heavy equipment | 150 | 28% | 18% | 2.7pp | 80% | 0.3pp | 82% | 0.1pp | 85% | 0.0pp |
| heavy equipment | 300 | 20% | 5% | 1.8pp | 70% | 0.2pp | 72% | 0.2pp | 72% | 0.2pp |
| aerospace fasteners | 75 | 28% | **40%** | 4.8pp | 62% | 2.4pp | 78% | 1.7pp | 8% | 3.2pp |
| aerospace fasteners | 150 | 32% | **40%** | 3.8pp | 72% | 0.9pp | 80% | 0.6pp | 2% | 3.3pp |
| aerospace fasteners | 300 | 52% | 28% | 4.2pp | 78% | 0.7pp | 82% | 0.4pp | 8% | 3.5pp |

Averaged over the nine cells:

| rule | picked the best family | mean regret | worst cell | p90 regret |
|---|---|---|---|---|
| `argmax` (ships today) | 72% | 0.81pp | 2.4pp | 2.6pp |
| `one_se` | 78% | 0.63pp | 1.7pp | 2.2pp |
| `ridge` (do nothing) | 36% | 2.36pp | 4.0pp | 4.6pp |

### The instability is real

Reshuffling the cross-validation folds — **same rows, same data, same
code** — changes which family auto-mode selects in up to **45%** of draws.
The winner's margin over the runner-up is smaller than one standard error of
its own fold scores in **20–65%** of draws. On the two hardest datasets at 75
parts, the choice is close to a coin flip between whichever two families
happen to be in front.

### The cost is not

Median regret for `argmax` is **0.0pp** in every single cell, and the mean is
**0.81pp** of median APE. The reason is visible in the table: the three
families land within 1.8–4.8pp of each other, and a noisy argmax over three
similar candidates still avoids the worst one most of the time. The noise is
real; it is mostly choosing between near-ties.

### The 1-SE rule is better, and not by enough

`one_se` beat `argmax` on hit rate in **all nine cells** and on mean regret in
six of nine, for a net improvement of **0.18pp** of median APE. It also picks
simpler models, which are faster and easier to explain. But 0.18pp is not a
number that justifies changing which model every existing user gets. Recorded
so the option is on the record, not re-litigated later.

Doing nothing at all is clearly worse: always-ridge gives up **2.36pp**, so
the selection loop is earning its keep even while being noisy.

## Sub-finding: ridge emits absurd prices on a heavy-tailed catalog

Not what this study went looking for. On aerospace fasteners, **ridge under
the log-target wrap produced at least one evaluation price absurd enough to
drive price-space R² below −10 in 68–75% of draws**, at every catalog size.
Random forest and XGBoost: 0%.

The mechanism is the log wrap — a large log-space prediction exponentiates
into an astronomical price — and it is the same mechanism behind the 10¹²⁰ %
interval half-width in
[`small_n_conformal.md`](small_n_conformal.md).

Two things are worth separating:

- **Auto-mode is safe.** Argmax on R² is brutally punished by one exploded
  prediction, so it picks the exploding family in only **2–5%** of draws. The
  selection rule is inadvertently acting as a filter.
- **An explicit `--algorithm ridge` is not.** Nothing between the estimator
  and the user checks that a returned price is finite or plausible, and the
  out-of-domain flag does not catch it (the *input* is in-domain; the
  *output* is the problem).

This likely deserves its own follow-up — a sanity bound on the
back-transformed prediction — rather than being bundled into a selection
change.

## Recommendation: surface it, don't change it

Per the project's rule that core changes need a significant capability gain,
**none of the three rules clears the bar on accuracy.** What is missing is not
a better choice but an honest one.

Suggested shape, all additive:

- `auto_train` already computes `scores` per algorithm. Retain the **per-fold**
  scores alongside them and derive a standard error.
- Add a `selection` block to the `--json` payload and to `get_model_quality`:
  the per-family scores, the winner's margin, and whether that margin exceeds
  1 SE — plus the `say_to_user` sentence the MCP contract requires, e.g.
  *"Random forest was chosen, but ridge scored close enough that the
  difference is within the noise of a catalog this size — either would give
  you a similar answer."*
- Nothing about which model is selected changes. No existing model, script or
  case study behaves differently.

This is worth doing on its own terms: a tool whose entire positioning is the
honesty layer currently reports a model family as a decision when at 75 parts
it is frequently a coin flip.

## What this does not cover

Hyperparameter selection inside `HalvingRandomSearchCV` runs on the same small
CV and was held out of this study (tuning is off throughout) — so every number
here is a **lower bound** on the total selection noise in the shipped path.
The interaction between selection noise and conformal calibration is also
untouched: both currently ride the same rows.
