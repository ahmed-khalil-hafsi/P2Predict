# evals/

**External validation.** A re-runnable harness that scores P2Predict against
the [Therapeutics Data Commons](https://tdcommons.ai/) **ADMET benchmark
group** — a public leaderboard where the train/test split, the metric and the
competing entries are all defined by a third party.

Read the write-up first: **[`tdc_admet_validation.md`](tdc_admet_validation.md)**.

## Why this folder exists, and why it is not `research/`

`research/` is for *open findings that propose a core change*, and its rule is
"prune when shipped". A benchmark run is the opposite genre: it proposes
nothing, it never ships, and it is most useful when it is kept and repeated.
So it lives here, at the top level, alongside the code it scores.

The second reason is dependencies. This harness needs **PyTDC** and **RDKit**,
which are external benchmark tooling and must never become P2Predict
dependencies. They are pinned in [`requirements.txt`](requirements.txt) in this
folder and installed into a **separate virtualenv**. Nothing here is imported
by anything under `src/`, and the root `requirements.txt` / `pyproject.toml`
are deliberately untouched.

Packaging is unaffected: `pyproject.toml` builds from `where = ["src"]` with
`include = ["p2predict*"]`, and pytest's `testpaths = ["tests"]`, so this
folder cannot reach the wheel, the sdist, or CI collection.

## Headline — P2Predict 1.1.0, run 2026-09-15

| Endpoint | Train rows | MAE (official TDC metric) | Leaderboard rank |
|---|---:|---|---|
| Caco-2 | 637 | 0.289 ± 0.004 | 7 of 24 |
| PPBR | 1,952 | 7.948 ± 0.108 | 7 of 20 |
| Lipophilicity | 2,940 | 0.580 ± 0.008 | ~16 of 22 |

Molecules were flattened into ~210 RDKit 2D descriptors — one row per
molecule, one column per property, structurally identical to a parts CSV — and
handed to P2Predict with no chemistry knowledge and no per-task tuning.

## Files

| File | Purpose |
|---|---|
| `tdc_admet_validation.md` | The write-up: results, three findings, methodology, caveats. |
| `featurize.py` | SMILES → ~210 RDKit 2D descriptors → flat table. The only domain-specific code, and it lives outside P2Predict. |
| `run_eval.py` | Runs one benchmark under TDC's official protocol and scores it with `group.evaluate_many()`. |
| `run_all.sh` | Full sweep: three endpoints × both HPO budgets, 5 seeds each. |
| `requirements.txt` | Eval-only dependencies and the install order they need. |
| `results/<version>/*.json` | Raw per-seed output: MAE, R², interval coverage, chosen algorithm, CV scores, timings. |

## Setup — the order matters

Built on **Python 3.13** (not 3.14 — wheel availability). Full rationale for
each line is in `requirements.txt`; the short version is that PyTDC 1.1.15 pins
stale numpy/pandas/rdkit, will drag in a scikit-learn with no cp313 wheel if
allowed to resolve its own dependencies, and still imports `pkg_resources`.

```bash
cd evals
python3.13 -m venv .venv
./.venv/bin/pip install rdkit "scikit-learn>=1.5" pandas numpy
./.venv/bin/pip install PyTDC --no-deps
./.venv/bin/pip install requests tqdm fuzzywuzzy seaborn huggingface_hub "pandas<3"
./.venv/bin/pip install "setuptools<81"
./.venv/bin/pip install ..          # P2Predict from this checkout, installed last
```

PyTDC's pin warnings about numpy/pandas/rdkit are safe to ignore — those pins
cover optional featurizers this harness does not use.

## Running

```bash
./.venv/bin/python run_eval.py --benchmark Caco2_Wang --budget fast --seeds 1 2 3 4 5
./run_all.sh          # all three endpoints, both budgets
```

Options: `--benchmark` (any TDC ADMET name), `--budget fast|thorough`,
`--seeds`, `--feature-cap N` (use P2Predict's own ranker instead of all
descriptors), `--tag`.

TDC downloads the benchmark group into `data/` on first run (gitignored).

## Re-run it on each release

Every result file records the `p2predict_version` it was produced with, and
`run_eval.py` writes into `results/<version>/`, so a new run **sits alongside**
the previous one rather than overwriting it. That is the point of keeping the
harness rather than filing a one-off report:

* **On each release**, re-run the sweep and diff the MAE against the previous
  version's directory. A regression on a third-party-scored board is worth more
  than any self-scored case study.
* **When a core change lands** — cross-fitting the conformal calibration, in
  [`../research/algorithm_shelf.md`](../research/algorithm_shelf.md), is the
  live candidate — the before/after delta here *is* the evidence for whether it
  helped. Run the baseline first, change core, run again, compare like for
  like: same endpoints, same 5 seeds, same budget.
* **Quote the version with the number.** "Rank 7 of 24" is a claim about
  P2Predict 1.1.0 on the 2026-09-15 leaderboard, not a permanent property.
  Leaderboards move.

Cost on the dev Mac: ~21 minutes of model fitting for the three `fast` sweeps,
~107 minutes for the three `thorough` ones, plus featurisation. CPU only.

## Honesty notes

TDC is an **honour-system leaderboard** — the test labels ship with the data.
The protocol used here is clean (see the write-up's methodology section), but
"we scored ourselves" belongs in any public version of these numbers. And the
headline result is **not novel**: descriptor-based gradient boosting matching
graph neural networks on molecular property prediction is already published,
and the top of the board already knows it. Both points are stated up front in
the write-up and should stay there.
