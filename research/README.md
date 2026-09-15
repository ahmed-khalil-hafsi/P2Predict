# research/

Working notes for **findings before core changes** — the analysis that
proposes a P2Predict improvement, with reproducible evidence, so it can be
reviewed and discussed *before* any core code is touched.

## What lives here

- **Open findings only.** A finding stays here while it's being proposed and
  discussed. Each is a short write-up, ideally paired with a re-runnable
  script and its captured results.
- **Prune when shipped.** Once a finding is implemented in core, delete it —
  git history preserves it. The one exception is a doc another permanent
  record points to (e.g. a CHANGELOG entry citing it as rationale); mark that
  one `Status: shipped in #NN` and keep it, so the reference doesn't dangle.

Keeping this folder lean is deliberate: a findings folder that accumulates
shipped or stale analysis stops being useful and starts being noise.

**Not here:** external benchmark runs. Those live in
[`../evals/`](../evals/) — a permanent, re-runnable record that proposes
nothing and is never pruned, with its own isolated dependencies (PyTDC, RDKit)
that must never become product dependencies. The two proposals below came out
of the run written up in
[`../evals/tdc_admet_validation.md`](../evals/tdc_admet_validation.md).

## Current contents

| File | Status |
|---|---|
| `algorithm_shelf.md` | **Decided (2026-09-15) — two declined, one carried forward.** The TDC run crowned XGBoost in 29 of 30 model selections and Ridge never, which suggested three changes to `auto_train`: add CatBoost/LightGBM, blend the runners-up, cross-fit the conformal calibration. The first two are **declined**. The ROADMAP's standing LightGBM/CatBoost decline holds: this run measured neither library, and Ridge's losses are partly an `Ipc`-magnitude artifact rather than a predictive one — Ridge, RF and XGBoost each win one of our own case studies, so the shelf is not one deep where it matters. Blending is declined too: the runner-up trails by 0.015–0.059 CV R² in every sweep, and the artifact contract assumes a single pipeline. Both stay filed as harness experiments behind a pre-registered adoption gate, so reopening either is a decision, not a re-investigation. **Carried forward:** cross-fitting the calibration — Finding 3 measured intervals going conservative and noisy at 91 calibration rows, and the typical catalogue calibrates on 10–60. Evidence in `../evals/`; nothing in core changed. |
| `agent_as_featurizer.md` | **Open — blocked on validation data, which is the first task.** Fingerprints beat descriptors because aggregate scalars miss *compositional* signal; a part is the same, and that signal sits unused in the free-text description column of most ERP exports (`M8x40 HEX BOLT ZINC PLATED GRADE 8.8`). Extraction is a language task, so the agent does it at `propose_training_plan` — but must emit a *deterministic, frozen extractor* replayed at inference like the feature list and calibration already are, and 5–10 judged flags rather than a 2048-bit fingerprint. No dataset yet proves it: aerospace-fasteners is already structured by its prep script, and the three partial candidates (BMIC descriptions, dropped heavy-equipment `fiProductClassDesc`, Craigslist text) each fail a different part of the test. Sequence and kill criteria in the doc. |
| `large_data_scalability.{md,py,txt}` | **Decided (2026-09-15) — one item scheduled, five declined.** P2Predict is documented and sized around the typical 50–300 part category, but should work equally well at hundreds of thousands of rows. The maths already does; the agent-facing plumbing does not (a 100k-row `predict_from_csv` returns ~37 MB of JSON where the CLI has always written results to a file). That inconsistency is ROADMAP item 8. The other five findings — uncapped `include_holdout`, full-data feature ranking, artifact bloat, band count pinned at 3, random_forest dominating `auto_train` — are declined as no-ops for the typical catalogue. Kept, not pruned: the measurements mean picking one up later is a decision, not a re-investigation. |
| `log_retransformation_bias.{md,py,json}` | **Open** — log-target models are flagged "unreliable" for a mean-vs-median gap; proposes fixing the verdict logic + an opt-in mean correction. |
| `bias_gate_materiality.{md,py,json}` + `bias_gate_equivalence.{py,json}` | **Shipped (#39).** The `unreliable` verdict tracked holdout size, not bias magnitude. Replaced by an equivalence test on the median relative residual against a ±5% band, in a four-state form (certified / likely / likely-not / certified-not) because at the 20–60 part holdout most users have, three states hand every model the same `unknown`. Measured: false-flag 37.3%→0.3%, miss 42.0%→5.7%, inversion 0.99→0.36. Retained only because the CHANGELOG cites it as rationale. |
| `methodology_review.md` + `benchmark_methodology_gaps.py` + `methodology_gaps_benchmark_results.txt` | **Open (partly superseded)** — July methodology audit. Findings 3 & 4 still open; 1 superseded by `log_retransformation_bias`, 2 sharpened by `bias_gate_materiality`, the OOD bullet shipped as #36. See its Status header. |
| `out_of_domain_flag.{md,py,json}` | **Shipped (#40).** Nothing in the predict path noticed a part outside the data the model was built on; on a well-fit model an impossible part earned a `trust` verdict, and — because band selection keys on the predicted value — often a *narrower* band than a legitimate part. Now an `in_domain` block on the four predict tools, capping the per-part verdict. Retained only because the CHANGELOG cites it as rationale. |
| `whatif_reliability_flag.md` | **Shipped (#35).** Retained only because the CHANGELOG cites it as rationale. |
