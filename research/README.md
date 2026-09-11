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

## Current contents

| File | Status |
|---|---|
| `log_retransformation_bias.{md,py,json}` | **Open** — log-target models are flagged "unreliable" for a mean-vs-median gap; proposes fixing the verdict logic + an opt-in mean correction. |
| `bias_gate_materiality.{md,py,json}` + `bias_gate_equivalence.{py,json}` | **Shipped (#39).** The `unreliable` verdict tracked holdout size, not bias magnitude. Replaced by an equivalence test on the median relative residual against a ±5% band, in a four-state form (certified / likely / likely-not / certified-not) because at the 20–60 part holdout most users have, three states hand every model the same `unknown`. Measured: false-flag 37.3%→0.3%, miss 42.0%→5.7%, inversion 0.99→0.36. Retained only because the CHANGELOG cites it as rationale. |
| `methodology_review.md` + `benchmark_methodology_gaps.py` + `methodology_gaps_benchmark_results.txt` | **Open (partly superseded)** — July methodology audit. Findings 3 & 4 still open; 1 superseded by `log_retransformation_bias`, 2 sharpened by `bias_gate_materiality`, the OOD bullet shipped as #36. See its Status header. |
| `out_of_domain_flag.{md,py,json}` | **Shipped (#40).** Nothing in the predict path noticed a part outside the data the model was built on; on a well-fit model an impossible part earned a `trust` verdict, and — because band selection keys on the predicted value — often a *narrower* band than a legitimate part. Now an `in_domain` block on the four predict tools, capping the per-part verdict. Retained only because the CHANGELOG cites it as rationale. |
| `whatif_reliability_flag.md` | **Shipped (#35).** Retained only because the CHANGELOG cites it as rationale. |
