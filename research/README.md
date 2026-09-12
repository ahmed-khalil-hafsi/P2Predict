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
| `bias_gate_materiality.{md,py,json}` + `bias_gate_equivalence.{py,json}` | **Open** — the `unreliable` verdict tracks holdout size, not bias magnitude: same model and same bias, flagged 10% of the time at n=50 and 100% at n=16,000. The 2026-09-11 addendum replaces the proposed effect-size floor with an **equivalence test** on the median relative residual, derives the 5% band, and dissolves the log-space and small-n items into it; a same-day revision at the end splits the can't-certify case into four states, because at the 20–60 part holdout most users actually have, three states hand every model the same `unknown`. Measured: false-flag 37.3%→0.3%, miss 42.0%→5.7%, inversion 0.99→0.36. |
| `methodology_review.md` + `benchmark_methodology_gaps.py` + `methodology_gaps_benchmark_results.txt` | **Open (partly superseded)** — July methodology audit. Findings 3 & 4 still open; 1 superseded by `log_retransformation_bias`, 2 sharpened by `bias_gate_materiality`, the OOD bullet shipped as #36. See its Status header. |
| `out_of_domain_flag.{md,py,json}` | **Open** — nothing in the predict path notices a part outside the data the model was built on; on a well-fit model an impossible part earns a `trust` verdict. Proposes an `in_domain` block that caps the verdict. |
| `whatif_reliability_flag.md` | **Shipped (#35).** Retained only because the CHANGELOG cites it as rationale. |
