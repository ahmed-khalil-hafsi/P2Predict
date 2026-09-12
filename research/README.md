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
| `out_of_domain_flag.{md,py,json}` | **Open** — nothing in the predict path notices a part outside the data the model was built on; on a well-fit model an impossible part earns a `trust` verdict. Proposes an `in_domain` block that caps the verdict. |
| `whatif_reliability_flag.md` | **Shipped (#35).** Retained only because the CHANGELOG cites it as rationale. |
