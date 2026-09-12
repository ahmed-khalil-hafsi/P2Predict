# Finding: nothing in the predict path notices a part the model has never seen

> **Status: shipped in #40.** Retained, rather than pruned, because the CHANGELOG
> cites this document as the rationale for the shipped `in_domain` block. The fix
> followed the sketch below: stored numeric ranges + the encoder's own categories,
> an `in_domain` block on the four predict tools, and a cap that stops an
> out-of-domain part ever returning `trust`.

Every answer P2Predict ships is auditable — the attribution decomposes, the
interval has a coverage proof, the what-if delta sums. All three audits answer
*"how good is this answer?"*. None answers the question that comes first:
**was this part answerable at all?**

Ask any model for a part whose specs sit outside the data it was built on and
you get a confident number, a likely-range, and a reliability verdict — with
nothing anywhere in the payload indicating that the specs were never observed.
On a well-fit model the verdict on a physically impossible part is `trust`:

> "Tight range — you can benchmark against this number with confidence."

Per the project's working agreement this note **proposes and changes nothing in
core**. It quantifies the gap and sketches the fix for a follow-up PR.
Reproduce everything below with `python research/out_of_domain_probe.py`.

## Why the shipped flag cannot catch this

`interval_reliability(low, prediction, high)` (`quality.py:114`) takes three
numbers. None of them is the input. The width it judges comes from the
conformal quantile, and band selection is
`np.searchsorted(edges, preds)` (`intervals.py:279`) — keyed on the **predicted
value alone**.

Under a log-target the bounds are `pred * exp(±q_hat)`, so the ratio the
verdict thresholds on,

```
(high - low) / prediction  =  exp(q_hat) - exp(-q_hat)
```

has the prediction cancel out entirely. It is constant within a band. With
three bands, **a model has at most three distinct verdicts it can ever emit**,
selected by predicted price. Two parts that price to the same number get the
same verdict — one a catalog staple, the other impossible. The flag grades the
*model's segment*, not the part, and was never designed to do otherwise.

`intervals.py` says so in its own module docstring: the guarantee holds
"under the assumption that future inputs come from the same distribution as the
training data." That precondition is documented and never checked.

## Measured — a model good enough to be trusted

A 1,200-row bracket-pricing model, holdout R² 0.978, 300 calibration points.
Deliberately easy: strong signal, 5% noise. That is the point — this is the
regime where the flag is *supposed* to earn a buyer's trust. Training data
spans mass 0.2–4.0 kg, 2–12 machined holes, 0.05–0.25 mm tolerance, three
materials, four suppliers; observed prices €22–€325.

| Part | Price | Likely range | Width | Verdict |
|---|---|---|---|---|
| in-domain: 1.5 kg alu bracket | €63.55 | 55.36 – 72.95 | 28% | `trust` |
| unseen supplier (new vendor) | €65.95 | 57.45 – 75.70 | 28% | `trust` |
| unseen material (titanium) | €63.82 | 55.60 – 73.26 | 28% | `trust` |
| **900 kg** — 225× the observed max | €91.94 | 82.49 – 102.46 | **22%** | `trust` |
| 900 kg, 4,000 holes, 0.00001 mm tol, "Unobtainium", unseen supplier | €126.04 | 109.98 – 144.45 | 27% | `trust` |

The last row is not a part. A 0.00001 mm tolerance is 10 picometres — smaller
than an atom. It priced at €126.04 and was described to the agent as a number
to benchmark against with confidence.

Note the fourth row: the 900 kg bracket gets a **narrower** band than the
legitimate part (22% vs 28%), so it reads as *more* trustworthy. That is the
banding mechanism working exactly as designed — the part priced into a
better-sampled price segment, and inherited that segment's confidence.

## Measured — the real case-study models

| Model | R² | Part | Width | Verdict |
|---|---|---|---|---|
| used cars (80k rows) | 0.771 | in-domain baseline | 112% | `quote` |
| | | unseen manufacturer | 112% | `quote` |
| | | year = 2× observed max | **80%** | `quote` |
| | | every spec out of domain | 112% | `quote` |
| heavy equipment (80k rows) | 0.738 | in-domain baseline | 116% | `quote` |
| | | unseen product_group | 116% | `quote` |
| | | age_at_sale = 2× observed max | **89%** | `quote` |
| | | every spec out of domain | 89% | `quote` |
| aerospace fasteners (37k rows) | 0.021 | in-domain baseline | 1916% | `quote` |
| | | every spec out of domain | 1007% | `quote` |

Three things worth separating out:

**1. The verdict is saturated.** Every row is `quote`, including the ordinary
median-spec baseline. On these models the flag is not distinguishing anything —
a buyer asking about a perfectly normal 2015 sedan is told to go get a quote.
That is a defensible conservative default, but it means the flag currently
carries no information on the project's own flagship case studies. (The
fasteners model at R² 0.021 is separately not fit to quote from at all.)

**2. Out-of-domain again reads *safer*.** Used cars: 112% in-domain → 80%
out. Heavy equipment: 116% → 89%. Consistent with the synthetic result and with
the mechanism — the direction of the error is toward false confidence.

**3. The prediction itself won't betray the extrapolation.** `year = 2×` and
`year = 50×` the observed max return **bit-identical** predictions ($22,929.21
on used cars; $13,239.45 on heavy equipment). A tree has no split beyond its
training range, so everything past the last threshold lands in the same leaf.
The model cannot express "much further out" — it silently returns its edge
value. There is no runaway number for an agent to notice.

## Why the agent path is exposed and the CLI is not

The information needed for the categorical half of this check is **already
computed and already thrown away**. `extract_feature_info()`
(`model_utils.py:18`) returns `(feature_types, all_categories)`. All six MCP
call sites discard the second value:

```
server.py:236   feature_types, _ = extract_feature_info(pipeline)
server.py:291   feature_types, _ = extract_feature_info(pipeline)
server.py:377   feature_types, _ = extract_feature_info(pipeline)
server.py:451   feature_types, _ = extract_feature_info(pipeline)
server.py:515   feature_types, _ = extract_feature_info(pipeline)
server.py:624   feature_types, _ = extract_feature_info(pipeline)
```

`cli/predict.py:476` keeps it — but only to build an interactive menu
(`choices=[str(c) for c in all_categories[feature]]`, line 661). Picking from a
list makes an unseen category *structurally unreachable*, which is why this has
never surfaced through the CLI. The agent passes a free-form dict, so the
railing that protects the CLI user isn't there. The agent path is the product
now.

The encoder's own behaviour is reasonable and is not the bug: an unseen
category encodes to the target mean (`preprocessing.py:68`, `:91`), i.e. *"a
supplier I've never seen prices like the catalog average."* That is a sound
modelling default. Reporting it as that specific supplier's price, with no
mention that the supplier was never observed, is the gap.

## Why this one matters more than a wide interval

The conformal guarantee is explicitly conditional on exchangeability. An
out-of-domain part is precisely the case where that condition fails — so it is
simultaneously the one input where the interval means nothing and the one input
where nothing warns. Every other reliability signal in the product degrades
gracefully; this one inverts.

And the everyday version is not exotic. "What should we pay for this part from
a supplier we haven't bought from before?" is a routine category-manager
question, and it is the unseen-category row in the table above. P2Predict
answers it today with the catalog average, presented as that supplier's price.

## Sketch of the fix (for the follow-up PR, not this one)

Additive, mirroring the `what_if` reliability flag shipped in v1.0.1 — same
shape, same plain-language contract, no change to the math.

- **Store the domain at train time.** Per-numeric-feature min/max alongside the
  categories the fitted encoder already holds. A few floats in model metadata.
- **Check it at predict time.** Add an `in_domain` block to `predict`,
  `predict_interval`, `predict_batch` and `predict_from_csv` naming which specs
  fell outside and by how much ("mass_kg 900 is 225× the largest part in the
  data"; "supplier 'Zeta Werke' was never in the data — priced as the catalog
  average").
- **Let it cap the verdict.** An out-of-domain part cannot return `trust`; a
  wholly out-of-domain part returns `quote` with a reason that names the spec,
  rather than a reason inferred from band width.
- **Old models keep working.** Categories are already in the fitted encoder, so
  the categorical check works on every model on disk with no retrain. Numeric
  ranges are not stored — but `background_sample` holds 100 training rows
  (verified on the heavy-equipment model), so a documented, conservative
  approximation is available; a model without one reports `domain: unknown`
  rather than a false all-clear.

The two halves are separable: the categorical check needs no format change at
all and covers the common case.

## Limits of this finding

- The synthetic model in Part A is synthetic *on purpose* — none of the real
  case-study models is tight enough to reach `trust`, so a real model could not
  demonstrate the false-confidence case. The mechanism is the same either way,
  and the analytic argument above does not depend on the data.
- A domain check catches *extrapolation*, not *wrongness*. A part comfortably
  inside the ranges can still be mispriced, and this flag would say nothing.
- Per-feature min/max is a box, not a hull. A part that is in-range on every
  spec individually but an unobserved *combination* (a 4 kg bracket with a
  0.05 mm tolerance, where every such part in the data was small) still passes.
  A box check is cheap, explainable to a buyer, and catches the cases measured
  here; a hull or density estimate is a much larger commitment for the
  remainder.
