# Finding: the description column is unpriced compositional signal — and the agent is the featurizer

**Status: open proposal, blocked on validation data — which is the first task,
not the last.** **No core code changed**, and none should be until there is a
dataset that can tell whether this works. Everything below is a design, an
honest inventory of what could test it, and a list of what would kill it.

Origin: the TDC ADMET validation
([`../evals/tdc_admet_validation.md`](../evals/tdc_admet_validation.md)). This
is the idea that run generated which is worth more than the leaderboard rank it
suggested chasing.

## TL;DR

On the molecular boards, the entries above P2Predict almost certainly win with
Morgan/ECFP **fingerprints** rather than a better booster. The transferable
lesson is not "add fingerprints" — that is chemistry feature engineering
outside the product and would move a vanity metric. It is *why* they win:

> **Aggregate scalars miss compositional signal.** Descriptors say how heavy
> and greasy a molecule is. Fingerprints say what it is *made of*.

A part is the same. It is not a bag of averages; it is a set of priced
features — four tapped holes, a zinc coating, a weld seam, a tolerance callout.
And that signal already sits unused in nearly every customer's ERP export, in
the free-text description column:

> `M8x40 HEX BOLT ZINC PLATED GRADE 8.8`

Five priced features in one string. Today P2Predict either drops that column or
lets it be misused as a high-cardinality categorical.

**The architectural point:** in chemistry you need RDKit, because substructure
extraction is a deterministic chemical algorithm. In procurement the extraction
is a **language task** — which the agent in front of P2Predict is already
better at than any parser we could write. It knows what grade 8.8 means, and
that this customer writes "ZN PL" in half the rows.

Two design constraints decide whether it works at all (schema freeze,
flag budget), and one honest problem blocks it (no validation data yet).

## What happens to that column today

Precise, because "it's dropped" is only half true.

* **`propose_training_plan` excludes it, by design and with a reason**
  (`mcp/server.py`): a non-numeric column where almost every row is unique is
  reported under `i_am_leaving_out` as *"Looks like an ID / free-text column
  (almost every row is unique), not a spec the model can learn from."* That is
  correct as far as it goes — as a *raw* column it is an ID.
* **If a user passes it to `train` explicitly**, it is treated as a categorical
  and target-encoded (`preprocessing.py`). With near-unique categories the
  `smooth="auto"` empirical-Bayes shrinkage pulls every one of them toward the
  global mean, so the column contributes approximately nothing. Harmless, and
  also useless.

So the situation is not that the signal is being mangled — it is that the
product correctly identifies the column as unusable *in the form it arrives in*
and stops there. The proposal is to change what happens after that sentence.

## The proposal

`propose_training_plan` is already the mandated first call before `train`, and
it is already the place where P2Predict tells the user what it will and will
not use. It becomes the hook: when it finds a free-text column, instead of only
excluding it, the agent reads a sample of the values, proposes **5–10 extracted
flags with plain-language names**, and the user confirms them the same way they
confirm the feature list today.

Downstream, nothing else changes shape: the flags are ordinary columns, the
existing attribution machinery prices them, and the user gets *"zinc plating
+$0.12"* — a line item a category manager can argue with in a negotiation,
which is the entire product thesis.

## Design constraint 1 — freeze the extraction, or the model silently breaks

**This is the real engineering risk and should be designed first.** A model's
features, background sample, calibration residuals and numeric domain are all
frozen into the artifact today (`trained_model_io.Serialize_Trained_Model`) and
replayed at inference. An extraction schema must join them. If the agent
extracts differently at predict time than at train time — a different flag set,
a different threshold for "coated", a different model version, a different mood
— the model does not error, it quietly prices a part against features that no
longer mean what they meant.

Two ways to freeze it:

**(a) Store the instructions, re-run the agent at predict time.** Natural, and
wrong. It is non-deterministic, it is not auditable, it costs a model call per
prediction, it drifts with whatever LLM is in front of the server that week,
and it does not work at all in the CLI or in any offline batch path.

**(b) Have the agent emit a deterministic extractor once, freeze *that*, and
have P2Predict execute it.** Recommended. The agent's output is not the
features, it is an ordered, inspectable rule set — token/regex patterns mapping
to named flags, with the synonyms and misspellings it observed in this
customer's data folded in (`ZINC PLATED | ZN PL | ZNPL → zinc_plated`). That
rule set is stored in the artifact, applied identically at train and predict
time, runs offline, costs nothing per row, and can be printed to a buyer or
diffed in review.

The agent is a **compiler**, not a runtime. That single choice removes most of
the risk in this proposal.

What the frozen schema must carry: the flag names and their definitions in the
user's language, the rules, the sample of source strings it was derived from,
the date and the extracting model, and the column it applies to. What the
predict path must do: apply it, and **fail loudly** if the source column is
missing or renamed — never silently emit zeros, which is a confidently-wrong
prediction of exactly the kind the `in_domain` work (#40) was built to stop.

There is a natural extension there: a description containing none of the known
tokens is the textual equivalent of an out-of-domain part, and should feed the
existing `in_domain` block rather than passing as "no flags set".

## Design constraint 2 — do not copy the fingerprint literally

ECFP uses 2048 sparse bits. That works on thousands of molecules. On a 100-part
dataset, 200 binary flags overfit catastrophically — and 50–300 parts is the
typical case.

The adaptation is **5–10 high-value flags chosen with judgment**, which is
precisely what an agent can do and a hash function cannot. Guardrails that
belong in the design:

* **Support floor and ceiling.** A flag set on 2 of 150 rows, or on 148 of 150,
  prices nothing. Require a meaningful share of rows both ways.
* **No duplicates of existing specs.** If the CSV already has a `finish`
  column, extracting `zinc_plated` from the text adds collinearity, not signal.
  Check against the structured columns before proposing.
* **Priced things only.** A flag has to be something a buyer would pay
  differently for. "Contains the word ASSY" is not.
* **Live inside the existing feature discipline.** Extracted flags compete with
  structured specs through the same ranker and the same `max_features` cap;
  they do not get a private allowance.

## The honest open problem: there is no validation dataset for this yet

Every existing case study runs on already-structured data, and the obvious
candidate is not one:

* **Aerospace fasteners is NOT a testbed.** `case-studies/aerospace-fasteners/prepare_data.py`
  already pivots the FLIS characteristics file into ten clean structured specs
  (`material`, `thread_diameter_in`, `head_style`, `finish`, `thread_class`, …).
  The extraction has been done, by hand, in the prep script. Its low R² is
  genuine data noise — which is that study's whole point — not missing
  compositional signal.

What does exist, and is worth being precise about, because "no data at all" is
not quite true either. Three partial candidates, none of them clean:

| Candidate | The text | Why it is partial |
|---|---|---|
| **Battery-management ICs** (150 rows — the right size) | Raw `bmics.csv` carries a `description` column dropped as bookkeeping: `IC BATT PROT LI-ION 1CELL 6WSON`, `IC BATT CONTRL LI-ION 1CELL 8DFN` — 109 unique strings over 150 rows, with exactly the abbreviation drift the proposal is about (`CNTL` vs `CONTRL`, `1CEL` vs `1CELL`). | DigiKey generates that string *from* the structured parametric fields, so extraction can at best **recover** what the curated columns already hold. It cannot demonstrate lift. |
| **Heavy equipment** (80k rows) | Raw Blue Book carries `fiProductClassDesc` (`Wheel Loader - 110.0 to 120.0 Horsepower`, `Hydraulic Excavator, Track - 12.0 to 14.0 Metric Tons`) and `fiModelDesc`, all dropped by `prepare_data.py`, which keeps only group / size / enclosure / state / age. | This is the one case where the text holds a spec the kept columns do **not** — a numeric size class, on a dataset where `product_size` is "unknown" on 53% of rows. But it is 80k rows, not the typical regime, and it is regular enough that a regex would parse it — so it tests "extraction adds signal", not "an agent beats a parser". |
| **Used cars** (Craigslist) | Raw file carries a free-text `description`, dropped. | Genuinely messy seller prose — and therefore likely to contain the asking price, i.e. leakage. Consumer resale, not a parts catalogue. Weakest of the three. |

**So the first task is "find or construct validation data", not "build the
feature."** A defensible sequence:

1. **Fidelity, on the BMIC study.** Extract flags from `description` alone and
   check them against the structured columns that generated it. This measures
   the mechanism's *accuracy* against ground truth, which nothing else here can.
2. **The description-only arm, also on BMIC.** Train on extracted flags with
   the curated specs withheld, and compare against the published model. This is
   the actual customer situation — an ERP export where the description is all
   there is — and a result at parity would already be worth having.
3. **Lift, on heavy equipment, subsampled** to 200–500 rows per product group
   so it sits in the regime the product is sized for. Does extracting the
   horsepower/tonnage class from `fiProductClassDesc` beat the shipped feature
   set? Compare against a regex baseline, not only against nothing — if a
   regex gets there, the agent is not the load-bearing part.
4. **A constructed proof-of-mechanism**, if 1–3 are inconclusive: take a
   structured case-study dataset, render realistic abbreviated descriptions
   from a subset of the specs, hide those specs, and measure how much of the
   lost R² extraction recovers. Synthetic, so it proves the mechanism and
   nothing about real ERP text.
5. **Only then** design the artifact change, and only then touch core.

A real customer export remains the only test that fully counts. Steps 1–4 are
what can be done without one.

## What would kill this

* **Extraction is not reproducible enough** to compile into stable rules — the
  agent proposes materially different flag sets from the same column on
  different days. Testable today, cheaply, before anything else.
* **The flags duplicate the structured specs** in every dataset that has both,
  so the feature only helps users who have nothing else — a smaller audience
  than assumed.
* **It overfits at 100 rows** even at 5–10 flags, and the honest verdict layer
  correctly downgrades every model that uses it. That would be the verdict
  layer doing its job, and the answer would be no.
* **A regex is enough.** If the semi-structured cases are all a parser handles,
  the agent's judgment is only load-bearing on the messiest inputs, and the
  feature shrinks to a nicety.

## Why it is worth the trouble anyway

The regime where P2Predict loses to deep learning — small data, measured in
Finding 1 of the validation — is the same regime where an agent's judgment
beats brute-force featurization. A hash function cannot pick the five flags
that matter on a 100-part catalogue; a competent reader of the data can. That
is a capability a pure AutoML competitor structurally cannot copy, and it sits
exactly on the project's agentic-first line: the agent does the part that needs
language and judgment, and the engine does the part that needs to be
deterministic, frozen and auditable.
