# Case study: Electronic component pricing

> **Status:** Template. Scripts and narrative are scaffolded — results need to be filled in once the case study has been run.

## The procurement question

> Given the technical specs of a passive or active electronic component (manufacturer, package, voltage rating, capacitance / resistance / inductance, tolerance, lead time), what's the expected unit price — and how should an EE procurement engineer act on a supplier quote that lands outside that range?

## Why this case study

This is the textbook parametric procurement use case. Every EE hardware company has a procurement team that benchmarks BOM costs against historical pricing — and most of them do it with spreadsheets and tribal knowledge. The cost-estimating community (ICEAA, SCEA) treats electronic-component pricing as a canonical example in their certification courses.

If P2Predict can demonstrate rigorous parametric pricing on a real catalog of components, EE procurement engineers will recognise the value in ten seconds.

## Data

**Recommended source:** [Octopart API v4](https://octopart.com/api/v4/reference). Free tier covers ~1,000 requests/month — enough for an initial 5–10k-component dataset.

**Alternatives** if Octopart access is a problem:
- [Mouser API](https://api.mouser.com/) — free with registration, slightly different schema.
- [DigiKey API](https://developer.digikey.com/) — free with developer registration, the largest catalog.

**License caveat:** API data is generally not redistributable, so we don't check in the raw data. The `fetch_data.py` script pulls fresh data via the API; users need their own credentials.

**Suggested initial pull** for the case study:
- 8,000–10,000 ceramic capacitors and electrolytic capacitors across three to five manufacturers (Murata, KEMET, Nichicon, Panasonic, AVX).
- Or: 5,000 resistors across three manufacturers (Vishay, Yageo, Susumu).
- Or: connectors, MOSFETs, op-amps — anything where parametric specs drive price.

Pick **one** category for the case study. Don't try to mix capacitors and connectors in one model — they have different cost-driver structures and the case study reads cleaner with one focus.

**Features the model should learn from:**
- Manufacturer (categorical, high cardinality)
- Package / case size (categorical)
- Voltage rating (numerical)
- Capacitance / resistance / inductance value (numerical, log-distributed)
- Tolerance (numerical or categorical: 1%, 5%, 10%, 20%)
- Lead time at quantity break (numerical)
- Temperature coefficient or dielectric type (categorical, when applicable)

**Target:** unit price at a chosen quantity break (e.g. 1,000 units). Document which quantity break in the case study — it matters because prices vary substantially by quantity.

## Reproducing this case study

```bash
# 1. Set your API credentials.
export OCTOPART_API_KEY="..."

# 2. Fetch the dataset (~5 minutes).
python case-studies/electronic-components/fetch_data.py

# 3. Train.
p2predict-train \
  --input case-studies/electronic-components/data/components.csv \
  --target unit_price_1k \
  --budget thorough \
  --outliers warn \
  --feature-outliers warn

# 4. Sample predictions with explanations and likely-range intervals.
python case-studies/electronic-components/predict_examples.py
```

## Results

> _To be filled in once the case study has been run._

- Algorithm selected (auto-mode): ?
- Holdout R²: ?
- Holdout MAE: ?
- Holdout MAPE: ?
- Empirical coverage at 90% target: ?
- Top feature importances: ?

## The procurement insight

> _To be filled in. Aim for a 3–4 sentence story that a procurement reader can quote. Suggested form:_
>
> *"On 8,000 ceramic capacitors from four manufacturers, P2Predict learned that the dominant cost driver isn't manufacturer choice (about Y% impact) but voltage rating and tolerance (about Z% combined). A buyer reviewing a supplier quote of $X for a part that the model expects at $Y can use `--explain` to see exactly which spec is pulling the quote higher — and `--whatif` to see what happens if engineering relaxes the tolerance one tier."*

## Worked examples for the README

> _Once results are in, add 2–3 worked examples here showing the full output of `p2predict -m M -p "..." --explain --interval 90 --whatif "..."`. These are what readers will skim first._
