<div align="center">

# P2Predict

### Know what every part should cost — before you sit down to negotiate.

**P2Predict turns your purchasing history into a price model your team can talk to. Ask what a part should cost, why, and how sure the answer is — in plain language, through the AI agent you already use.**

![Works with any AI agent](https://img.shields.io/badge/works%20with-any%20AI%20agent-6E56CF) &nbsp;
![Local first](https://img.shields.io/badge/local--first-your%20data%20never%20leaves-1F8A4C) &nbsp;
![Built for procurement](https://img.shields.io/badge/built%20for-procurement-0B6BCB) &nbsp;
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-555)

![P2Predict MCP Demo](./documentation/p2predict_mcp_demo.gif)

</div>

## The problem

Your team negotiates thousands of prices a year against gut feel and last year's PO. The senior buyer who *knew* what a part should cost is retiring. Engineering adds a tolerance and nobody can say what it costs. A supplier quote lands and the buyer has no defensible number to push back with.

The pricing knowledge is in your purchase history. It just isn't usable.

## What P2Predict does

P2Predict learns from your own purchasing data what drives the price of a part — supplier, material, size, spec, region — and gives your team a defensible target for any new or proposed part. Your buyers ask in plain English. The answer comes back grounded in what you have actually paid.

It runs on your machine, on your data, through any AI agent your team already uses. Nothing is uploaded. No vendor catalog, no cloud, no per-seat data-sharing.

## Where the real value is: it tells you how much to trust the number

Most tools hand you a number and walk away. P2Predict hands you the number **and tells you how confident to be in it** — per part, in dollars.

- **A confidence range on every estimate.** Not "$12.40" — "$12.40, and 9 times in 10 the real price lands between $10.80 and $13.90." A tight range means negotiate hard. A wide one means get a quote first. Your buyer walks in knowing which.
- **An honest map of where the model is strong and where it's thin.** P2Predict flags which parts of your category it can benchmark with confidence and which need a real quote — so nobody negotiates off a number the data can't support.
- **A reason for every number.** Every estimate breaks down into what each spec and the supplier contribute — "supplier +$0.85, rush delivery +$1.20, tighter tolerance +$0.42." You argue the components, not the total.

That honesty is the point. A confident-but-wrong benchmark loses you credibility in a sourcing conversation. P2Predict is built to never do that.

## What your team can do with it

**Sanity-check a quote in seconds.** Supplier quotes $14.20, the model says $12.40 with a $10.80–$13.90 range. Now the buyer has a number — and the breakdown to argue it line by line.

**Quantify a supplier premium you can negotiate away.** Hold the spec fixed, swap the supplier, read the delta. "Moving this 16-cell pack monitor from supplier A to supplier B is −37.7%, −$2.07 a unit." A week of RFQs, answered in 30 seconds — a lever you take into the room.

**Triage an RFQ fast.** Drop 200 line items on the agent. Every line gets a target and a range. The 8–15 that fall outside the range are the ones worth a call. The rest are routine. Your team spends the afternoon on what moves the number.

**Cost design changes before they're locked in.** In a design review or spec workshop, engineering proposes a tighter tolerance. Ask the agent: "+$0.42 a unit, +18%." Now the conversation is "is 18% worth this requirement?" — a costed trade-off, not the loudest voice in the room.

**Win the spec-analysis workshop.** Walk into the cost-down workshop with every spec priced: which features carry real cost, which premiums are negotiable, where the design is paying for something it doesn't need. Backed by your own data, with the confidence level on each finding.

**Defend a decision months later.** Finance asks about a PO from six months ago. The model's estimate, range, and the drivers behind it are on file — an auditable trail, not a memory.

**Keep the retiring buyer's intuition.** Twenty years of "that feels high" walks out the door when a senior buyer leaves. A model trained on their buys captures what drives cost and what normal looks like. The next buyer inherits a baseline, not just a spreadsheet.

## How it fits your stack

P2Predict speaks to **any AI agent** — Claude, and any other assistant your team runs — through a standard connector. Your buyers don't learn a new tool. They ask the assistant they already use, and it does the analysis.

Everything runs **locally**. Your purchasing data is your most sensitive commercial asset; it stays on your machine. No upload, no third-party model training on your spend, no data-residency conversation with legal.

It complements should-cost tools. Bottom-up should-costing models build a part up from material, labor, and machine time. P2Predict answers the other question every buyer actually asks: *what has the market charged us for parts like this, and what should the next one cost?*

## Proof on public data

Three worked case studies, each reproducible end-to-end on data anyone can download:

- **[Battery Management ICs](case-studies/battery-management-ics/)** — the closest to a real procurement job: a small, realistic parts slice, a supplier-premium lever you can quote, and an honest read on where the model is solid and where it isn't.
- **[Used vehicles](case-studies/used-cars/)** — the easy-to-follow walkthrough on prices that span orders of magnitude.
- **[Aerospace fasteners](case-studies/aerospace-fasteners/)** — the honesty story: how P2Predict tells you when the data, not the model, is the limit, so you stop chasing accuracy that isn't there.

Each shows the same thing a buyer cares about: lead with results, show where to trust them, point to exactly where each number comes from.

## Who built it

P2Predict is built and maintained by **[Ahmed K. Hafsi](https://ahmedhafsi.com)** — Senior Manager, Negotiation Excellence at Infineon, where he leads negotiation strategy across automotive, semiconductor, consumer, and chemicals categories in Asia-Pacific. He built and led Dyson's global Negotiation Excellence capability and advised on negotiation and applied game theory at TWS Partners in Munich and London. He trained as an engineer at the Karlsruhe Institute of Technology and works across three continents in four languages.

P2Predict comes out of that work: the tools a procurement team actually needs to walk into a negotiation knowing its number and its leverage. More on his approach to negotiation, pricing, and game theory at **[ahmedhafsi.com](https://ahmedhafsi.com)**.

## Try it / set it up

- **Set it up with your agent** — see **[INSTALL.md](INSTALL.md)** (install, connect your AI assistant, point it at your data).
- **How it works under the hood** — the models, the math, the full reference: **[TECHNICAL.md](TECHNICAL.md)**.

## Licensing

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE).

- **Free for internal use** — use P2Predict inside your own organization at no cost.
- **Commercial use requires a license** — deploying for clients, embedding it in a paid service, or consulting engagements.

For a commercial license, partnership, or to share a procurement dataset, reach out: **[ahmedhafsi.com/contact](https://ahmedhafsi.com/contact/)**.

© Ahmed K. Hafsi. P2Predict is a copyrighted work; all rights reserved except as granted under the license above.
