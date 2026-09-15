#!/bin/bash
# Full sweep: three ADMET endpoints x both HPO budgets, 5 seeds each, under
# TDC's own protocol. Results land in results/<p2predict-version>/.
#
# Run this on each release: the before/after delta against the previous
# version's directory is third-party-scored evidence that a core change
# helped or hurt.
#
# Cost on the dev Mac (v1.1.0 run): ~21 min of model fitting for the three
# `fast` sweeps, ~107 min for the three `thorough` ones, plus featurisation.
#
# In the original v1.1.0 run Caco-2 was launched by hand first and the other
# two endpoints through this script; the commands were identical.
cd "$(dirname "$0")" || exit 1
export PYTHONWARNINGS=ignore
PY=./.venv/bin/python

for bench in Caco2_Wang PPBR_AZ Lipophilicity_AstraZeneca; do
  for budget in fast thorough; do
    echo "########## $bench / $budget ##########"
    $PY run_eval.py --benchmark "$bench" --budget "$budget" \
        --seeds 1 2 3 4 5 --tag all-features
  done
done
echo "########## ALL DONE ##########"
