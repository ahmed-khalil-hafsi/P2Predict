"""Shim: ``python3 p2predict_train.py ...`` keeps working alongside the
pip-installed ``p2predict-train`` console script.

The real implementation lives in ``p2predict.cli.train``. If you've run
``pip install -e .`` or ``pip install p2predict``, you can use the
``p2predict-train`` command directly instead — they do the same thing.
"""
import os
import sys

# Allow "git clone + run" without a prior install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from p2predict.cli.train import train  # noqa: E402

if __name__ == "__main__":
    train()
