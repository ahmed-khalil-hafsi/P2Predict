"""Allow ``python -m p2predict`` to invoke the predict CLI without needing
the console script entry point. Equivalent to ``p2predict`` after
``pip install``.

For training, use ``python -m p2predict.cli.train`` or the
``p2predict-train`` console script.
"""
from p2predict.cli.predict import main

if __name__ == "__main__":
    main()
