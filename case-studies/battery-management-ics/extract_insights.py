"""Probe the trained BMIC model for substantive findings.

Hold every feature fixed except one; sweep that one across plausible
values; record the price effect. The output of this script is the
source of the README's "So what?" section.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from p2predict import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def _latest_model() -> Path:
    return sorted(MODELS_DIR.glob("*_unit_price_at_1_usd_*.model"))[-1]


# Baseline: a deliberately bland mid-tier BMIC. Each sweep below
# reads as "what if we change ONLY this feature?".
BASE_BMIC = {
    "manufacturer": "Texas Instruments",
    "Battery Chemistry": "Lithium Ion/Polymer",
    "Interface": "I2C",
    "max_cells_supported": 1.0,
    "op_temp_min_C": -40.0,
    "op_temp_max_C": 85.0,
    "package_pins": 8.0,
    "is_multi_cell": "False",
}


def vary(model, col: str, values: list) -> pd.DataFrame:
    rows = []
    for v in values:
        r = dict(BASE_BMIC)
        r[col] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    preds = model.predict(df)
    return pd.DataFrame({col: values, "predicted": preds})


def main() -> None:
    m = load_model(_latest_model())["model"]
    base = m.predict(pd.DataFrame([BASE_BMIC]))[0]
    print("Baseline BMIC for all sweeps:")
    for k, v in BASE_BMIC.items():
        print(f"  {k:<22} {v}")
    print(f"\nBase prediction: ${base:.2f}\n")

    print("=" * 72)
    print(f"MANUFACTURER PREMIUM (vs base ${base:.2f}, TI single-cell I2C)")
    print("=" * 72)
    mfrs = [
        "Texas Instruments", "Analog Devices Inc./Maxim Integrated",
        "Analog Devices Inc.", "Microchip Technology",
        "STMicroelectronics", "Monolithic Power Systems Inc.",
        "Infineon Technologies", "Nordic Semiconductor ASA",
        "onsemi", "NXP USA Inc.",
    ]
    r = vary(m, "manufacturer", mfrs).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {row['manufacturer']:<38} ${row['predicted']:>5.2f}  "
              f"({delta:+.2f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("CELL-COUNT SCALING (1 → 16 cells supported)")
    print("=" * 72)
    cells = [1, 2, 3, 4, 6, 8, 10, 12, 16]
    r = vary(m, "max_cells_supported", cells)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {int(row['max_cells_supported']):>2} cells   ${row['predicted']:>5.2f}  "
              f"({delta:+.2f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("MULTI-CELL PREMIUM (is_multi_cell flag, holds at 4 cells)")
    print("=" * 72)
    for cells in [1, 4, 16]:
        for mc in ["False", "True"]:
            r = dict(BASE_BMIC, max_cells_supported=float(cells), is_multi_cell=mc)
            pred = m.predict(pd.DataFrame([r]))[0]
            delta = pred - base
            print(f"  {cells:>2}-cell, is_multi_cell={mc:<5}  ${pred:>5.2f}  ({delta:+.2f})")

    print()
    print("=" * 72)
    print("INTERFACE PREMIUM")
    print("=" * 72)
    interfaces = ["unknown", "I2C", "SPI", "SMBus", "USB", "I2C, USB", "HDQ, I2C",
                  "On/Off", "Parallel"]
    r = vary(m, "Interface", interfaces).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {row['Interface']:<18} ${row['predicted']:>5.2f}  "
              f"({delta:+.2f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("BATTERY CHEMISTRY PREMIUM")
    print("=" * 72)
    chems = ["Lithium Ion/Polymer", "Lithium", "Nickel Metal Hydride",
             "Lead Acid", "Multi-Chemistry", "unknown"]
    r = vary(m, "Battery Chemistry", chems).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {row['Battery Chemistry']:<24} ${row['predicted']:>5.2f}  "
              f"({delta:+.2f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("PIN-COUNT SCALING (package complexity proxy)")
    print("=" * 72)
    pins = [6, 8, 10, 14, 16, 20, 24, 32, 48]
    r = vary(m, "package_pins", pins)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {int(row['package_pins']):>2} pins   ${row['predicted']:>5.2f}  "
              f"({delta:+.2f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("OPERATING TEMPERATURE GRADE (industrial vs automotive)")
    print("=" * 72)
    for (lo, hi, label) in [
        (-40.0, 85.0, "industrial (-40/85)"),
        (-40.0, 105.0, "extended industrial (-40/105)"),
        (-40.0, 125.0, "automotive AEC-Q100 (-40/125)"),
        (0.0, 70.0, "commercial (0/70)"),
    ]:
        r = dict(BASE_BMIC, op_temp_min_C=lo, op_temp_max_C=hi)
        pred = m.predict(pd.DataFrame([r]))[0]
        delta = pred - base
        pct = 100 * delta / base
        print(f"  {label:<32} ${pred:>5.2f}  ({delta:+.2f}, {pct:+.0f}%)")


if __name__ == "__main__":
    main()
