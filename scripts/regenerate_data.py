#!/usr/bin/env python
"""Regenerate every data product in dataset/ from dataset/tles.txt.

Teams only need this if they want to change the scenario (different TLEs,
horizon, off-nadir envelope, slew model, target list).  The pre-computed data is
already shipped, so this is optional.

    python scripts/regenerate_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eo_tasking.generate import build_instance, write_dataset, GenerateConfig

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dataset = root / "dataset"
    inst = build_instance(dataset / "tles.txt", GenerateConfig())
    write_dataset(inst, dataset)
    print(inst.summary())
    print(f"\nwrote dataset to {dataset}")
