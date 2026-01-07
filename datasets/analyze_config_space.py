import os
import glob
from typing import List, Optional

import pandas as pd


# ================== User settings ==================
DATASETS_DIR = "../datasets"

# All systems (edit freely)
SYSTEMS: List[str] = [
    "batik", "dconvert", "h2", "jump3r",
    "kanzi", "lrzip", "x264", "xz", "z3"
]

# e.g. ["h2", "z3"], otherwise keep None
ONLY_SYSTEMS: Optional[List[str]] = None
# ===================================================


def sci_notation(x: int, sig: int = 3) -> str:
    """Scientific notation formatting."""
    if x == 0:
        return "0"
    return f"{x:.{sig}e}"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pandas index columns if present."""
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def compute_config_space(df: pd.DataFrame) -> int:
    """
    First n-1 columns are configurations,
    last column is performance.
    """
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns")

    config_df = df.iloc[:, :-1]
    space_size = 1

    for col in config_df.columns:
        uniq = config_df[col].dropna().nunique()
        space_size *= uniq

    return space_size, config_df.shape[1]


def main():
    systems = ONLY_SYSTEMS if ONLY_SYSTEMS else SYSTEMS

    for system in systems:
        system_dir = os.path.join(DATASETS_DIR, system)

        print("=" * 70)
        print(f"System: {system}")
        print(f"Path  : {system_dir}")

        if not os.path.isdir(system_dir):
            print("  [WARN] System directory not found.")
            continue

        csv_files = sorted(glob.glob(os.path.join(system_dir, "*.csv")))
        if not csv_files:
            print("  [WARN] No CSV files found.")
            continue

        print("-" * 70)
        print(f"{'Workload CSV':<45} {'#Dims':>6} {'Config Space Size':>18}")
        print("-" * 70)

        for csv_file in csv_files:
            name = os.path.basename(csv_file)
            try:
                df = pd.read_csv(csv_file)
                df = clean_columns(df)

                space_size, dims = compute_config_space(df)
                space_str = sci_notation(space_size)

                print(f"{name:<45} {dims:>6} {space_str:>18}")

            except Exception as e:
                print(f"{name:<45} {'-':>6} {'ERROR':>18}  ({e})")

        print()

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
