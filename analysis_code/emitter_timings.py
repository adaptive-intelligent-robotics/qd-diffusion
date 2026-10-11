import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config


def get_experiment_subdir(base_path: Path) -> Optional[Path]:
    if not base_path.exists() or not base_path.is_dir():
        return None

    subdirs = [p for p in base_path.iterdir() if p.is_dir()]
    if not subdirs:
        return None

    return sorted(subdirs)[-1]


def collect_timings(root_dir: Path, algorithm_dicts: Dict[str, Dict[str, str]], seeds: List[int]) -> pd.DataFrame:
    records = []
    missing = []

    for algo_key, config in algorithm_dicts.items():
        for seed in seeds:
            base_path = root_dir / str(seed) / config["path"]
            exp_subdir = get_experiment_subdir(base_path)
            if exp_subdir is None:
                missing.append((algo_key, seed, str(base_path)))
                continue

            csv_path = exp_subdir / "metrics" / "mapelites-logs.csv"
            if not csv_path.exists():
                missing.append((algo_key, seed, str(csv_path)))
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                missing.append((algo_key, seed, f"read error: {exc}"))
                continue

            if df.empty or "emitter_time" not in df.columns:
                missing.append((algo_key, seed, "missing emitter_time"))
                continue

            records.append({
                "Algorithm": config["label"],
                "Algorithm Key": algo_key,
                "Seed": seed,
                "Final Emitter Time": float(df["emitter_time"].iloc[-1]),
                "CSV Path": str(csv_path),
            })

    if missing:
        print(f"Skipped {len(missing)} missing or invalid run(s).")
        for algo_key, seed, reason in missing:
            print(f"  - {algo_key} seed {seed}: {reason}")

    return pd.DataFrame(records)


def save_timings_table(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No timing data collected, nothing to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved timing table to: {output_path}")

    summary = (
        df.groupby("Algorithm")["Final Emitter Time"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary_path = output_path.parent / "emitter_timing_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved timing summary to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect final emitter wall-clock times per algorithm and seed.")
    config.add_model_argument(parser)
    args = parser.parse_args()

    cfg = config.get_model_config(args.model)
    timing_df = collect_timings(Path(cfg["root_dir"]), cfg["algorithms"], cfg["seeds"])
    save_timings_table(timing_df, Path(cfg["output_dir"]) / "emitter_timings.csv")
