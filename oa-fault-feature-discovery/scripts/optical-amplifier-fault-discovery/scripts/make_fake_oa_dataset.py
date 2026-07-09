import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_feature_cols(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def baseline_value(feature, rng):
    if "温" in feature:
        return rng.normal(465.0, 8.0)
    if "EDFA增益" in feature or "可见增益" in feature:
        return rng.normal(1700.0, 15.0)
    if "3V3" in feature:
        return rng.normal(326.0, 1.5)
    if "PAIN" in feature:
        return rng.normal(250.0, 8.0)
    if "BAOUT" in feature:
        return rng.normal(1950.0, 12.0)
    if "BAIN" in feature:
        return rng.normal(375.0, 6.0)
    if "PAOUT" in feature:
        return rng.normal(1320.0, 10.0)
    if "驱动电流" in feature:
        return rng.normal(3650.0, 25.0)
    if "制冷电流" in feature:
        return rng.normal(4800.0, 80.0)
    if "管芯温度" in feature:
        return rng.normal(250.0, 1.0)
    if "背光电流" in feature:
        return rng.normal(660.0, 8.0)
    if "TEC电压" in feature:
        return rng.normal(70.0, 3.0)
    if "累计上电运行时间" in feature:
        return rng.normal(2_500_000.0, 20_000.0)
    if "上电后模块" in feature or "上电运行时间" in feature:
        return rng.normal(150_000.0, 5_000.0)
    if "累计上电次数" in feature:
        return rng.normal(15.0, 2.0)
    if "超温" in feature:
        return rng.normal(0.0, 0.1)
    return rng.normal(100.0, 5.0)


def add_fault_effect(feature, value, severity, t_in_fault, rng):
    drift = severity * (0.4 + 0.6 * t_in_fault)
    noise = rng.normal(0.0, 0.5)
    compact = feature.replace(" ", "")
    if "温" in compact:
        return value + 18.0 * drift + noise
    if "LSR1驱动电流" in compact:
        return value + 140.0 * drift + noise
    if "LSR1制冷电流" in compact:
        return value + 260.0 * drift + noise
    if "LSR1背光电流" in compact:
        return value - 25.0 * drift + noise
    if "LSR1TEC电压" in compact:
        return value + 8.0 * drift + noise
    if "PAOUT" in compact:
        return value - 45.0 * drift + noise
    if "BAOUT" in compact:
        return value - 35.0 * drift + noise
    if "EDFA增益" in compact or "可见增益" in compact:
        return value + 20.0 * drift + noise
    if "3V3" in compact:
        return value + rng.normal(0.0, 0.5)
    return value + noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/fake_oa_fault.parquet")
    parser.add_argument("--feature-cols", default="configs/feature_cols.txt")
    parser.add_argument("--n-sn", type=int, default=18)
    parser.add_argument("--points-per-sn", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    feature_cols = read_feature_cols(args.feature_cols)
    rows = []
    start = pd.Timestamp("2025-05-01 00:00:00")

    for sn_idx in range(args.n_sn):
        base_sn = f"032RUH10JC{sn_idx:06d}"
        messy_sn = base_sn if sn_idx % 3 else f" {base_sn[:5]} {base_sn[5:].lower()} "
        is_faulty_device = sn_idx >= int(args.n_sn * 0.55)
        severity = rng.uniform(0.7, 1.3)
        base_values = {feature: baseline_value(feature, rng) for feature in feature_cols}
        cumulative_offset = rng.integers(0, 30_000)

        for t in range(args.points_per_sn):
            timestamp = start + pd.Timedelta(minutes=15 * t + sn_idx)
            in_fault_zone = is_faulty_device and t >= int(args.points_per_sn * 0.55)
            t_in_fault = max(0.0, (t - args.points_per_sn * 0.55) / max(args.points_per_sn * 0.45, 1))
            label = int(in_fault_zone)
            row = {
                "DateTime": timestamp,
                "sn": messy_sn,
                "source_file": f"fake_port{sn_idx % 4}_FILE{sn_idx:02d}.csv",
                "alarm_time": timestamp + pd.Timedelta(days=2) if label else pd.NaT,
                "label": label,
            }
            for feature in feature_cols:
                value = base_values[feature] + rng.normal(0.0, 2.0)
                if "累计上电运行时间" in feature:
                    value = 2_000_000 + cumulative_offset + t * 15
                elif "上电后模块" in feature or "上电运行时间" in feature:
                    value = 50_000 + t * 15
                elif "累计上电次数" in feature:
                    value = 10 + sn_idx % 8
                elif "超温" in feature:
                    value = max(0.0, (t - args.points_per_sn * 0.75) * 2.0) if in_fault_zone else 0.0
                elif in_fault_zone:
                    value = add_fault_effect(feature, value, severity, t_in_fault, rng)
                row[feature] = float(value)
            rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in feature_cols if c in df.columns]
    mask = rng.random((len(df), len(numeric_cols))) < 0.01
    for j, col in enumerate(numeric_cols):
        df.loc[mask[:, j], col] = np.nan

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Wrote {len(df)} rows, {len(feature_cols)} features to {output}")
    print(df[["DateTime", "sn", "label"]].head().to_string(index=False))
    print("label counts:", df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
