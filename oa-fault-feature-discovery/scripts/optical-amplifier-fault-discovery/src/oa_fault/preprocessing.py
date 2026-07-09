import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def read_table(path, encoding="utf-8-sig"):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path, encoding=encoding)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported input file type: {suffix}")


def clean_sn(series, uppercase=True, strip_spaces=True):
    out = series.astype(str).str.strip()
    if strip_spaces:
        out = out.str.replace(r"\s+", "", regex=True)
    if uppercase:
        out = out.str.upper()
    out = out.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    return out


def make_unique_columns(columns):
    seen = {}
    result = []
    for col in columns:
        base = str(col).strip()
        count = seen.get(base, 0)
        result.append(base if count == 0 else f"{base}__dup{count}")
        seen[base] = count + 1
    return result


def coerce_numeric_features(df, protected_cols):
    feature_cols = []
    for col in df.columns:
        if col in protected_cols:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        converted = converted.replace([np.inf, -np.inf], np.nan)
        if converted.notna().any():
            df[col] = converted
            feature_cols.append(col)
    return df, feature_cols


def normalize_column_name(name):
    return re.sub(r"\s+", "", str(name).strip())


def normalize_column_name_without_units(name):
    normalized = normalize_column_name(name)
    return re.sub(r"[\(\uFF08][^\)\uFF09]*[\)\uFF09]", "", normalized)


def build_column_lookup(columns, normalizer):
    lookup = {}
    for col in columns:
        key = normalizer(col)
        lookup.setdefault(key, []).append(col)
    return lookup


def choose_column_candidate(configured_col, candidates):
    if not candidates:
        return None
    configured_norm = normalize_column_name(configured_col)
    configured_nounit = normalize_column_name_without_units(configured_col)

    def score(col):
        col_norm = normalize_column_name(col)
        col_nounit = normalize_column_name_without_units(col)
        if col == configured_col:
            return (0, len(col))
        if col_norm == configured_norm:
            return (1, len(col))
        if col_nounit == configured_nounit and col_norm == configured_nounit:
            return (2, len(col))
        if col_nounit == configured_nounit:
            return (3, len(col))
        return (9, len(col))

    return sorted(candidates, key=score)[0]


def select_feature_columns(df, protected_cols, requested_feature_cols=None, exclude_feature_cols=None):
    exclude_feature_cols = set(exclude_feature_cols or [])
    if requested_feature_cols:
        normalized_to_actual = build_column_lookup(df.columns, normalize_column_name)
        nounit_to_actual = build_column_lookup(df.columns, normalize_column_name_without_units)

        feature_cols = []
        resolved_pairs = []
        missing = []
        for configured_col in requested_feature_cols:
            if configured_col in df.columns:
                actual_col = configured_col
            else:
                normalized = normalize_column_name(configured_col)
                normalized_no_unit = normalize_column_name_without_units(configured_col)
                actual_col = choose_column_candidate(configured_col, normalized_to_actual.get(normalized, []))
                if actual_col is None:
                    actual_col = choose_column_candidate(configured_col, nounit_to_actual.get(normalized_no_unit, []))
            if actual_col is None:
                missing.append(configured_col)
                continue
            if actual_col in protected_cols:
                raise ValueError(f"Configured feature column is protected and cannot be used: {actual_col}")
            if configured_col not in feature_cols:
                values = pd.to_numeric(df[actual_col], errors="coerce")
                values = values.replace([np.inf, -np.inf], np.nan)
                df[configured_col] = values
                feature_cols.append(configured_col)
                if actual_col != configured_col:
                    resolved_pairs.append((configured_col, actual_col))
        if missing:
            raise ValueError(f"Configured feature_cols not found in input data: {missing}")
        if resolved_pairs:
            print("[prepare] Feature column aliases resolved:")
            for configured_col, actual_col in resolved_pairs[:20]:
                print(f"  - {configured_col} -> {actual_col}")
            if len(resolved_pairs) > 20:
                print(f"  ... {len(resolved_pairs) - 20} more")
        return df, feature_cols

    df, feature_cols = coerce_numeric_features(df, protected_cols)
    feature_cols = [col for col in feature_cols if col not in exclude_feature_cols]
    return df, feature_cols


def derive_min_max_features(df, feature_cols):
    by_key = {}
    for col in feature_cols:
        compact = re.sub(r"\s+", "", col)
        max_match = re.match(r"(.+?)MAX(.*)", compact, flags=re.IGNORECASE)
        min_match = re.match(r"(.+?)MIN(.*)", compact, flags=re.IGNORECASE)
        if max_match:
            key = (max_match.group(1), max_match.group(2))
            by_key.setdefault(key, {})["max"] = col
        if min_match:
            key = (min_match.group(1), min_match.group(2))
            by_key.setdefault(key, {})["min"] = col

    new_cols = []
    for (name, unit), pair in by_key.items():
        if "max" not in pair or "min" not in pair:
            continue
        max_col, min_col = pair["max"], pair["min"]
        safe_name = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", name).strip("_")
        mean_col = f"{safe_name}_MEAN"
        range_col = f"{safe_name}_RANGE"
        if mean_col not in df.columns:
            df[mean_col] = (df[max_col] + df[min_col]) / 2.0
            new_cols.append(mean_col)
        if range_col not in df.columns:
            df[range_col] = df[max_col] - df[min_col]
            new_cols.append(range_col)
    return df, feature_cols + new_cols


def deduplicate(df, group_col, timestamp_col, feature_cols, label_col, policy="last"):
    sort_cols = [group_col, timestamp_col]
    df = df.sort_values(sort_cols).copy()
    if policy == "last":
        return df.drop_duplicates(sort_cols, keep="last")
    if policy == "mean":
        agg = {col: "mean" for col in feature_cols}
        agg[label_col] = "max"
        return df.groupby(sort_cols, as_index=False).agg(agg)
    raise ValueError("duplicate_policy must be 'last' or 'mean'")


def impute_features(df, group_col, feature_cols, add_indicators=True):
    indicator_cols = []
    if add_indicators:
        for col in feature_cols:
            ind_col = f"{col}__missing"
            df[ind_col] = df[col].isna().astype(float)
            indicator_cols.append(ind_col)

    df[feature_cols] = df.groupby(group_col, group_keys=False)[feature_cols].apply(lambda x: x.ffill().bfill())
    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians).fillna(0.0)
    return df, feature_cols + indicator_cols, medians.to_dict()


def clip_and_standardize(df, feature_cols, clip_quantiles=None, standardize=True):
    stats = {"clip": {}, "mean": {}, "std": {}}
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True)).fillna(0.0)
    if clip_quantiles:
        lo_q, hi_q = clip_quantiles
        for col in feature_cols:
            lo = float(df[col].quantile(lo_q))
            hi = float(df[col].quantile(hi_q))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                df[col] = df[col].clip(lo, hi)
                stats["clip"][col] = [lo, hi]
    if standardize:
        for col in feature_cols:
            mean = float(df[col].mean())
            std = float(df[col].std(ddof=0))
            if not np.isfinite(std) or std == 0:
                std = 1.0
            if not np.isfinite(mean):
                mean = 0.0
            df[col] = (df[col] - mean) / std
            stats["mean"][col] = mean
            stats["std"][col] = std
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, stats


def split_groups(groups, labels=None, val_size=0.2, test_size=0.0, random_seed=42):
    group_series = pd.Series(groups)
    unique = group_series.drop_duplicates().to_numpy()
    if len(unique) < 2:
        return set(unique), set(), set()

    group_labels = None
    if labels is not None:
        label_df = pd.DataFrame({"group": groups, "label": np.asarray(labels).reshape(-1)})
        group_labels = label_df.groupby("group")["label"].max().reindex(unique).fillna(0).astype(int).to_numpy()

    def holdout_split(items, holdout_size, seed, stratify_labels=None):
        if isinstance(holdout_size, float):
            n_holdout = int(np.ceil(len(items) * holdout_size))
        else:
            n_holdout = int(holdout_size)
        can_stratify = (
            stratify_labels is not None
            and len(np.unique(stratify_labels)) > 1
            and min(np.bincount(stratify_labels)) >= 2
            and n_holdout >= len(np.unique(stratify_labels))
            and (len(items) - n_holdout) >= len(np.unique(stratify_labels))
        )
        if can_stratify:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
            return next(splitter.split(items, stratify_labels))
        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
        return next(splitter.split(items, groups=items))

    train_groups = unique
    val_groups = np.array([])
    test_groups = np.array([])
    train_labels = group_labels
    if test_size and test_size > 0:
        train_idx, test_idx = holdout_split(unique, test_size, random_seed, group_labels)
        train_groups, test_groups = unique[train_idx], unique[test_idx]
        train_labels = group_labels[train_idx] if group_labels is not None else None
    if val_size and val_size > 0 and len(train_groups) > 1:
        train_idx, val_idx = holdout_split(train_groups, val_size, random_seed, train_labels)
        val_groups = train_groups[val_idx]
        train_groups = train_groups[train_idx]
    return set(train_groups), set(val_groups), set(test_groups)


def window_label(values, strategy):
    if strategy == "last":
        return float(values[-1])
    if strategy == "any":
        return float(np.nanmax(values) > 0)
    if strategy == "max":
        return float(np.nanmax(values))
    raise ValueError("label_strategy must be last, any, or max")


def estimate_window_bytes(n_windows, n_features, window_len):
    return int(n_windows) * int(n_features) * int(window_len) * np.dtype(np.float32).itemsize


def build_windows(
    df,
    group_col,
    timestamp_col,
    label_col,
    feature_cols,
    window_len,
    stride,
    min_history,
    label_strategy,
    max_windows_total=None,
    window_sample_mode="uniform",
    window_sample_seed=42,
):
    candidates = []
    for sn, part in df.groupby(group_col, sort=False):
        part = part.sort_values(timestamp_col).reset_index(drop=True)
        if len(part) < min_history:
            continue
        for end in range(window_len, len(part) + 1, stride):
            start = end - window_len
            candidates.append((sn, start, end))

    if not candidates:
        raise ValueError("No windows were created. Check window_len/min_history and per-sn history length.")

    estimated_windows = len(candidates)
    if max_windows_total is not None:
        max_windows_total = int(max_windows_total)
        if estimated_windows > max_windows_total:
            if window_sample_mode == "uniform":
                keep_idx = np.linspace(0, estimated_windows - 1, max_windows_total, dtype=int)
            elif window_sample_mode == "random":
                rng = np.random.default_rng(int(window_sample_seed))
                keep_idx = np.sort(rng.choice(estimated_windows, size=max_windows_total, replace=False))
            else:
                raise ValueError("window_sample_mode must be 'uniform' or 'random'")
            candidates = [candidates[i] for i in keep_idx]

    n_windows = len(candidates)
    n_features = len(feature_cols)
    estimated_bytes = estimate_window_bytes(n_windows, n_features, window_len)
    print(
        "Building windows: "
        f"estimated={estimated_windows:,}, kept={n_windows:,}, "
        f"shape=({n_windows:,}, {n_features}, {window_len}), "
        f"memory={estimated_bytes / (1024 ** 3):.2f} GiB"
    )

    grouped = {
        sn: part.sort_values(timestamp_col).reset_index(drop=True)
        for sn, part in df.groupby(group_col, sort=False)
    }
    X = np.empty((n_windows, n_features, window_len), dtype=np.float32)
    y = np.empty((n_windows, 1), dtype=np.float32)
    rows = []
    for i, (sn, start, end) in enumerate(candidates):
        part = grouped[sn]
        values = part[feature_cols].iloc[start:end].to_numpy(dtype=np.float32)
        labels = part[label_col].iloc[start:end].to_numpy(dtype=np.float32)
        times = part[timestamp_col].astype(str).to_numpy()
        X[i] = values.T
        y[i, 0] = window_label(labels, label_strategy)
        rows.append({
            "sn": sn,
            "start_time": times[start],
            "end_time": times[end - 1],
            "source_index_start": int(start),
            "source_index_end": int(end - 1),
        })

    return X, y, pd.DataFrame(rows)


def prepare_data(input_path, output_dir, config):
    data_cfg = config["data"]
    prep_cfg = config.get("preprocessing", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Reading input table: {input_path}")
    df = read_table(input_path, encoding=data_cfg.get("input_encoding", "utf-8-sig"))
    print(f"[prepare] Loaded rows={len(df):,}, columns={len(df.columns):,}")
    df.columns = make_unique_columns(df.columns)

    timestamp_col = data_cfg["timestamp_col"]
    group_col = data_cfg["group_col"]
    label_col = data_cfg["label_col"]
    required = [timestamp_col, group_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[group_col] = clean_sn(
        df[group_col],
        uppercase=prep_cfg.get("sn_uppercase", True),
        strip_spaces=prep_cfg.get("sn_strip_spaces", True),
    )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    before_drop = len(df)
    df = df.dropna(subset=[group_col, timestamp_col, label_col]).copy()
    print(f"[prepare] Dropped {before_drop - len(df):,} rows with missing sn/time/label.")
    df[label_col] = (df[label_col] > 0).astype(float)

    ignore_cols = set(data_cfg.get("ignore_cols", [])) | {timestamp_col, group_col, label_col}
    print("[prepare] Selecting feature columns...")
    df, feature_cols = select_feature_columns(
        df,
        protected_cols=ignore_cols,
        requested_feature_cols=data_cfg.get("feature_cols"),
        exclude_feature_cols=data_cfg.get("exclude_feature_cols"),
    )
    if not feature_cols:
        raise ValueError("No numeric feature columns were selected.")
    print(f"[prepare] Selected {len(feature_cols)} feature columns.")
    if data_cfg.get("derive_min_max_features", True):
        df, feature_cols = derive_min_max_features(df, feature_cols)

    before_dedup = len(df)
    print("[prepare] Sorting and deduplicating rows...")
    df = deduplicate(
        df,
        group_col=group_col,
        timestamp_col=timestamp_col,
        feature_cols=feature_cols,
        label_col=label_col,
        policy=data_cfg.get("duplicate_policy", "last"),
    )
    print(f"[prepare] Removed {before_dedup - len(df):,} duplicate sn/time rows.")
    print("[prepare] Imputing missing feature values...")
    df, feature_cols, medians = impute_features(
        df,
        group_col=group_col,
        feature_cols=feature_cols,
        add_indicators=data_cfg.get("add_missing_indicators", True),
    )
    print("[prepare] Clipping and standardizing features...")
    df, scale_stats = clip_and_standardize(
        df,
        feature_cols,
        clip_quantiles=prep_cfg.get("clip_quantiles"),
        standardize=prep_cfg.get("standardize", True),
    )

    print("[prepare] Building windows...")
    X, y, meta = build_windows(
        df,
        group_col=group_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
        feature_cols=feature_cols,
        window_len=int(data_cfg.get("window_len", 64)),
        stride=int(data_cfg.get("stride", 1)),
        min_history=int(data_cfg.get("min_history", data_cfg.get("window_len", 64))),
        label_strategy=data_cfg.get("label_strategy", "last"),
        max_windows_total=data_cfg.get("max_windows_total"),
        window_sample_mode=data_cfg.get("window_sample_mode", "uniform"),
        window_sample_seed=data_cfg.get("window_sample_seed", data_cfg.get("random_seed", 42)),
    )
    if not np.isfinite(X).all():
        raise ValueError("Processed X contains NaN or Inf values after preprocessing.")
    if not np.isfinite(y).all():
        raise ValueError("Processed y contains NaN or Inf values after preprocessing.")

    print("[prepare] Splitting windows by sn...")
    train_groups, val_groups, test_groups = split_groups(
        meta["sn"],
        labels=y.reshape(-1),
        val_size=float(data_cfg.get("val_size", 0.2)),
        test_size=float(data_cfg.get("test_size", 0.0)),
        random_seed=int(data_cfg.get("random_seed", 42)),
    )
    split = np.array([
        "train" if sn in train_groups else "val" if sn in val_groups else "test"
        for sn in meta["sn"]
    ])
    meta["split"] = split
    print(
        "[prepare] Split counts: "
        + ", ".join(f"{k}={(split == k).sum():,}" for k in ["train", "val", "test"])
    )

    print(f"[prepare] Saving processed arrays to: {output_dir}")
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    meta.to_csv(output_dir / "window_metadata.csv", index=False, encoding="utf-8-sig")
    cleaned_path = output_dir / "cleaned_rows.parquet"
    try:
        df.to_parquet(cleaned_path, index=False)
    except ImportError:
        cleaned_path = output_dir / "cleaned_rows.csv"
        df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_raw_rows": int(len(df)),
        "n_windows": int(len(X)),
        "n_features": int(len(feature_cols)),
        "window_shape": list(X.shape[1:]),
        "window_array_gib": estimate_window_bytes(len(X), len(feature_cols), int(data_cfg.get("window_len", 64))) / (1024 ** 3),
        "label_positive_rate": float(y.mean()),
        "splits": {k: int((split == k).sum()) for k in ["train", "val", "test"]},
        "feature_cols": feature_cols,
        "impute_medians": medians,
        "scale_stats": scale_stats,
        "cleaned_rows_path": str(cleaned_path),
        "config": config,
    }
    with open(output_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
