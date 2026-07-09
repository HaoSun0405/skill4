import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def compact_name(name):
    return re.sub(r"\s+", "", str(name))


def display_feature_name(name):
    compact = compact_name(name)
    return re.sub(r"[\(\uFF08][^\)\uFF09]*[\)\uFF09]", "", compact)


def strip_stat(name):
    compact = compact_name(name)
    return re.sub(r"(MAX|MIN)(?=[\(\uFF08]|$)", "", compact, flags=re.IGNORECASE)


def stat_of(name):
    compact = compact_name(name).upper()
    if "MAX" in compact:
        return "MAX"
    if "MIN" in compact:
        return "MIN"
    return "VALUE"


def unit_scale(name):
    compact = compact_name(name)
    if "0.001mA" in compact:
        return 1e-6, "A"
    if "0.1mA" in compact:
        return 1e-4, "A"
    if "0.01mA" in compact:
        return 1e-5, "A"
    if "mA" in compact:
        return 1e-3, "A"
    if "0.01V" in compact:
        return 1e-2, "V"
    if "0.1V" in compact:
        return 1e-1, "V"
    if re.search(r"[\(\uFF08]V[\)\uFF09]", compact):
        return 1.0, "V"
    if "0.01dB" in compact:
        return 1e-2, "dB"
    if "0.1dB" in compact:
        return 1e-1, "dB"
    if "dB" in compact:
        return 1.0, "dB"
    if "0.1" in compact and ("℃" in compact or "鈩" in compact):
        return 1e-1, "C"
    if "分钟" in compact or "鍒嗛挓" in compact:
        return 1.0, "min"
    return 1.0, "raw"


def expected_units(quantity):
    if quantity in {"tec_voltage", "supply_voltage"}:
        return {"V"}
    if quantity in {"chip_temperature", "case_temperature", "temperature"}:
        return {"C"}
    if quantity in {"backlight_current", "drive_current", "cooling_current"}:
        return {"A"}
    if quantity in {"pain_power", "paout_power", "bain_power", "baout_power", "edfa_gain", "visible_gain"}:
        return {"dB"}
    return None


def validate_unit(name, quantity, unit):
    expected = expected_units(quantity)
    if expected is None or unit == "raw":
        return
    if unit not in expected:
        raise ValueError(
            "Feature unit does not match semantic quantity: "
            f"{name!r} was parsed as quantity={quantity!r}, unit={unit!r}, "
            f"expected one of {sorted(expected)}. Fix configs/feature_cols.txt and rerun preprocessing."
        )


def quantity_of(name):
    compact = compact_name(name)
    if "TEC" in compact and ("电压" in compact or "鐢靛帇" in compact):
        return "tec_voltage"
    if "3V3" in compact or "电源" in compact or "鐢垫簮" in compact:
        return "supply_voltage"
    if "驱动电流" in compact or "椹卞姩鐢垫祦" in compact:
        return "drive_current"
    if "制冷电流" in compact or "鍒跺喎鐢垫祦" in compact:
        return "cooling_current"
    if "背光电流" in compact or "鑳屽厜鐢垫祦" in compact:
        return "backlight_current"
    if "管芯温度" in compact or "绠¤姱娓╁害" in compact:
        return "chip_temperature"
    if "壳温" in compact or "澹虫俯" in compact:
        return "case_temperature"
    if "温度" in compact or "娓╁害" in compact:
        return "temperature"
    if "PAIN" in compact:
        return "pain_power"
    if "PAOUT" in compact:
        return "paout_power"
    if "BAIN" in compact:
        return "bain_power"
    if "BAOUT" in compact:
        return "baout_power"
    if "EDFA" in compact and ("增益" in compact or "澧炵泭" in compact):
        return "edfa_gain"
    if "可见" in compact or "鍙" in compact:
        return "visible_gain"
    return "unknown"


def component_of(name):
    compact = compact_name(name).upper()
    if "LSR1" in compact:
        return "LSR1"
    if "LSR2" in compact:
        return "LSR2"
    if "PAIN" in compact or "PAOUT" in compact:
        return "PA"
    if "BAIN" in compact or "BAOUT" in compact:
        return "BA"
    return None


def feature_semantics(feature_names):
    rows = []
    for idx, name in enumerate(feature_names):
        quantity = quantity_of(name)
        scale, unit = unit_scale(name)
        validate_unit(name, quantity, unit)
        rows.append({
            "idx": idx,
            "feature": str(name),
            "compact": compact_name(name),
            "base": strip_stat(name),
            "stat": stat_of(name),
            "quantity": quantity,
            "component": component_of(name),
            "scale": scale,
            "unit": unit,
        })
    return rows


def restore_raw_windows(windows, feature_names, preprocess_summary):
    windows = np.asarray(windows, dtype=np.float64)
    stats = (preprocess_summary or {}).get("scale_stats", {})
    means = stats.get("mean", {})
    stds = stats.get("std", {})
    restored = windows.copy()
    for idx, name in enumerate(feature_names):
        mean = float(means.get(name, 0.0))
        std = float(stds.get(name, 1.0))
        if not math.isfinite(std) or std == 0.0:
            std = 1.0
        if not math.isfinite(mean):
            mean = 0.0
        restored[:, idx, :] = restored[:, idx, :] * std + mean
    return restored


def physical_feature_series(window, sem):
    return np.asarray(window[sem["idx"]], dtype=np.float64) * float(sem["scale"])


def db_to_linear(db_values):
    return np.power(10.0, np.asarray(db_values, dtype=np.float64) / 10.0)


def safe_ratio(num, den):
    den = np.asarray(den, dtype=np.float64)
    return np.asarray(num, dtype=np.float64) / np.where(np.abs(den) < EPS, np.nan, den)


def summarize_series_delta(start_series, end_series):
    start_series = np.asarray(start_series, dtype=np.float64)
    end_series = np.asarray(end_series, dtype=np.float64)
    delta = end_series - start_series
    finite = np.isfinite(delta)
    if not finite.any():
        return 0.0, 0.0, 0.0
    clean = np.where(finite, delta, 0.0)
    mean_delta = float(np.nanmean(delta))
    range_delta = float(np.nanmax(delta) - np.nanmin(delta)) if finite.any() else 0.0
    if clean.size <= 1:
        slope_delta = 0.0
    else:
        t = np.linspace(-0.5, 0.5, clean.size)
        slope_delta = float(np.sum(clean * t) / (np.sum(t * t) + EPS))
    return mean_delta, slope_delta, range_delta


def abs_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    return float(abs(np.corrcoef(x, y)[0, 1]))


def max_abs_corr(x, *ys):
    values = [abs_corr(x, y) for y in ys]
    finite = [v for v in values if np.isfinite(v)]
    return float(max(finite)) if finite else np.nan


def relevance_level(corr):
    if corr is None or not np.isfinite(float(corr)):
        return "unknown"
    corr = abs(float(corr))
    if corr > 0.75:
        return "strong"
    return "weak"


def _add_spec(specs, name, category, semantics, output_unit, func, meaning):
    specs.append({
        "name": name,
        "category": category,
        "features": [s["feature"] for s in semantics],
        "feature_indices": [int(s["idx"]) for s in semantics],
        "output_unit": output_unit,
        "meaning": meaning,
        "func": func,
    })


def _semantic_series(window, sem):
    return physical_feature_series(window, sem)


def _standardized_stack_series(series_list):
    clean = []
    for values in series_list:
        values = np.asarray(values, dtype=np.float64)
        std = np.nanstd(values)
        if not np.isfinite(std) or std <= EPS:
            clean.append(np.zeros_like(values, dtype=np.float64))
        else:
            clean.append((values - np.nanmean(values)) / std)
    return clean


def _operation_func(operation, semantics):
    operation = normalize_operation(operation)
    if operation == "difference":
        if len(semantics) != 2:
            raise ValueError("difference requires exactly two fields")
        return lambda w, a=semantics[0], b=semantics[1]: _semantic_series(w, a) - _semantic_series(w, b)
    if operation == "abs_difference":
        if len(semantics) != 2:
            raise ValueError("abs_difference requires exactly two fields")
        return lambda w, a=semantics[0], b=semantics[1]: np.abs(_semantic_series(w, a) - _semantic_series(w, b))
    if operation == "ratio":
        if len(semantics) != 2:
            raise ValueError("ratio requires exactly two fields")
        return lambda w, a=semantics[0], b=semantics[1]: safe_ratio(_semantic_series(w, a), _semantic_series(w, b))
    if operation == "product":
        if len(semantics) < 2:
            raise ValueError("product requires at least two fields")
        def product_func(w, sems=tuple(semantics)):
            out = np.ones_like(_semantic_series(w, sems[0]), dtype=np.float64)
            for sem in sems:
                out = out * _semantic_series(w, sem)
            return out
        return product_func
    if operation in {"range", "center"}:
        raise ValueError(f"{operation} is single-field stability evidence, not a semantic coupling operation")
    if operation == "co_movement":
        if len(semantics) < 2:
            raise ValueError("co_movement requires at least two fields")
        def co_movement_func(w, sems=tuple(semantics)):
            series = _standardized_stack_series([_semantic_series(w, sem) for sem in sems])
            return np.nanmean(np.vstack(series), axis=0)
        return co_movement_func
    if operation == "direction_agreement":
        if len(semantics) < 2:
            raise ValueError("direction_agreement requires at least two fields")
        def direction_func(w, sems=tuple(semantics)):
            diffs = [np.sign(np.diff(_semantic_series(w, sem), prepend=np.nan)) for sem in sems]
            stacked = np.vstack(diffs)
            ref = stacked[0]
            agree = np.nanmean(stacked == ref, axis=0)
            return np.where(np.isfinite(agree), agree, 0.0)
        return direction_func
    supported = "difference, abs_difference, ratio, product, co_movement, direction_agreement"
    if operation in {"symmetry", "lsr_symmetry"}:
        raise ValueError(
            "Unsupported semantic coupling operation: "
            f"{operation!r}. Use operation='difference' or 'abs_difference' for symmetry candidates, "
            "and put symmetry/lsr_symmetry in type/category/name/meaning. "
            f"Supported operations: {supported}"
        )
    raise ValueError(f"Unsupported semantic coupling operation: {operation!r}. Supported operations: {supported}")


def normalize_operation(operation):
    operation = str(operation).strip()
    if operation in {"consistency", "direction_consistency"}:
        return "direction_agreement"
    return operation


def _candidate_fields(candidate):
    fields = candidate.get("fields", [])
    out = []
    for item in fields:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            raw_name = item.get("raw_name") or item.get("name") or item.get("field")
            if raw_name:
                out.append(str(raw_name))
    return out


def _validate_semantic_candidate(candidate, semantics):
    operation = normalize_operation(candidate.get("operation", ""))
    units = [sem["unit"] for sem in semantics]
    quantities = [sem["quantity"] for sem in semantics]

    if operation in {"range", "center"}:
        raise ValueError("range/center max-min pairs are not allowed in semantic coupling candidates")
    if any(sem["unit"] in {"min", "count"} for sem in semantics):
        raise ValueError("runtime/count/background fields cannot be core semantic couplings")

    if operation in {"difference", "abs_difference"}:
        if len(set(units)) != 1:
            raise ValueError(f"{operation} requires matching units, got units={units}")
    if operation == "ratio":
        if "dB" in units:
            raise ValueError("ratio is not allowed for dB fields")
        if len(set(units)) != 1:
            raise ValueError(f"ratio currently requires matching units, got units={units}")
    if operation == "product":
        if sorted(units) != ["A", "V"]:
            raise ValueError(f"product currently requires exactly one V field and one A field, got units={units}")
    return quantities


def _candidate_label(candidate, idx):
    if not isinstance(candidate, dict):
        return f"candidate_{idx:03d}"
    return str(candidate.get("coupling_id") or candidate.get("name") or f"candidate_{idx:03d}")


def default_coupling_meaning(operation, semantics):
    names = [display_feature_name(sem["feature"]) for sem in semantics]
    if operation == "difference" and len(names) == 2:
        return f"公式: {names[0]} - {names[1]}。用于观察两个字段之间的差值变化是否随风险升高而稳定增强。"
    if operation == "abs_difference" and len(names) == 2:
        return f"公式: |{names[0]} - {names[1]}|。用于观察两个字段之间的不对称程度或偏离程度是否随风险升高而稳定增强。"
    if operation == "ratio" and len(names) == 2:
        return f"公式: {names[0]} / {names[1]}。用于观察两个同单位字段的相对比例变化是否随风险升高而稳定增强。"
    if operation == "product" and len(names) >= 2:
        return f"公式: {' * '.join(names)}。用于观察电压与电流组合形成的电功率类候选变化是否随风险升高而稳定增强。"
    if operation == "co_movement":
        return f"公式: {'、'.join(names)} 的标准化协同变化。用于观察这些字段是否在风险升高方向上共同变化。"
    if operation == "direction_agreement":
        return f"公式: {'、'.join(names)} 的变化方向一致性。用于观察这些字段在风险升高方向上的变化方向是否保持一致。"
    return f"公式: {operation}({', '.join(names)})。用于观察该语义耦合候选是否随风险升高而稳定变化。"


def build_semantic_coupling_specs(feature_names, semantic_couplings_path):
    path = Path(semantic_couplings_path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        candidates = None
        for key in ("candidates", "semantic_couplings", "couplings", "semantic_coupling_candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break
        if candidates is None and {"fields", "operation"}.issubset(payload):
            candidates = [payload]
    else:
        candidates = payload
    if not isinstance(candidates, list):
        raise ValueError(
            "Semantic coupling JSON must be a list, a single candidate object, "
            "or an object with a candidates/semantic_couplings/couplings list"
        )

    semantics = feature_semantics(feature_names)
    by_feature = {sem["feature"]: sem for sem in semantics}
    specs = []
    skipped = []
    for idx, candidate in enumerate(candidates, start=1):
        label = _candidate_label(candidate, idx)
        if not isinstance(candidate, dict):
            skipped.append((label, "candidate is not an object"))
            continue
        raw_fields = _candidate_fields(candidate)
        if not raw_fields:
            skipped.append((label, "candidate has no fields"))
            continue
        missing = [name for name in raw_fields if name not in by_feature]
        if missing:
            skipped.append((label, f"references missing fields: {missing}"))
            continue
        try:
            sems = [by_feature[name] for name in raw_fields]
            _validate_semantic_candidate(candidate, sems)
            operation = normalize_operation(candidate.get("operation", ""))
            func = _operation_func(operation, sems)
        except ValueError as exc:
            skipped.append((label, str(exc)))
            continue
        name = str(candidate.get("name") or candidate.get("coupling_id") or f"semantic_coupling_{idx:03d}")
        meaning = str(candidate.get("meaning") or candidate.get("why_candidate") or "").strip()
        if not meaning:
            meaning = default_coupling_meaning(operation, sems)
        specs.append({
            "name": name,
            "category": str(candidate.get("type") or candidate.get("category") or "semantic_generated"),
            "features": [sem["feature"] for sem in sems],
            "feature_indices": [int(sem["idx"]) for sem in sems],
            "output_unit": str(candidate.get("expected_output_unit") or candidate.get("output_unit") or "derived"),
            "meaning": meaning,
            "operation": operation,
            "func": func,
        })
    if skipped:
        preview = "; ".join(f"{label}: {reason}" for label, reason in skipped[:8])
        if len(skipped) > 8:
            preview += f"; ... {len(skipped) - 8} more"
        print(
            f"[semantic-coupling] Skipped {len(skipped)} invalid candidate(s); "
            f"kept {len(specs)} valid candidate(s). {preview}"
        )
    return specs


def semantic_coupling_contributions(
    start_windows,
    end_windows,
    morph_df,
    feature_names,
    preprocess_summary,
    top_n=12,
    semantic_couplings_path=None,
):
    if semantic_couplings_path:
        specs = build_semantic_coupling_specs(feature_names, semantic_couplings_path)
    else:
        specs = []
    if not specs:
        return pd.DataFrame(), []

    raw_start = restore_raw_windows(start_windows, feature_names, preprocess_summary)
    raw_end = restore_raw_windows(end_windows, feature_names, preprocess_summary)
    rows = []
    for hf_id in sorted(morph_df["hf_id"].unique()):
        idx = morph_df.index[morph_df["hf_id"] == hf_id].to_numpy()
        risk_values = pd.to_numeric(morph_df.loc[idx, "risk_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        logit_values = pd.to_numeric(morph_df.loc[idx, "logit_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for spec in specs:
            mean_deltas = []
            slope_deltas = []
            range_deltas = []
            window_strengths = []
            for i in idx:
                try:
                    start_series = spec["func"](raw_start[i])
                    end_series = spec["func"](raw_end[i])
                except Exception:
                    continue
                mean_delta, slope_delta, range_delta = summarize_series_delta(start_series, end_series)
                mean_deltas.append(mean_delta)
                slope_deltas.append(slope_delta)
                range_deltas.append(range_delta)
                window_strengths.append(float(np.sqrt(mean_delta ** 2 + slope_delta ** 2 + range_delta ** 2)))
            if not mean_deltas:
                continue
            mean_delta = float(np.nanmean(mean_deltas))
            slope_delta = float(np.nanmean(slope_deltas))
            range_delta = float(np.nanmean(range_deltas))
            strength = float(np.sqrt(mean_delta ** 2 + slope_delta ** 2 + range_delta ** 2))
            risk_corr = max_abs_corr(window_strengths, risk_values, logit_values)
            rows.append({
                "hf_id": hf_id,
                "coupling": spec["name"],
                "category": spec["category"],
                "meaning": spec["meaning"],
                "features": " | ".join(spec["features"]),
                "feature_indices": ",".join(str(i) for i in spec["feature_indices"]),
                "output_unit": spec["output_unit"],
                "mean_delta": mean_delta,
                "slope_delta": slope_delta,
                "range_delta": range_delta,
                "abs_strength": strength,
                "risk_corr": risk_corr,
                "relevance_level": relevance_level(risk_corr),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out, specs
    out = (
        out.sort_values(["hf_id", "abs_strength"], ascending=[True, False])
        .reset_index(drop=True)
    )
    serializable_specs = [{k: v for k, v in spec.items() if k != "func"} for spec in specs]
    return out, serializable_specs
