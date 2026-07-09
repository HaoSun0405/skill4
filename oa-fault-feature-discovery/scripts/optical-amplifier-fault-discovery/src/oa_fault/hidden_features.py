import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .coupling_features import semantic_coupling_contributions
from .datasets import WindowDataset
from .models import FaultTCN, WindowVAE


COUPLING_NAME_TRANSLATIONS = {
    "die_temperature": "管芯温度",
    "chip_temperature": "管芯温度",
    "tec_voltage": "TEC电压",
    "drive_current": "驱动电流",
    "cooling_current": "制冷电流",
    "backlight_current": "背光电流",
    "visible_gain": "可见增益",
    "edfa_gain": "EDFA增益",
    "power": "光功率",
}


COUPLING_CATEGORY_TRANSLATIONS = {
    "range": "窗口范围变化",
    "difference": "差值关系",
}


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_names(data_dir):
    summary_path = Path(data_dir) / "preprocess_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing preprocess summary: {summary_path}")
    return read_json(summary_path).get("feature_cols", [])


def load_preprocess_summary(data_dir):
    summary_path = Path(data_dir) / "preprocess_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing preprocess summary: {summary_path}")
    return read_json(summary_path)


def load_predictor(path, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]["predictor"]
    model = FaultTCN(
        n_features=int(checkpoint["n_features"]),
        hidden_channels=int(cfg.get("hidden_channels", 96)),
        num_blocks=int(cfg.get("num_blocks", 5)),
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_vae(path, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]["generator"]
    model = WindowVAE(
        n_features=int(checkpoint["n_features"]),
        window_len=int(checkpoint["window_len"]),
        latent_dim=int(cfg.get("latent_dim", 32)),
        hidden_channels=int(cfg.get("hidden_channels", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def group_for_feature(name):
    compact = str(name).replace(" ", "")
    if "模块壳温" in compact:
        return "模块壳温组"
    if "扣板温度" in compact:
        return "扣板温度组"
    if "主板温度1" in compact:
        return "主板温度1组"
    if "主板温度2" in compact:
        return "主板温度2组"
    if "LSR1管芯温度" in compact:
        return "LSR1管芯温度组"
    if "LSR2管芯温度" in compact:
        return "LSR2管芯温度组"
    if "EDFA增益" in compact:
        return "EDFA增益组"
    if "可见增益" in compact:
        return "可见增益组"
    if "PAIN" in compact:
        return "PAIN光功率组"
    if "PAOUT" in compact:
        return "PAOUT光功率组"
    if "BAIN" in compact:
        return "BAIN光功率组"
    if "BAOUT" in compact:
        return "BAOUT光功率组"
    if "LSR1驱动电流" in compact:
        return "LSR1驱动电流组"
    if "LSR2驱动电流" in compact:
        return "LSR2驱动电流组"
    if "LSR1制冷电流" in compact:
        return "LSR1制冷电流组"
    if "LSR2制冷电流" in compact:
        return "LSR2制冷电流组"
    if "LSR1背光电流" in compact:
        return "LSR1背光电流组"
    if "LSR2背光电流" in compact:
        return "LSR2背光电流组"
    if "电源" in compact or "3V3" in compact:
        return "电源电压组"
    if "温度" in compact or "壳温" in compact or "扣板" in compact or "主板" in compact or "管芯" in compact:
        return "其他温度组"
    if "增益" in compact:
        return "其他增益组"
    if "光功率" in compact:
        return "其他光功率组"
    if "驱动电流" in compact:
        return "其他驱动电流组"
    if "制冷电流" in compact:
        return "其他制冷电流组"
    if "背光电流" in compact:
        return "其他背光电流组"
    if "TEC" in compact:
        if "LSR1" in compact:
            return "LSR1TEC电压组"
        if "LSR2" in compact:
            return "LSR2TEC电压组"
        return "TEC电压组"
    if "运行时间" in compact or "上电" in compact or "超温" in compact or "累计" in compact:
        return "运行时间组"
    return "其他组"


def field_semantics(name):
    compact = str(name).replace(" ", "")
    tags = []
    constraints = []
    allow_as_core_driver = True

    is_recent_power_runtime = "最近一次" in compact and "上电后" in compact and "运行时间" in compact
    is_strict_monotonic = (
        "上电次数" in compact
        or ("累计" in compact and not is_recent_power_runtime)
        or ("超温运行时间" in compact and not is_recent_power_runtime)
    )
    if is_strict_monotonic:
        tags.append("monotonic_background")
        constraints.append("non_decreasing")
        allow_as_core_driver = True
    elif is_recent_power_runtime:
        tags.append("resettable_runtime_background")
        allow_as_core_driver = True
    if "LSR1" in compact:
        tags.append("paired_component_signal")
        component = "LSR1"
    elif "LSR2" in compact:
        tags.append("paired_component_signal")
        component = "LSR2"
    else:
        component = None
    if any(token in compact for token in ("温度", "壳温", "光功率", "电流", "电压", "电源", "3V3", "增益")):
        tags.append("state_signal")
    if not tags:
        tags.append("unknown")

    return {
        "feature": str(name),
        "group": group_for_feature(name),
        "tags": sorted(set(tags)),
        "constraints": constraints,
        "allow_as_core_driver": allow_as_core_driver,
        "component": component,
    }


def build_feature_groups(feature_names):
    groups = {}
    for idx, name in enumerate(feature_names):
        groups.setdefault(group_for_feature(name), []).append(idx)
    return {k: v for k, v in groups.items() if v}


def is_runtime_group(group):
    return "运行时间" in str(group)


def series_slope(arr, axis=-1):
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.shape[axis]
    if n <= 1:
        return np.zeros(arr.shape[:axis] + arr.shape[axis + 1 :], dtype=np.float64)
    t = np.linspace(-0.5, 0.5, n, dtype=np.float64)
    denom = np.sum(t * t) + 1e-12
    moved = np.moveaxis(arr, axis, -1)
    return np.sum(moved * t, axis=-1) / denom


def summarize_window(x):
    x = np.asarray(x, dtype=np.float64)
    half = max(x.shape[-1] // 2, 1)
    return {
        "mean": x.mean(axis=-1),
        "early": x[..., :half].mean(axis=-1),
        "late": x[..., half:].mean(axis=-1),
        "delta_late_early": x[..., half:].mean(axis=-1) - x[..., :half].mean(axis=-1),
        "slope": series_slope(x, axis=-1),
        "volatility": x.std(axis=-1),
        "range": x.max(axis=-1) - x.min(axis=-1),
    }


def feature_index_lookup(feature_names):
    return {str(name).replace(" ", ""): idx for idx, name in enumerate(feature_names)}


def find_feature_idx(feature_names, contains_all):
    lookup_names = [(idx, str(name).replace(" ", "")) for idx, name in enumerate(feature_names)]
    for idx, compact in lookup_names:
        if all(token in compact for token in contains_all):
            return idx
    return None


def build_physical_relations(feature_names):
    specs = []

    def add_pair(name, meaning, left_tokens, right_tokens, relation_type="difference"):
        left = find_feature_idx(feature_names, left_tokens)
        right = find_feature_idx(feature_names, right_tokens)
        if left is not None and right is not None:
            specs.append({
                "name": name,
                "meaning": meaning,
                "type": relation_type,
                "left_idx": left,
                "right_idx": right,
                "left_feature": feature_names[left],
                "right_feature": feature_names[right],
            })

    for stat in ["MAX", "MIN"]:
        add_pair(
            f"PA级功率增益{stat}",
            "PAOUT 与 PAIN 的差值变化；若字段为 dB 类单位，可近似理解为 PA 级输入输出增益变化。",
            ["PAOUT", stat],
            ["PAIN", stat],
        )
        add_pair(
            f"BA级功率增益{stat}",
            "BAOUT 与 BAIN 的差值变化；若字段为 dB 类单位，可近似理解为 BA 级输入输出增益变化。",
            ["BAOUT", stat],
            ["BAIN", stat],
        )
        add_pair(
            f"EDFA-可见增益偏离{stat}",
            "EDFA 增益与可见增益之间的差值变化，用于观察两类增益读数是否同步偏离。",
            ["EDFA增益", stat],
            ["可见增益", stat],
        )
        for metric in ["驱动电流", "制冷电流", "背光电流", "TEC电压", "管芯温度"]:
            add_pair(
                f"LSR1-LSR2{metric}差异{stat}",
                f"LSR1 与 LSR2 的{metric}差值变化，用于观察双通道对称性是否改变。",
                ["LSR1", metric, stat],
                ["LSR2", metric, stat],
            )

    # Same-signal MAX/MIN range changes.
    bases = [
        ("PAIN光功率范围", ["PAIN"]),
        ("PAOUT光功率范围", ["PAOUT"]),
        ("BAIN光功率范围", ["BAIN"]),
        ("BAOUT光功率范围", ["BAOUT"]),
        ("EDFA增益范围", ["EDFA增益"]),
        ("可见增益范围", ["可见增益"]),
        ("LSR1驱动电流范围", ["LSR1", "驱动电流"]),
        ("LSR2驱动电流范围", ["LSR2", "驱动电流"]),
        ("LSR1制冷电流范围", ["LSR1", "制冷电流"]),
        ("LSR2制冷电流范围", ["LSR2", "制冷电流"]),
        ("LSR1背光电流范围", ["LSR1", "背光电流"]),
        ("LSR2背光电流范围", ["LSR2", "背光电流"]),
        ("LSR1TEC电压范围", ["LSR1", "TEC电压"]),
        ("LSR2TEC电压范围", ["LSR2", "TEC电压"]),
        ("LSR1管芯温度范围", ["LSR1", "管芯温度"]),
        ("LSR2管芯温度范围", ["LSR2", "管芯温度"]),
    ]
    for name, tokens in bases:
        max_idx = find_feature_idx(feature_names, tokens + ["MAX"])
        min_idx = find_feature_idx(feature_names, tokens + ["MIN"])
        if max_idx is not None and min_idx is not None:
            specs.append({
                "name": name,
                "meaning": "同一指标 MAX 与 MIN 的差值变化，用于观察窗口内范围/波动是否扩大。",
                "type": "range",
                "left_idx": max_idx,
                "right_idx": min_idx,
                "left_feature": feature_names[max_idx],
                "right_feature": feature_names[min_idx],
            })
    return specs


def relation_series(x, spec):
    left = x[spec["left_idx"]]
    right = x[spec["right_idx"]]
    return left - right




def delta_vector(x_start, x_final, feature_names, groups):
    delta = np.asarray(x_final - x_start, dtype=np.float64)
    summary = summarize_window(delta)
    values = {}
    for i, name in enumerate(feature_names):
        values[f"field::{name}::mean_delta"] = float(summary["mean"][i])
        values[f"field::{name}::slope_delta"] = float(summary["slope"][i])
        values[f"field::{name}::volatility_delta"] = float(summary["volatility"][i])
        values[f"field::{name}::range_delta"] = float(summary["range"][i])
    for group, indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        values[f"group::{group}::mean_delta"] = float(np.mean(summary["mean"][idx]))
        values[f"group::{group}::slope_delta"] = float(np.mean(summary["slope"][idx]))
        values[f"group::{group}::volatility_delta"] = float(np.mean(summary["volatility"][idx]))
    return values


def flatten_feature_dicts(dicts):
    keys = sorted({key for row in dicts for key in row})
    arr = np.zeros((len(dicts), len(keys)), dtype=np.float32)
    for i, row in enumerate(dicts):
        for j, key in enumerate(keys):
            arr[i, j] = float(row.get(key, 0.0))
    return arr, keys


def choose_indices(dataset, pred_df, max_windows, seed, selection_mode="low_mid"):
    meta = dataset.metadata()
    rng = np.random.default_rng(seed)
    if pred_df is not None and "pred_score" in pred_df.columns and len(pred_df) == len(meta):
        scores = pd.to_numeric(pred_df["pred_score"], errors="coerce").fillna(0.0).to_numpy()
        if selection_mode == "high":
            cutoff = np.quantile(scores, 0.7)
            candidates = np.where(scores >= cutoff)[0]
        elif selection_mode == "all":
            candidates = np.arange(len(meta))
        else:
            lower, upper = np.quantile(scores, [0.1, 0.85])
            candidates = np.where((scores >= lower) & (scores <= upper))[0]
        if len(candidates) < min(max_windows, len(meta)) // 3:
            candidates = np.arange(len(meta))
    else:
        candidates = np.arange(len(meta))
    if len(candidates) > max_windows:
        candidates = rng.choice(candidates, size=max_windows, replace=False)
    return np.sort(candidates)


def predict_scores(model, x, device):
    with torch.no_grad():
        return torch.sigmoid(model(x.to(device))).detach().cpu().numpy().reshape(-1)


def estimate_risk_centroid_direction(dataset, pred_df, vae, device, max_windows=5000, seed=42):
    if pred_df is None or "pred_score" not in pred_df.columns or len(pred_df) != len(dataset):
        print("[hidden] risk-centroid fallback: missing prediction scores; using labels if available.")
        scores = np.asarray([float(dataset[i][1].item()) for i in range(len(dataset))], dtype=float)
    else:
        scores = pd.to_numeric(pred_df["pred_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(dataset))
    if len(all_idx) > max_windows:
        all_idx = rng.choice(all_idx, size=max_windows, replace=False)

    local_scores = scores[all_idx]
    low_cut = np.quantile(local_scores, 0.25)
    high_cut = np.quantile(local_scores, 0.75)
    low_idx = all_idx[local_scores <= low_cut]
    high_idx = all_idx[local_scores >= high_cut]
    if len(low_idx) == 0 or len(high_idx) == 0:
        raise ValueError("Unable to estimate risk-centroid direction: high or low score windows are empty.")

    def encode_mean(indices):
        subset = Subset(dataset, indices.tolist())
        loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)
        zs = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="[hidden] Encoding risk-centroid windows", leave=False):
                mu, _ = vae.encode(x.to(device))
                zs.append(mu.detach().cpu())
        return torch.cat(zs, dim=0).mean(dim=0)

    z_low = encode_mean(low_idx)
    z_high = encode_mean(high_idx)
    direction = z_high - z_low
    norm = torch.linalg.vector_norm(direction).clamp_min(1e-8)
    direction = direction / norm
    print(
        "[hidden] risk-centroid direction: "
        f"low_windows={len(low_idx)}, high_windows={len(high_idx)}, "
        f"low_score_mean={float(np.mean(scores[low_idx])):.6g}, "
        f"high_score_mean={float(np.mean(scores[high_idx])):.6g}"
    )
    return direction.to(device)


def apply_latent_norm_limit(z, z0, max_latent_norm):
    if max_latent_norm is None or max_latent_norm <= 0:
        return z.detach()
    shift = z - z0
    shift_norm = torch.linalg.vector_norm(shift, dim=1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(float(max_latent_norm) / shift_norm, max=1.0)
    return (z0 + shift * scale).detach()


def non_decreasing_feature_indices(feature_names):
    indices = []
    for idx, name in enumerate(feature_names or []):
        meta = field_semantics(name)
        if "non_decreasing" in meta.get("constraints", []):
            indices.append(idx)
    return indices


def project_non_decreasing_features(start, final, feature_indices):
    if not feature_indices:
        zeros = torch.zeros(start.shape[0], dtype=torch.long, device=start.device)
        return final, zeros
    projected = final.clone()
    idx = torch.as_tensor(feature_indices, dtype=torch.long, device=final.device)
    before = projected[:, idx, :]
    floor = start[:, idx, :]
    adjusted_mask = before < floor
    projected[:, idx, :] = torch.maximum(before, floor)
    adjusted_counts = adjusted_mask.reshape(adjusted_mask.shape[0], -1).sum(dim=1)
    return projected, adjusted_counts


def morph_batch_gradient_core(
    predictor,
    vae,
    z0,
    start,
    logit_start,
    pred_start,
    device,
    steps,
    step_size,
    max_latent_norm,
    target_logit_delta=3.0,
    target_pred_score=None,
):
    batch_size = z0.shape[0]
    selected = torch.zeros(batch_size, dtype=torch.bool, device=device)
    selected_step = torch.zeros(batch_size, dtype=torch.long, device=device)
    selected_final = start.detach().clone()
    selected_logit = logit_start.detach().clone()
    selected_pred = pred_start.detach().clone()
    last_final = start.detach().clone()
    last_logit = logit_start.detach().clone()
    last_pred = pred_start.detach().clone()

    z = z0.clone().detach().requires_grad_(True)
    for step in range(1, int(steps) + 1):
        decoded = vae.decode(z)
        logits = predictor(decoded).reshape(-1)
        objective = logits.sum()
        grad = torch.autograd.grad(objective, z, retain_graph=False, create_graph=False)[0]
        grad_norm = torch.linalg.vector_norm(grad, dim=1, keepdim=True).clamp_min(1e-8)
        z = (z + float(step_size) * grad / grad_norm).detach()
        z = apply_latent_norm_limit(z, z0, max_latent_norm)
        with torch.no_grad():
            decoded_step = vae.decode(z)
            logits_step = predictor(decoded_step).detach()
            preds_step = torch.sigmoid(logits_step).detach()
        last_final, last_logit, last_pred = decoded_step.detach(), logits_step.detach(), preds_step.detach()
        logit_delta = (logits_step - logit_start).reshape(-1)
        condition = logit_delta >= float(target_logit_delta)
        if target_pred_score is not None:
            condition = condition | (preds_step.reshape(-1) >= float(target_pred_score))
        new_select = condition & (~selected)
        if new_select.any():
            selected_final[new_select] = decoded_step.detach()[new_select]
            selected_logit[new_select] = logits_step.detach()[new_select]
            selected_pred[new_select] = preds_step.detach()[new_select]
            selected_step[new_select] = step
            selected[new_select] = True
        z.requires_grad_(True)

    with torch.no_grad():
        not_selected = ~selected
        if not_selected.any():
            selected_final[not_selected] = last_final[not_selected]
            selected_logit[not_selected] = last_logit[not_selected]
            selected_pred[not_selected] = last_pred[not_selected]
            selected_step[not_selected] = int(steps)
    return {
        "final": selected_final,
        "last_final": last_final,
        "selected_step": selected_step,
        "target_reached": selected,
        "logit_final": selected_logit,
        "last_logit_final": last_logit,
        "pred_final": selected_pred,
        "last_pred_final": last_pred,
    }


def morph_batch(
    predictor,
    vae,
    x,
    device,
    steps,
    step_size,
    max_latent_norm,
    method="gradient",
    risk_direction=None,
    target_logit_delta=3.0,
    target_pred_score=None,
    feature_names=None,
):
    x = x.to(device)
    with torch.no_grad():
        mu, _ = vae.encode(x)
        start = vae.decode(mu)
        logit_start = predictor(start).detach()
        pred_start = torch.sigmoid(logit_start).detach()

    z0 = mu.detach()
    batch_size = x.shape[0]
    selected = torch.zeros(batch_size, dtype=torch.bool, device=device)
    selected_step = torch.zeros(batch_size, dtype=torch.long, device=device)
    selected_final = start.detach().clone()
    selected_logit = logit_start.detach().clone()
    selected_pred = pred_start.detach().clone()
    last_final = start.detach().clone()
    last_logit = logit_start.detach().clone()
    last_pred = pred_start.detach().clone()

    if method == "risk-centroid":
        if risk_direction is None:
            raise ValueError("risk_direction is required when method='risk-centroid'")
        direction = risk_direction.view(1, -1).to(device)
        for step in range(1, int(steps) + 1):
            z = z0 + float(step) * float(step_size) * direction
            z = apply_latent_norm_limit(z, z0, max_latent_norm)
            with torch.no_grad():
                decoded = vae.decode(z)
                logits = predictor(decoded).detach()
                preds = torch.sigmoid(logits).detach()
            last_final, last_logit, last_pred = decoded.detach(), logits.detach(), preds.detach()
            logit_delta = (logits - logit_start).reshape(-1)
            condition = logit_delta >= float(target_logit_delta)
            if target_pred_score is not None:
                condition = condition | (preds.reshape(-1) >= float(target_pred_score))
            new_select = condition & (~selected)
            if new_select.any():
                selected_final[new_select] = decoded.detach()[new_select]
                selected_logit[new_select] = logits.detach()[new_select]
                selected_pred[new_select] = preds.detach()[new_select]
                selected_step[new_select] = step
                selected[new_select] = True
    elif method in ("gradient", "conservative-gradient", "multi-gradient"):
        if method == "conservative-gradient":
            candidates = [(float(step_size) * 0.5, float(max_latent_norm) * 0.5, float(target_logit_delta) * 0.75)]
        elif method == "multi-gradient":
            candidates = [
                (float(step_size) * 0.5, float(max_latent_norm) * 0.5, float(target_logit_delta) * 0.75),
                (float(step_size), float(max_latent_norm), float(target_logit_delta)),
                (float(step_size) * 1.5, float(max_latent_norm), float(target_logit_delta) * 1.25),
            ]
        else:
            candidates = [(float(step_size), float(max_latent_norm), float(target_logit_delta))]

        candidate_outs = []
        for candidate_idx, (cand_step, cand_norm, cand_target) in enumerate(candidates):
            out = morph_batch_gradient_core(
                predictor=predictor,
                vae=vae,
                z0=z0,
                start=start,
                logit_start=logit_start,
                pred_start=pred_start,
                device=device,
                steps=steps,
                step_size=cand_step,
                max_latent_norm=cand_norm,
                target_logit_delta=cand_target,
                target_pred_score=target_pred_score,
            )
            out["candidate_idx"] = torch.full((batch_size,), candidate_idx, dtype=torch.long, device=device)
            out["candidate_step_size"] = torch.full((batch_size,), cand_step, dtype=torch.float32, device=device)
            out["candidate_max_latent_norm"] = torch.full((batch_size,), cand_norm, dtype=torch.float32, device=device)
            out["candidate_target_logit_delta"] = torch.full((batch_size,), cand_target, dtype=torch.float32, device=device)
            candidate_outs.append(out)

        if len(candidate_outs) == 1:
            chosen = candidate_outs[0]
        else:
            scores = []
            monotonic_indices = non_decreasing_feature_indices(feature_names)
            for out in candidate_outs:
                projected_final, adjusted_counts = project_non_decreasing_features(start, out["final"], monotonic_indices)
                projected_logit = predictor(projected_final).detach()
                logit_gain = (projected_logit - logit_start).reshape(-1)
                mse = torch.mean((start - projected_final) ** 2, dim=(1, 2))
                reached_bonus = out["target_reached"].float() * 0.25
                projection_penalty = adjusted_counts.float() / max(1, len(monotonic_indices) * start.shape[-1])
                scores.append(logit_gain + reached_bonus - 0.35 * mse - 0.15 * projection_penalty)
            score_mat = torch.stack(scores, dim=0)
            best_idx = torch.argmax(score_mat, dim=0)
            chosen = {}
            for key, first_val in candidate_outs[0].items():
                chosen_val = first_val.clone()
                for sample_idx in range(batch_size):
                    chosen_val[sample_idx] = candidate_outs[int(best_idx[sample_idx].item())][key][sample_idx]
                chosen[key] = chosen_val

        selected = chosen["target_reached"]
        selected_step = chosen["selected_step"]
        selected_final = chosen["final"]
        selected_logit = chosen["logit_final"]
        selected_pred = chosen["pred_final"]
        last_final = chosen["last_final"]
        last_logit = chosen["last_logit_final"]
        last_pred = chosen["last_pred_final"]
        candidate_idx = chosen["candidate_idx"]
        candidate_step_size = chosen["candidate_step_size"]
        candidate_max_latent_norm = chosen["candidate_max_latent_norm"]
        candidate_target_logit_delta = chosen["candidate_target_logit_delta"]
    else:
        raise ValueError(f"Unknown morph method: {method}")

    with torch.no_grad():
        not_selected = ~selected
        if not_selected.any():
            selected_final[not_selected] = last_final[not_selected]
            selected_logit[not_selected] = last_logit[not_selected]
            selected_pred[not_selected] = last_pred[not_selected]
            selected_step[not_selected] = int(steps)
        monotonic_indices = non_decreasing_feature_indices(feature_names)
        final_raw = selected_final
        last_final_raw = last_final
        final, projection_adjusted_values = project_non_decreasing_features(start, final_raw, monotonic_indices)
        last_final, last_projection_adjusted_values = project_non_decreasing_features(start, last_final_raw, monotonic_indices)
        logit_final = predictor(final).detach()
        pred_final = torch.sigmoid(logit_final).detach()
        last_logit = predictor(last_final).detach()
        last_pred = torch.sigmoid(last_logit).detach()
        logit_delta = (logit_final - logit_start).reshape(-1)
        selected = logit_delta >= float(target_logit_delta)
        if target_pred_score is not None:
            selected = selected | (pred_final.reshape(-1) >= float(target_pred_score))
        original_to_start_mse = torch.mean((x - start) ** 2, dim=(1, 2))
        start_to_final_mse = torch.mean((start - final) ** 2, dim=(1, 2))
    return {
        "start": start.detach().cpu().numpy(),
        "final": final.detach().cpu().numpy(),
        "last_final": last_final.detach().cpu().numpy(),
        "selected_step": selected_step.cpu().numpy().reshape(-1),
        "target_reached": selected.cpu().numpy().reshape(-1),
        "logit_start": logit_start.cpu().numpy().reshape(-1),
        "logit_final": logit_final.cpu().numpy().reshape(-1),
        "last_logit_final": last_logit.cpu().numpy().reshape(-1),
        "pred_start": pred_start.cpu().numpy().reshape(-1),
        "pred_final": pred_final.cpu().numpy().reshape(-1),
        "last_pred_final": last_pred.cpu().numpy().reshape(-1),
        "original_to_start_mse": original_to_start_mse.cpu().numpy().reshape(-1),
        "start_to_final_mse": start_to_final_mse.cpu().numpy().reshape(-1),
        "start_to_last_mse": torch.mean((start - last_final) ** 2, dim=(1, 2)).cpu().numpy().reshape(-1),
        "physical_projection_adjusted_values": projection_adjusted_values.cpu().numpy().reshape(-1),
        "last_physical_projection_adjusted_values": last_projection_adjusted_values.cpu().numpy().reshape(-1),
        "physical_projection_feature_count": np.full(batch_size, len(monotonic_indices), dtype=int),
        "candidate_idx": (candidate_idx.cpu().numpy().reshape(-1) if "candidate_idx" in locals() else np.zeros(batch_size, dtype=int)),
        "candidate_step_size": (candidate_step_size.cpu().numpy().reshape(-1) if "candidate_step_size" in locals() else np.full(batch_size, float(step_size))),
        "candidate_max_latent_norm": (candidate_max_latent_norm.cpu().numpy().reshape(-1) if "candidate_max_latent_norm" in locals() else np.full(batch_size, float(max_latent_norm))),
        "candidate_target_logit_delta": (candidate_target_logit_delta.cpu().numpy().reshape(-1) if "candidate_target_logit_delta" in locals() else np.full(batch_size, float(target_logit_delta))),
    }


def collect_morph_samples(dataset, local_indices, predictor, vae, feature_names, groups, args, device, risk_direction=None):
    subset = Subset(dataset, local_indices.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    meta = dataset.metadata().iloc[local_indices].reset_index(drop=True)

    morph_rows = []
    delta_rows = []
    last_delta_rows = []
    start_windows = []
    final_windows = []
    last_windows = []
    cursor = 0
    for x, y in tqdm(loader, desc="[hidden] Morphing windows", leave=False):
        out = morph_batch(
            predictor=predictor,
            vae=vae,
            x=x,
            device=device,
            steps=args.morph_steps,
            step_size=args.step_size,
            max_latent_norm=args.max_latent_norm,
            method=args.morph_method,
            risk_direction=risk_direction,
            target_logit_delta=args.target_logit_delta,
            target_pred_score=args.target_pred_score,
            feature_names=feature_names,
        )
        batch_size = len(x)
        for i in range(batch_size):
            meta_row = meta.iloc[cursor + i].to_dict()
            risk_delta = float(out["pred_final"][i] - out["pred_start"][i])
            logit_delta = float(out["logit_final"][i] - out["logit_start"][i])
            last_risk_delta = float(out["last_pred_final"][i] - out["pred_start"][i])
            last_logit_delta = float(out["last_logit_final"][i] - out["logit_start"][i])
            row = {
                "sample_id": int(cursor + i),
                "true_label": float(y[i].item()),
                "morph_method": args.morph_method,
                "selection_mode": args.selection_mode,
                "window_len": int(x.shape[-1]),
                "target_logit_delta": float(args.target_logit_delta),
                "target_pred_score": args.target_pred_score if args.target_pred_score is not None else "None",
                "selected_step": int(out["selected_step"][i]),
                "candidate_idx": int(out["candidate_idx"][i]),
                "candidate_step_size": float(out["candidate_step_size"][i]),
                "candidate_max_latent_norm": float(out["candidate_max_latent_norm"][i]),
                "candidate_target_logit_delta": float(out["candidate_target_logit_delta"][i]),
                "target_reached": bool(out["target_reached"][i]),
                "logit_start": float(out["logit_start"][i]),
                "logit_final": float(out["logit_final"][i]),
                "logit_delta": logit_delta,
                "last_logit_final": float(out["last_logit_final"][i]),
                "last_logit_delta": last_logit_delta,
                "pred_start": float(out["pred_start"][i]),
                "pred_final": float(out["pred_final"][i]),
                "risk_delta": risk_delta,
                "last_pred_final": float(out["last_pred_final"][i]),
                "last_risk_delta": last_risk_delta,
                "original_to_start_mse": float(out["original_to_start_mse"][i]),
                "start_to_final_mse": float(out["start_to_final_mse"][i]),
                "start_to_last_mse": float(out["start_to_last_mse"][i]),
                "physical_projection_adjusted_values": int(out["physical_projection_adjusted_values"][i]),
                "last_physical_projection_adjusted_values": int(out["last_physical_projection_adjusted_values"][i]),
                "physical_projection_feature_count": int(out["physical_projection_feature_count"][i]),
                **{k: meta_row.get(k) for k in meta_row},
            }
            morph_rows.append(row)
            delta_rows.append(delta_vector(out["start"][i], out["final"][i], feature_names, groups))
            last_delta_rows.append(delta_vector(out["start"][i], out["last_final"][i], feature_names, groups))
        start_windows.append(out["start"])
        final_windows.append(out["final"])
        last_windows.append(out["last_final"])
        cursor += batch_size

    return (
        pd.DataFrame(morph_rows),
        delta_rows,
        last_delta_rows,
        np.concatenate(start_windows, axis=0),
        np.concatenate(final_windows, axis=0),
        np.concatenate(last_windows, axis=0),
    )


def cluster_hidden_features(delta_rows, morph_df, n_clusters, seed):
    X, keys = flatten_feature_dicts(delta_rows)
    if len(X) < 2:
        labels = np.zeros(len(X), dtype=int)
        scaled = X
        centers = X[:1]
        scaler = None
    else:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        k = max(1, min(int(n_clusters), len(X)))
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(scaled)
        centers = model.cluster_centers_

    morph_df = morph_df.copy()
    morph_df["hf_id"] = [f"HF_{int(label) + 1:03d}" for label in labels]
    if len(np.unique(labels)) > 0:
        _, dist = pairwise_distances_argmin_min(scaled, centers)
        morph_df["cluster_distance"] = dist

    # Convert distance and risk gain into a soft activation probability.
    for hf_id in sorted(morph_df["hf_id"].unique()):
        mask = morph_df["hf_id"] == hf_id
        d = morph_df.loc[mask, "cluster_distance"].to_numpy(dtype=float)
        gain = morph_df.loc[mask, "risk_delta"].to_numpy(dtype=float)
        d_score = -(d - np.median(d)) / (np.std(d) + 1e-6)
        g_score = (gain - np.median(gain)) / (np.std(gain) + 1e-6)
        morph_df.loc[mask, "hf_score"] = d_score + g_score
        morph_df.loc[mask, "hf_prob"] = sigmoid(morph_df.loc[mask, "hf_score"].to_numpy(dtype=float))
    return morph_df, X, keys, labels


def abs_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
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


def top_field_contributions(delta_rows, morph_df, feature_names, top_n=12):
    rows = []
    for hf_id in sorted(morph_df["hf_id"].unique()):
        idx = morph_df.index[morph_df["hf_id"] == hf_id].to_numpy()
        risk_values = pd.to_numeric(morph_df.loc[idx, "risk_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        logit_values = pd.to_numeric(morph_df.loc[idx, "logit_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for feature in feature_names:
            mean_key = f"field::{feature}::mean_delta"
            slope_key = f"field::{feature}::slope_delta"
            vol_key = f"field::{feature}::volatility_delta"
            range_key = f"field::{feature}::range_delta"
            mean_vals = np.array([delta_rows[i].get(mean_key, 0.0) for i in idx], dtype=float)
            slope_vals = np.array([delta_rows[i].get(slope_key, 0.0) for i in idx], dtype=float)
            vol_vals = np.array([delta_rows[i].get(vol_key, 0.0) for i in idx], dtype=float)
            range_vals = np.array([delta_rows[i].get(range_key, 0.0) for i in idx], dtype=float)
            window_strengths = np.sqrt(mean_vals ** 2 + slope_vals ** 2 + vol_vals ** 2 + range_vals ** 2)
            mean_delta = float(np.mean(mean_vals))
            slope_delta = float(np.mean(slope_vals))
            volatility_delta = float(np.mean(vol_vals))
            range_delta = float(np.mean(range_vals))
            strength = float(np.sqrt(
                mean_delta ** 2
                + slope_delta ** 2
                + volatility_delta ** 2
                + range_delta ** 2
            ))
            risk_corr = max_abs_corr(window_strengths, risk_values, logit_values)
            rows.append({
                "hf_id": hf_id,
                "feature": feature,
                "group": group_for_feature(feature),
                "mean_delta": mean_delta,
                "slope_delta": slope_delta,
                "volatility_delta": volatility_delta,
                "range_delta": range_delta,
                "abs_strength": strength,
                "risk_corr": risk_corr,
                "relevance_level": relevance_level(risk_corr),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["hf_id", "abs_strength"], ascending=[True, False]).reset_index(drop=True)


def validate_physical_constraints(delta_rows, morph_df, feature_names, decrease_threshold=1e-6):
    semantics = {name: field_semantics(name) for name in feature_names}
    monotonic_features = [
        name for name, meta in semantics.items()
        if "non_decreasing" in meta.get("constraints", [])
    ]
    pattern_rows = []
    violation_rows = []
    total_violations = 0

    for hf_id in sorted(morph_df["hf_id"].unique()):
        idx = morph_df.index[morph_df["hf_id"] == hf_id].to_numpy()
        n_windows = int(len(idx))
        hf_violation_count = 0
        hf_checks = 0
        top_violations = []
        for feature in monotonic_features:
            key = f"field::{feature}::mean_delta"
            values = np.array([delta_rows[i].get(key, 0.0) for i in idx], dtype=float)
            if len(values) == 0:
                continue
            violated = values < -float(decrease_threshold)
            violation_count = int(np.sum(violated))
            hf_violation_count += violation_count
            hf_checks += int(len(values))
            if violation_count:
                rate = float(violation_count / max(len(values), 1))
                mean_delta = float(np.mean(values))
                row = {
                    "hf_id": hf_id,
                    "feature": feature,
                    "constraint": "non_decreasing",
                    "violation": "decreased_after_morph",
                    "violation_count": violation_count,
                    "checked_windows": int(len(values)),
                    "violation_rate": rate,
                    "mean_delta": mean_delta,
                }
                violation_rows.append(row)
                top_violations.append(row)
                total_violations += violation_count

        violation_rate = float(hf_violation_count / max(hf_checks, 1)) if hf_checks else 0.0
        sub = morph_df.loc[idx]
        shift_values = pd.to_numeric(sub.get("start_to_final_mse", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        risk_values = pd.to_numeric(sub.get("risk_delta", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        large_shift_rate = float(np.mean(shift_values.to_numpy() > 1.0)) if len(shift_values) else 0.0
        non_positive_risk_rate = float(np.mean(risk_values.to_numpy() <= 0.0)) if len(risk_values) else 0.0
        physical_score = float(np.clip(
            1.0
            - 0.60 * violation_rate
            - 0.25 * large_shift_rate
            - 0.15 * non_positive_risk_rate,
            0.0,
            1.0,
        ))
        if physical_score < 0.5:
            validity = "low"
        elif physical_score < 0.8:
            validity = "medium"
        else:
            validity = "high"
        pattern_rows.append({
            "hf_id": hf_id,
            "n_windows": n_windows,
            "physical_validity": validity,
            "monotonic_checked_features": len(monotonic_features),
            "monotonic_checked_values": hf_checks,
            "monotonic_violation_count": hf_violation_count,
            "monotonic_violation_rate": violation_rate,
            "large_shift_rate": large_shift_rate,
            "non_positive_risk_rate": non_positive_risk_rate,
            "physical_score": physical_score,
            "top_violations": sorted(
                top_violations,
                key=lambda row: row["violation_rate"],
                reverse=True,
            )[:5],
        })

    overall_checks = int(sum(row["monotonic_checked_values"] for row in pattern_rows))
    overall_rate = float(total_violations / max(overall_checks, 1)) if overall_checks else 0.0
    if pattern_rows:
        overall_large_shift_rate = float(np.mean([row["large_shift_rate"] for row in pattern_rows]))
        overall_non_positive_risk_rate = float(np.mean([row["non_positive_risk_rate"] for row in pattern_rows]))
    else:
        overall_large_shift_rate = 0.0
        overall_non_positive_risk_rate = 0.0
    overall_score = float(np.clip(
        1.0
        - 0.60 * overall_rate
        - 0.25 * overall_large_shift_rate
        - 0.15 * overall_non_positive_risk_rate,
        0.0,
        1.0,
    ))
    overall_validity = "low" if overall_score < 0.5 else "medium" if overall_score < 0.8 else "high"
    return {
        "summary": {
            "physical_validity": overall_validity,
            "physical_score": overall_score,
            "monotonic_feature_count": len(monotonic_features),
            "monotonic_checked_values": overall_checks,
            "monotonic_violation_count": int(total_violations),
            "monotonic_violation_rate": overall_rate,
            "large_shift_rate": overall_large_shift_rate,
            "non_positive_risk_rate": overall_non_positive_risk_rate,
            "score_formula": "score = 1 - 0.60*monotonic_violation_rate - 0.25*large_shift_rate - 0.15*non_positive_risk_rate",
            "note": "当前版本综合检查强单调背景字段下降、生成变化幅度过大、以及 morph 后风险未提升。",
        },
        "field_semantics": list(semantics.values()),
        "patterns": pattern_rows,
        "violations": violation_rows,
    }


def physical_relation_contributions(start_windows, end_windows, morph_df, relation_specs, top_n=12):
    rows = []
    if not relation_specs:
        return pd.DataFrame()
    for hf_id in sorted(morph_df["hf_id"].unique()):
        idx = morph_df.index[morph_df["hf_id"] == hf_id].to_numpy()
        for spec in relation_specs:
            mean_deltas = []
            slope_deltas = []
            for i in idx:
                start_rel = relation_series(start_windows[i], spec)
                end_rel = relation_series(end_windows[i], spec)
                delta_rel = end_rel - start_rel
                summary = summarize_window(delta_rel)
                mean_deltas.append(float(np.mean(summary["mean"])))
                slope_deltas.append(float(np.mean(summary["slope"])))
            mean_delta = float(np.mean(mean_deltas))
            slope_delta = float(np.mean(slope_deltas))
            strength = float(np.sqrt(mean_delta ** 2 + slope_delta ** 2))
            rows.append({
                "hf_id": hf_id,
                "relation": spec["name"],
                "meaning": spec["meaning"],
                "relation_type": spec["type"],
                "left_feature": spec["left_feature"],
                "right_feature": spec["right_feature"],
                "mean_delta": mean_delta,
                "slope_delta": slope_delta,
                "abs_strength": strength,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["hf_id", "abs_strength"], ascending=[True, False]).reset_index(drop=True)


def group_delta_table(delta_rows, morph_df, groups):
    rows = []
    for hf_id in sorted(morph_df["hf_id"].unique()):
        idx = morph_df.index[morph_df["hf_id"] == hf_id].to_numpy()
        for group in sorted(groups):
            mean_key = f"group::{group}::mean_delta"
            slope_key = f"group::{group}::slope_delta"
            vol_key = f"group::{group}::volatility_delta"
            rows.append({
                "hf_id": hf_id,
                "group": group,
                "mean_delta": float(np.mean([delta_rows[i].get(mean_key, 0.0) for i in idx])),
                "slope_delta": float(np.mean([delta_rows[i].get(slope_key, 0.0) for i in idx])),
                "volatility_delta": float(np.mean([delta_rows[i].get(vol_key, 0.0) for i in idx])),
            })
    return pd.DataFrame(rows)



def hidden_feature_summary(
    morph_df,
    field_df,
    group_df,
    semantic_coupling_df=None,
):
    def relevant_rows(df, strength_col="abs_strength", relative_min=0.50, std_min=1.0, absolute_min=1e-8):
        if df is None or df.empty or strength_col not in df.columns:
            return pd.DataFrame()
        out = df.copy()
        if "relevance_level" in out.columns:
            out = out[out["relevance_level"].isin(["strong"])].copy()
            if out.empty:
                return out
        strengths = pd.to_numeric(out[strength_col], errors="coerce").fillna(0.0).abs()
        max_strength = float(strengths.max()) if len(strengths) else 0.0
        if max_strength <= 0.0:
            return out.iloc[0:0].copy()
        distribution_threshold = float(strengths.mean() + float(std_min) * strengths.std(ddof=0)) if len(strengths) > 1 else 0.0
        threshold = max(float(absolute_min), max_strength * float(relative_min), distribution_threshold)
        out = out.loc[strengths >= threshold].copy()
        if "risk_corr" in out.columns:
            out["_level_rank"] = out["relevance_level"].map({"strong": 0}).fillna(9)
            return out.sort_values(["_level_rank", "risk_corr", strength_col], ascending=[True, False, False]).drop(columns=["_level_rank"])
        return out.sort_values(strength_col, ascending=False)

    features = []
    for hf_id in sorted(morph_df["hf_id"].unique()):
        sub = morph_df[morph_df["hf_id"] == hf_id]
        field_sub = relevant_rows(field_df[field_df["hf_id"] == hf_id]) if not field_df.empty else pd.DataFrame()
        coupling_sub = relevant_rows(
            semantic_coupling_df[semantic_coupling_df["hf_id"] == hf_id],
            relative_min=0.0,
            std_min=0.25,
        ) if semantic_coupling_df is not None and not semantic_coupling_df.empty else pd.DataFrame()
        group_sub = group_df[group_df["hf_id"] == hf_id].copy() if not group_df.empty else pd.DataFrame()
        active_groups = []
        if not group_sub.empty:
            group_sub["strength"] = np.sqrt(group_sub["mean_delta"] ** 2 + group_sub["slope_delta"] ** 2)
            active_groups = group_sub.sort_values("strength", ascending=False).head(5)["group"].tolist()
        features.append({
            "hf_id": hf_id,
            "n_windows": int(len(sub)),
            "mean_selected_step": float(sub["selected_step"].mean()),
            "target_reached_rate": float(sub["target_reached"].mean()),
            "mean_logit_start": float(sub["logit_start"].mean()),
            "mean_logit_final": float(sub["logit_final"].mean()),
            "mean_logit_delta": float(sub["logit_delta"].mean()),
            "mean_last_logit_final": float(sub["last_logit_final"].mean()),
            "mean_last_logit_delta": float(sub["last_logit_delta"].mean()),
            "mean_pred_start": float(sub["pred_start"].mean()),
            "mean_pred_final": float(sub["pred_final"].mean()),
            "mean_risk_delta": float(sub["risk_delta"].mean()),
            "mean_last_pred_final": float(sub["last_pred_final"].mean()),
            "mean_last_risk_delta": float(sub["last_risk_delta"].mean()),
            "mean_hf_prob": float(sub["hf_prob"].mean()),
            "mean_original_to_start_mse": float(sub["original_to_start_mse"].mean()),
            "mean_start_to_final_mse": float(sub["start_to_final_mse"].mean()),
            "active_groups": active_groups,
            "top_fields": field_sub[[
                "feature",
                "group",
                "mean_delta",
                "slope_delta",
                "volatility_delta",
                "range_delta",
                "abs_strength",
                "risk_corr",
                "relevance_level",
            ]].to_dict("records"),
            "top_semantic_couplings": coupling_sub[[
                "coupling",
                "category",
                "meaning",
                "features",
                "output_unit",
                "mean_delta",
                "slope_delta",
                "range_delta",
                "abs_strength",
                "risk_corr",
                "relevance_level",
            ]].to_dict("records") if not coupling_sub.empty else [],
            "score_definition": "cluster_similarity_plus_risk_gain",
            "probability_definition": "sigmoid(hf_score)",
        })
    return features


def describe_direction(value, pos="上升", neg="下降"):
    if value > 0.05:
        return pos
    if value < -0.05:
        return neg
    return "变化较小"


def is_temperature_feature(compact):
    return any(token in compact for token in ("温度", "壳温"))


def change_flags(mean_delta, slope_delta, volatility_delta, range_delta, threshold):
    level_like = mean_delta if abs(mean_delta) >= abs(slope_delta) else slope_delta
    return {
        "up": level_like >= threshold,
        "down": level_like <= -threshold,
        "unstable": volatility_delta >= threshold or range_delta >= threshold,
        "stable": volatility_delta <= -threshold or range_delta <= -threshold,
    }


def feature_risk_hint(row, threshold=0.0):
    feature = str(row["feature"])
    group = str(row.get("group", ""))
    mean_delta = float(row.get("mean_delta", 0.0))
    slope_delta = float(row.get("slope_delta", 0.0))
    volatility_delta = float(row.get("volatility_delta", 0.0))
    range_delta = float(row.get("range_delta", 0.0))
    abs_mean = abs(mean_delta)
    abs_slope = abs(slope_delta)
    abs_vol = abs(volatility_delta)
    abs_range = abs(range_delta)

    stress_words = []
    if abs_mean >= threshold:
        stress_words.append("整体水平" + ("升高" if mean_delta > 0 else "降低"))
    if abs_slope >= threshold:
        stress_words.append("随时间" + ("继续抬升" if slope_delta > 0 else "继续走低"))
    if abs_vol >= threshold:
        stress_words.append("波动" + ("增大" if volatility_delta > 0 else "减小"))
    if abs_range >= threshold:
        stress_words.append("窗口内变化范围" + ("扩大" if range_delta > 0 else "缩小"))
    if not stress_words:
        stress_text = "变化幅度较小"
    else:
        stress_text = "、".join(stress_words)

    compact = feature.replace(" ", "")
    flags = change_flags(mean_delta, slope_delta, volatility_delta, range_delta, threshold)
    if is_temperature_feature(compact):
        if flags["up"]:
            return f"{stress_text}，可能提示热负荷上升、散热余量下降或温控状态开始不稳定。"
        if flags["down"]:
            return f"{stress_text}，可能提示温度工作点下移或温控状态发生偏移，需要结合制冷电流、TEC电压和光功率变化确认。"
        if flags["unstable"]:
            return f"{stress_text}，可能提示温度稳定性变差或温控调节不稳定。"
        return f"{stress_text}，可能提示温度状态发生偏移，需要结合光功率和制冷相关指标确认。"
    if "光功率" in compact:
        if flags["down"]:
            return f"{stress_text}，可能提示光路输出能力下降、链路衰减增大或放大器工作点偏离。"
        if flags["up"]:
            return f"{stress_text}，可能提示输出光功率工作点上移，需要结合输入光功率、增益和驱动电流判断是否属于异常补偿或链路状态改变。"
        if flags["unstable"]:
            return f"{stress_text}，可能提示光功率稳定性下降，存在输出抖动或调节不稳的风险。"
        return f"{stress_text}，说明光功率状态发生偏移，建议结合输入输出功率和增益变化复核。"
    if "增益" in compact:
        if flags["down"]:
            return f"{stress_text}，可能提示增益能力下降或增益控制余量变小。"
        if flags["up"]:
            return f"{stress_text}，可能提示增益工作点上移或控制补偿增强，需要结合输入/输出光功率判断是否存在异常放大状态。"
        if flags["unstable"]:
            return f"{stress_text}，可能提示增益控制稳定性变差。"
        return f"{stress_text}，说明增益相关状态发生偏移，需要结合业务阈值和真实故障样本复核。"
    if "驱动电流" in compact:
        if flags["up"]:
            return f"{stress_text}，可能提示激光器或驱动链路需要更高驱动才能维持工作状态，存在器件老化或效率下降风险。"
        if flags["down"]:
            return f"{stress_text}，可能提示驱动工作点下移或控制状态发生偏移，需要结合输出光功率判断是否存在输出能力下降。"
        if flags["unstable"]:
            return f"{stress_text}，可能提示驱动控制不稳定或工作点波动，需要结合背光电流和输出光功率复核。"
        return f"{stress_text}，可能提示驱动工作点发生偏移，需要结合光功率变化判断是否有效输出下降。"
    if "制冷电流" in compact or "TEC" in compact:
        if flags["up"]:
            return f"{stress_text}，可能提示温控系统负担加重，制冷余量下降或热状态趋于紧张。"
        if flags["down"]:
            return f"{stress_text}，可能提示温控执行量下降或温控工作点偏移，需要结合温度变化判断是否属于制冷需求下降还是控制异常。"
        if flags["unstable"]:
            return f"{stress_text}，可能提示温控调节不稳定。"
        return f"{stress_text}，说明温控执行量在风险升高方向上发生偏移。"
    if "背光电流" in compact:
        if flags["up"]:
            return f"{stress_text}，可能提示激光器监测反馈增强，需要结合驱动电流和输出光功率判断效率或反馈链路是否异常。"
        if flags["down"]:
            return f"{stress_text}，可能提示激光器监测反馈减弱，需要结合驱动电流和输出光功率判断是否存在输出效率下降。"
        if flags["unstable"]:
            return f"{stress_text}，可能提示激光器反馈监测不稳定，需要结合双通道差异和输出光功率复核。"
        return f"{stress_text}，可能提示激光器监测反馈状态变化，需要结合驱动电流和输出光功率判断效率是否异常。"
    if "电源" in compact or "3V3" in compact:
        if flags["unstable"]:
            return f"{stress_text}，可能提示供电稳定性下降，对光功率或控制环路产生影响。"
        if flags["down"]:
            return f"{stress_text}，可能提示供电电压下移或电源裕量下降，需要关注供电稳定性。"
        if flags["up"]:
            return f"{stress_text}，可能提示供电工作点上移，需要结合电源规格和其它控制量判断是否异常。"
        return f"{stress_text}，说明供电相关状态发生偏移，需要关注电源裕量。"
    if "运行时间" in compact or "上电" in compact or "累计" in compact or "超温" in compact:
        return f"{stress_text}，更像寿命或老化背景信息，提示该模式可能出现在长期运行或历史负荷较高的设备上。"
    return f"{stress_text}，说明该指标在模型风险升高方向上发生稳定变化，建议结合业务阈值和真实故障样本复核。"


def adaptive_change_threshold(top_fields):
    if top_fields.empty:
        return 0.0
    keys = ("mean_delta", "slope_delta", "volatility_delta", "range_delta")
    values = []
    for _, row in top_fields.iterrows():
        for key in keys:
            value = float(row.get(key, 0.0))
            if np.isfinite(value) and value != 0.0:
                values.append(abs(value))
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    positive = arr[arr > 0]
    if len(positive) == 0:
        return 0.0
    return float(max(np.quantile(positive, 0.60), positive.max() * 0.15))


def format_field_change(row, threshold=0.0):
    changes = [
        ("整体水平", float(row.get("mean_delta", 0.0)), "升高", "降低"),
        ("斜率", float(row.get("slope_delta", 0.0)), "增长斜率变大", "下降斜率变大"),
        ("波动", float(row.get("volatility_delta", 0.0)), "增强", "减弱"),
        ("范围", float(row.get("range_delta", 0.0)), "扩大", "缩小"),
    ]
    active = [item for item in changes if abs(item[1]) >= threshold]
    if not active:
        return "变化不明显"

    active.sort(key=lambda item: abs(item[1]), reverse=True)
    phrases = []
    for name, value, pos_text, neg_text in active[:3]:
        if name == "斜率":
            phrases.append(pos_text if value > 0 else neg_text)
        else:
            phrases.append(f"{name}{pos_text if value > 0 else neg_text}")
    return "，".join(phrases)


def has_meaningful_field_change(row, threshold=0.0):
    return any(
        abs(float(row.get(key, 0.0))) >= threshold
        for key in ("mean_delta", "slope_delta", "volatility_delta", "range_delta")
    )


def fmt_metric(value, digits=4):
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 10 ** (-digits):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def strip_report_unit(name):
    return re.sub(r"[\(\uFF08][^\)\uFF09]*[\)\uFF09]", "", str(name))


def display_signal_name(name, strip_units=True):
    text = str(name or "").strip()
    if not text:
        return "unknown"
    text = text.replace("field::", "").replace("coupling::", "")
    if strip_units:
        text = strip_report_unit(text)

    if "__minus__" in text:
        left, right = text.split("__minus__", 1)
        stat = ""
        if "__" in right:
            right, stat = right.rsplit("__", 1)
        parts = [f"{display_signal_name(left, strip_units=strip_units)} - {display_signal_name(right, strip_units=strip_units)}"]
        if stat:
            parts.append(stat.upper())
        return " ".join(parts)

    base, stat = text, ""
    if "__" in text:
        base, stat = text.rsplit("__", 1)
        stat = stat.upper() if stat else ""

    for component in ("LSR1", "LSR2"):
        prefix = f"{component}_"
        if base.startswith(prefix):
            key = base[len(prefix):]
            label = COUPLING_NAME_TRANSLATIONS.get(key)
            if label:
                return f"{component}{label}{stat}"

    for stage in ("PA", "BA"):
        prefix = f"{stage}_"
        if base.startswith(prefix):
            key = base[len(prefix):]
            label = COUPLING_NAME_TRANSLATIONS.get(key)
            if label:
                return f"{stage}{label}{stat}"

    label = COUPLING_NAME_TRANSLATIONS.get(base)
    if label:
        return f"{label}{stat}"

    return text


def display_feature_list(features):
    if isinstance(features, (list, tuple)):
        parts = features
    else:
        parts = re.split(r"\s*\|\s*|\s*、\s*", str(features or ""))
    return " | ".join(display_signal_name(part, strip_units=False) for part in parts if str(part).strip())


def display_coupling_category(category):
    text = str(category or "").strip()
    return COUPLING_CATEGORY_TRANSLATIONS.get(text, text or "unknown")


def core_driver_fields(top_fields):
    if top_fields.empty or "feature" not in top_fields.columns:
        return top_fields
    keep = [
        bool(field_semantics(feature).get("allow_as_core_driver", True))
        for feature in top_fields["feature"].astype(str)
    ]
    return top_fields.loc[keep].copy()


def risk_evidence_rows(item, top_fields, pair_sub=None, multi_sub=None, rel_sub=None):
    rows = []
    risk_gain = max(float(item.get("mean_logit_delta", 0.0)), 0.0)
    coverage_weight = math.sqrt(max(float(item.get("n_windows", 1)), 1.0))
    hf_prob = float(item.get("mean_hf_prob", 0.5))

    for _, row in core_driver_fields(top_fields).iterrows():
        strength = float(row["abs_strength"])
        score = strength * (1.0 + min(risk_gain, 5.0) / 5.0) * (0.35 + hf_prob) / max(coverage_weight / 10.0, 1.0)
        direction = describe_direction(row["mean_delta"])
        slope_direction = describe_direction(row["slope_delta"], pos="趋势增强", neg="趋势减弱")
        volatility_direction = describe_direction(row.get("volatility_delta", 0.0), pos="波动增强", neg="波动减弱")
        range_direction = describe_direction(row.get("range_delta", 0.0), pos="范围扩大", neg="范围缩小")
        rows.append({
            "type": "单字段变化",
            "target": str(row["feature"]),
            "detail": f"字段组: {row['group']}",
            "change": f"水平{direction}，{slope_direction}，{volatility_direction}，{range_direction}",
            "evidence": (
                f"综合变化强度 {row['abs_strength']:.4f}; "
                f"风险相关性 {fmt_metric(row.get('risk_corr', np.nan))}; "
                f"相关等级 {row.get('relevance_level', 'unknown')}; "
                f"水平变化 {fmt_metric(row['mean_delta'])}; "
                f"趋势变化 {fmt_metric(row['slope_delta'])}; "
                f"波动变化 {fmt_metric(row.get('volatility_delta', 0.0))}; "
                f"范围变化 {fmt_metric(row.get('range_delta', 0.0))}"
            ),
            "score": float(score),
            "risk_corr": float(row.get("risk_corr", np.nan)),
            "relevance_level": str(row.get("relevance_level", "unknown")),
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def filter_relevant_evidence(rows, score_key="score", relative_min=0.50, std_min=1.0, absolute_min=1e-8):
    rows = list(rows or [])
    if not rows:
        return []
    rows = [row for row in rows if row.get("relevance_level") in {"strong"}]
    if not rows:
        return []
    scores = [abs(float(row.get(score_key, 0.0))) for row in rows]
    max_score = max(scores)
    if max_score <= 0.0:
        return []
    distribution_threshold = float(np.mean(scores) + float(std_min) * np.std(scores)) if len(scores) > 1 else 0.0
    threshold = max(float(absolute_min), max_score * float(relative_min), distribution_threshold)
    return [
        row for row in rows
        if abs(float(row.get(score_key, 0.0))) >= threshold
    ]


def filter_relevant_couplings(couplings, relative_min=0.0, std_min=0.25, absolute_min=1e-8):
    couplings = list(couplings or [])
    if not couplings:
        return []
    couplings = [row for row in couplings if row.get("relevance_level") in {"strong"}]
    if not couplings:
        return []
    strengths = [abs(float(row.get("abs_strength", 0.0))) for row in couplings]
    max_strength = max(strengths)
    if max_strength <= 0.0:
        return []
    distribution_threshold = float(np.mean(strengths) + float(std_min) * np.std(strengths)) if len(strengths) > 1 else 0.0
    threshold = max(float(absolute_min), max_strength * float(relative_min), distribution_threshold)
    return [
        row for row in couplings
        if abs(float(row.get("abs_strength", 0.0))) >= threshold
    ]


def build_user_friendly_conclusion(item, top_fields, risk_rows=None):
    if risk_rows:
        names = "、".join(str(row["target"]) for row in risk_rows[:3])
    else:
        top_fields = core_driver_fields(top_fields)
        if top_fields.empty:
            return "该模式暂时没有稳定的强相关单指标变化，建议增加 morph 窗口数后重新生成报告。"
        names = "、".join(top_fields["feature"].head(3).astype(str).tolist())
    if not names:
        return "该模式暂时没有稳定的单指标变化，建议增加 morph 窗口数后重新生成报告。"
    return (
        f"该模式主要由 {names} 等指标变化构成。"
        "这些变化不是单次越限告警，而是在模型认为风险升高的过程中持续出现的候选前兆。"
        "建议回到真实高风险/故障样本中复核，并观察同一 sn 的后续窗口是否反复出现。"
    )


def dominant_title_direction(row):
    mean_delta = float(row.get("mean_delta", 0.0))
    slope_delta = float(row.get("slope_delta", 0.0))
    volatility_delta = float(row.get("volatility_delta", 0.0))
    range_delta = float(row.get("range_delta", 0.0))
    level_like = mean_delta if abs(mean_delta) >= abs(slope_delta) else slope_delta
    fluctuation = max(volatility_delta, range_delta)
    if abs(level_like) >= max(abs(fluctuation), 1e-12):
        if level_like > 0:
            return "up"
        if level_like < 0:
            return "down"
    if fluctuation > 0:
        return "unstable"
    return "shift"


def pattern_title(top_fields, pattern_no):
    top_fields = core_driver_fields(top_fields)
    if top_fields.empty:
        return f"模式 {pattern_no}: 风险变化待确认"
    top = top_fields.iloc[0]
    feature = str(top["feature"])
    compact = feature.replace(" ", "")
    direction = dominant_title_direction(top)

    if is_temperature_feature(compact):
        if direction == "up":
            name = "温度升高 / 热负荷加重"
        elif direction == "down":
            name = "温度降低 / 热状态偏移"
        elif direction == "unstable":
            name = "温度波动增大 / 温控不稳定"
        else:
            name = "温度状态偏移"
    elif "光功率" in compact:
        if direction == "down":
            name = "光功率下降 / 输出能力减弱"
        elif direction == "up":
            name = "光功率升高 / 工作点上移"
        elif direction == "unstable":
            name = "光功率波动增大 / 输出不稳定"
        else:
            name = "光功率状态偏移"
    elif "增益" in compact:
        if direction == "down":
            name = "增益下降 / 放大能力减弱"
        elif direction == "up":
            name = "增益升高 / 控制补偿增强"
        elif direction == "unstable":
            name = "增益波动增大 / 控制不稳定"
        else:
            name = "增益状态偏移"
    elif "驱动电流" in compact:
        if direction == "up":
            name = "驱动电流升高 / 器件补偿加重"
        elif direction == "down":
            name = "驱动电流降低 / 工作点偏移"
        elif direction == "unstable":
            name = "驱动电流波动增大 / 控制不稳定"
        else:
            name = "驱动工作点偏移"
    elif "制冷电流" in compact or "TEC" in compact:
        if direction == "up":
            name = "温控负担加重"
        elif direction == "down":
            name = "温控执行量降低 / 工作点偏移"
        elif direction == "unstable":
            name = "温控波动增大 / 调节不稳定"
        else:
            name = "温控执行量偏移"
    elif "背光电流" in compact:
        if direction == "up":
            name = "背光电流升高 / 反馈状态偏移"
        elif direction == "down":
            name = "背光电流降低 / 反馈状态偏移"
        elif direction == "unstable":
            name = "背光电流波动增大 / 反馈不稳定"
        else:
            name = "激光器反馈状态变化"
    elif "电源" in compact or "3V3" in compact:
        if direction == "down":
            name = "供电电压降低 / 电源裕量偏移"
        elif direction == "up":
            name = "供电电压升高 / 工作点偏移"
        elif direction == "unstable":
            name = "供电状态波动"
        else:
            name = "供电状态偏移"
    elif "运行时间" in compact or "上电" in compact or "累计" in compact or "超温" in compact:
        name = "长期运行 / 老化背景"
    else:
        name = "关键指标状态偏移"
    return f"模式 {pattern_no}: {name}"


def pattern_kind(top_fields):
    top_fields = core_driver_fields(top_fields)
    if top_fields.empty:
        return "风险变化待确认"
    title = pattern_title(top_fields, 0)
    return title.split(": ", 1)[1] if ": " in title else title


def pattern_dedupe_key(top_fields):
    top_fields = core_driver_fields(top_fields)
    kind = pattern_kind(top_fields)
    if top_fields.empty:
        return (kind,)
    core_features = tuple(top_fields["feature"].head(2).astype(str).tolist())
    return (kind,) + core_features


def hf_relevance_score(item):
    rows = list(item.get("top_fields", [])) + list(item.get("top_semantic_couplings", []))
    values = [
        abs(float(row.get("risk_corr", 0.0)))
        for row in rows
        if row.get("relevance_level") in {"strong"} and np.isfinite(float(row.get("risk_corr", 0.0)))
    ]
    return float(max(values)) if values else 0.0


def has_core_evidence(item):
    return hf_relevance_score(item) > 0.60


def select_report_features(features, field_df=None):
    selected = [item for item in features if has_core_evidence(item)]
    return sorted(
        selected,
        key=lambda item: (
            hf_relevance_score(item),
            float(item.get("mean_risk_delta", 0.0)),
            int(item.get("n_windows", 0)),
        ),
        reverse=True,
    )


def build_evidence_payload(run_dir, features, field_df, morph_df, feature_names, physical_validation):
    selected = select_report_features(features, field_df)
    selected_ids = {item["hf_id"] for item in selected}
    validation_by_hf = {
        row["hf_id"]: row
        for row in (physical_validation or {}).get("patterns", [])
    }
    patterns = []
    for item in sorted(features, key=lambda row: int(row.get("n_windows", 0)), reverse=True):
        hf_id = item["hf_id"]
        top_fields = field_df[field_df["hf_id"] == hf_id] if not field_df.empty else pd.DataFrame()
        change_threshold = adaptive_change_threshold(top_fields)
        risk_rows = filter_relevant_evidence(risk_evidence_rows(item, top_fields))
        core_features = []
        for row in risk_rows:
            feature_row = top_fields[top_fields["feature"] == row["target"]]
            if feature_row.empty:
                continue
            feature_row = feature_row.iloc[0]
            core_features.append({
                "name": str(row["target"]),
                "group": str(feature_row.get("group", "")),
                "semantics": field_semantics(row["target"]),
                "change_summary": format_field_change(feature_row, change_threshold),
                "dominant_direction": dominant_title_direction(feature_row),
                "mean_delta": float(feature_row.get("mean_delta", 0.0)),
                "slope_delta": float(feature_row.get("slope_delta", 0.0)),
                "volatility_delta": float(feature_row.get("volatility_delta", 0.0)),
                "range_delta": float(feature_row.get("range_delta", 0.0)),
                "risk_corr": float(feature_row.get("risk_corr", np.nan)),
                "relevance_level": str(feature_row.get("relevance_level", "unknown")),
                "ranking_score": float(row["score"]),
            })
        patterns.append({
            "hf_id": hf_id,
            "report_selected": hf_id in selected_ids,
            "title": pattern_title(top_fields, 0).split(": ", 1)[1],
            "coverage_windows": int(item["n_windows"]),
            "mean_selected_step": float(item.get("mean_selected_step", 0.0)),
            "target_reached_rate": float(item.get("target_reached_rate", 0.0)),
            "mean_logit_delta": float(item.get("mean_logit_delta", 0.0)),
            "mean_risk_delta": float(item.get("mean_risk_delta", 0.0)),
            "physical_validation": validation_by_hf.get(hf_id, {}),
            "core_features": core_features,
        })

    return {
        "run_dir": str(run_dir),
        "morph": {
            "method": str(morph_df["morph_method"].iloc[0]) if "morph_method" in morph_df.columns and len(morph_df) else "NA",
            "selection_mode": str(morph_df["selection_mode"].iloc[0]) if "selection_mode" in morph_df.columns and len(morph_df) else "NA",
            "window_len": int(morph_df["window_len"].iloc[0]) if "window_len" in morph_df.columns and len(morph_df) else None,
            "target_logit_delta": float(morph_df["target_logit_delta"].iloc[0]) if "target_logit_delta" in morph_df.columns and len(morph_df) else None,
            "target_pred_score": str(morph_df["target_pred_score"].iloc[0]) if "target_pred_score" in morph_df.columns and len(morph_df) else "None",
        },
        "field_semantics": [field_semantics(name) for name in feature_names],
        "physical_validation_summary": (physical_validation or {}).get("summary", {}),
        "patterns": patterns,
    }


def window_scope_text(morph_df):
    if "window_len" in morph_df.columns and len(morph_df):
        window_len = int(pd.to_numeric(morph_df["window_len"], errors="coerce").dropna().iloc[0])
    else:
        window_len = None
    point_text = f"{window_len} 个连续时间点" if window_len else "固定长度的连续时间点"
    return f"每个窗口包含 {point_text}。"


def format_semantic_coupling_change(row):
    mean_delta = float(row.get("mean_delta", 0.0))
    slope_delta = float(row.get("slope_delta", 0.0))
    range_delta = float(row.get("range_delta", 0.0))
    eps = 1e-12
    phrases = []

    if abs(mean_delta) > eps:
        phrases.append("整体上升" if mean_delta > 0 else "整体下降")
    if abs(slope_delta) > eps:
        phrases.append("趋势增强" if slope_delta > 0 else "趋势减弱")
    if abs(range_delta) > eps:
        phrases.append("范围扩大" if range_delta > 0 else "范围缩小")

    return "，".join(phrases[:3]) if phrases else "变化不明显"


def cross_pattern_common_findings(report_features):
    field_counts = {}
    coupling_counts = {}
    for item in report_features:
        hf_id = item.get("hf_id")
        for row in item.get("top_fields", []):
            if row.get("relevance_level") not in ("strong",):
                continue
            key = str(row.get("feature", "")).strip()
            if not key:
                continue
            entry = field_counts.setdefault(key, {"ids": set(), "max_corr": 0.0, "level": "strong"})
            entry["ids"].add(hf_id)
            corr = abs(float(row.get("risk_corr", 0.0) or 0.0))
            entry["max_corr"] = max(entry["max_corr"], corr)
            if row.get("relevance_level") == "strong":
                entry["level"] = "strong"
        for row in item.get("top_semantic_couplings", []):
            if row.get("relevance_level") not in ("strong",):
                continue
            key = display_signal_name(row.get("coupling", ""))
            if not key:
                continue
            entry = coupling_counts.setdefault(key, {
                "ids": set(),
                "max_corr": 0.0,
                "level": "strong",
                "meaning": str(row.get("meaning", "")),
            })
            entry["ids"].add(hf_id)
            corr = abs(float(row.get("risk_corr", 0.0) or 0.0))
            entry["max_corr"] = max(entry["max_corr"], corr)
            if row.get("relevance_level") == "strong":
                entry["level"] = "strong"

    field_items = [
        (name, info)
        for name, info in field_counts.items()
        if len(info["ids"]) >= 2
    ]
    coupling_items = [
        (name, info)
        for name, info in coupling_counts.items()
        if len(info["ids"]) >= 2
    ]
    field_items.sort(key=lambda x: (len(x[1]["ids"]), x[1]["max_corr"]), reverse=True)
    coupling_items.sort(key=lambda x: (len(x[1]["ids"]), x[1]["max_corr"]), reverse=True)
    return field_items[:5], coupling_items[:5]


def make_report(
    run_dir,
    features,
    field_df,
    last_field_df,
    physical_df,
    last_physical_df,
    morph_df,
    feature_names=None,
    groups=None,
    physical_validation=None,
):
    total_morph_windows = len(morph_df)
    scope_text = window_scope_text(morph_df)
    report_features = select_report_features(features, field_df)
    skipped_features = [item for item in features if not has_core_evidence(item)]
    validation_by_hf = {
        row["hf_id"]: row
        for row in (physical_validation or {}).get("patterns", [])
    }
    physical_summary = (physical_validation or {}).get("summary", {})
    physical_validity_text = {"high": "高", "medium": "中", "low": "低"}.get(
        physical_summary.get("physical_validity", "unknown"),
        "未知",
    )
    field_common, coupling_common = cross_pattern_common_findings(report_features)
    lines = [
        "# 光放大器故障特征挖掘报告",
        "",
        "## 1. 分析概况",
        f"- 分析窗口数量: {total_morph_windows}",
        f"- 时间窗口: {scope_text}",
        f"- 反事实生成方式: 从低/中风险窗口向高风险方向生成",
        f"- 物理可信度: {physical_validity_text}",
        f"- 有效候选模式数量: {len(report_features)}",
        "",
        "## 2. 总体结论",
        f"- 本次分析展示 {len(report_features)} 类存在强相关证据的候选前兆模式。",
        f"- 另有 {len(skipped_features)} 类内部候选模式暂未发现强相关核心证据，未在正文展开。",
        "- 当前版本同时关注原始字段变化和经代码校验计算的语义耦合关系。",
        "- 报告中的候选前兆表示模型在风险升高方向上发现的稳定变化，不等同于最终故障原因或告警规则。",
        "",
    ]
    if not report_features:
        lines.extend([
            "本次分析未发现具备强相关证据的候选前兆模式。主要原因可能是 morph 后风险提升幅度较小，或字段/耦合变化与风险提升之间没有形成稳定关系。",
            "",
            "建议使用更多真实故障样本重新分析，或扩大可用窗口数量后复核。",
            "",
        ])

    lines.extend([
        "## 3. 候选前兆模式",
        "",
    ])
    for pattern_no, item in enumerate(report_features, start=1):
        hf_id = item["hf_id"]
        top_fields = field_df[field_df["hf_id"] == hf_id] if not field_df.empty else pd.DataFrame()
        change_threshold = adaptive_change_threshold(top_fields)
        risk_rows = filter_relevant_evidence(risk_evidence_rows(item, top_fields))
        title = pattern_title(top_fields, pattern_no)
        validation = validation_by_hf.get(hf_id)
        lines.extend([
            f"### {title}",
            f"- 内部追溯 ID: {hf_id}",
            f"- 出现范围: 本次分析的 {total_morph_windows} 个窗口中，有 {item['n_windows']} 个窗口属于该模式。",
            "- 风险变化: 模型判断该模式对应的故障风险升高。",
        ])
        if validation:
            validity_text = {"high": "高", "medium": "中", "low": "低"}.get(
                validation.get("physical_validity", "unknown"),
                "未知",
            )
            lines.append(f"- 生成可信度: {validity_text}")
            if validation.get("physical_validity") != "high" and validation.get("top_violations"):
                first = validation["top_violations"][0]
                lines.append(
                    f"- 可信度提示: {first['feature']} 在生成后出现不符合单调约束的下降，"
                    "该字段不应作为直接故障前兆解释。"
                )
        lines.extend([
            "",
            "#### 核心指标",
        ])
        if not risk_rows:
            lines.append("- 暂未发现稳定的单指标变化。")
        else:
            for row in risk_rows:
                feature_row = top_fields[top_fields["feature"] == row["target"]]
                if feature_row.empty:
                    continue
                feature_row = feature_row.iloc[0]
                lines.extend([
                    f"- {row['target']}",
                    f"  - 相关性: {feature_row.get('relevance_level', 'unknown')} ({fmt_metric(feature_row.get('risk_corr', np.nan))})",
                    f"  - 怎么变: {format_field_change(feature_row, change_threshold)}",
                    f"  - 可能意味着: {feature_risk_hint(feature_row, change_threshold)}",
                ])

        semantic_couplings = item.get("top_semantic_couplings", [])
        lines.extend([
            "",
            "#### 核心耦合关系",
        ])
        if not semantic_couplings:
            lines.append("- 暂未发现稳定的语义耦合变化。")
        else:
            for coupling in semantic_couplings:
                lines.extend([
                    f"- {display_signal_name(coupling.get('coupling', 'unknown'))}",
                    f"  - 类型: {display_coupling_category(coupling.get('category', 'unknown'))}",
                    f"  - 相关性: {coupling.get('relevance_level', 'unknown')} ({fmt_metric(coupling.get('risk_corr', np.nan))})",
                    f"  - 涉及字段: {display_feature_list(coupling.get('features', ''))}",
                    f"  - 怎么变: {format_semantic_coupling_change(coupling)}",
                    f"  - 含义: {coupling.get('meaning', '')}",
                ])

        lines.extend([
            "",
            "#### 工程解读",
            build_user_friendly_conclusion(item, top_fields, risk_rows=risk_rows),
            "",
        ])

    lines.extend([
        "## 4. 跨模式共性发现",
    ])
    if not field_common and not coupling_common:
        lines.append("- 本次未发现跨多个候选模式反复出现的强相关共性证据。")
    if field_common:
        lines.append("- 反复出现的原始字段信号:")
        for name, info in field_common:
            lines.append(
                f"  - {name}: 出现在 {len(info['ids'])} 个候选模式中，最高相关性 {info['level']} ({info['max_corr']:.2f})。"
            )
    if coupling_common:
        lines.append("- 反复出现的耦合关系信号:")
        for name, info in coupling_common:
            meaning = f"，含义: {info['meaning']}" if info.get("meaning") else ""
            lines.append(
                f"  - {name}: 出现在 {len(info['ids'])} 个候选模式中，最高相关性 {info['level']} ({info['max_corr']:.2f}){meaning}。"
            )
    lines.extend([
        "",
        "## 5. 未展开模式说明",
    ])
    if skipped_features:
        for item in skipped_features:
            lines.append(f"- 内部追溯 ID {item.get('hf_id')}: 暂未发现强相关字段或语义耦合证据。")
    else:
        lines.append("- 无。")
    lines.extend([
        "",
        "## 6. 建议",
        "- 优先在真实故障样本中复核上述候选模式和跨模式共性信号。",
        "- 关注同一 sn 在连续窗口中是否反复出现同类字段变化或耦合关系变化。",
        "- 本报告用于发现候选故障前兆，不是最终告警规则；暂不建议直接作为硬告警规则。",
        "- 如果某个信号在多个模式中反复出现，且在真实故障前也持续增强，可以再把它固化为明确规则或监控指标。",
        "",
    ])
    report = "\n".join(lines)
    path = Path(run_dir) / "tool_generated_report.md"
    path.write_text(report, encoding="utf-8")
    return path, report


def discover_hidden_features(run_dir, args):
    run_dir = Path(run_dir)
    data_dir = run_dir / "processed_data"
    predictor_path = run_dir / "predictor" / "model.best.pt"
    vae_path = run_dir / "generator" / "vae.pt"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing processed data directory: {data_dir}")
    if not predictor_path.exists():
        raise FileNotFoundError(f"Missing predictor checkpoint: {predictor_path}")
    if not vae_path.exists():
        raise FileNotFoundError(f"Missing generator checkpoint: {vae_path}")

    output_dir = run_dir / "hidden_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[hidden] Using device={device}")

    preprocess_summary = load_preprocess_summary(data_dir)
    feature_names = preprocess_summary.get("feature_cols", [])
    groups = build_feature_groups(feature_names)
    physical_relations = build_physical_relations(feature_names)
    print(f"[hidden] Loaded {len(feature_names)} features in {len(groups)} groups.")
    print(f"[hidden] Built {len(physical_relations)} physical relation templates.")

    predictor, pred_ckpt = load_predictor(predictor_path, device)
    vae, vae_ckpt = load_vae(vae_path, device)
    if pred_ckpt["n_features"] != vae_ckpt["n_features"] or pred_ckpt["window_len"] != vae_ckpt["window_len"]:
        raise ValueError("Predictor and generator input dimensions do not match.")

    dataset = WindowDataset(data_dir, split=args.split)
    pred_filename = {
        "train": "train_predictions.csv",
        "val": "validation_predictions.csv",
        "test": "test_predictions.csv",
    }.get(args.split, f"{args.split}_predictions.csv")
    pred_path = run_dir / "predictor" / pred_filename
    pred_df = pd.read_csv(pred_path) if pred_path.exists() else None
    local_indices = choose_indices(dataset, pred_df, args.max_windows, args.seed, selection_mode=args.selection_mode)
    print(f"[hidden] Selected {len(local_indices)} {args.split} windows for morphing.")

    try:
        baseline_dataset = WindowDataset(data_dir, split="train")
        baseline_pred_path = run_dir / "predictor" / "train_predictions.csv"
        baseline_pred_df = pd.read_csv(baseline_pred_path) if baseline_pred_path.exists() else None
    except ValueError:
        baseline_dataset = dataset
        baseline_pred_df = pred_df
    risk_direction = None
    if args.morph_method == "risk-centroid":
        direction_dataset = baseline_dataset
        direction_pred_df = baseline_pred_df
        risk_direction = estimate_risk_centroid_direction(
            dataset=direction_dataset,
            pred_df=direction_pred_df,
            vae=vae,
            device=device,
            max_windows=args.max_direction_windows,
            seed=args.seed,
        )

    morph_df, delta_rows, last_delta_rows, start_windows, final_windows, last_windows = collect_morph_samples(
        dataset=dataset,
        local_indices=local_indices,
        predictor=predictor,
        vae=vae,
        feature_names=feature_names,
        groups=groups,
        args=args,
        device=device,
        risk_direction=risk_direction,
    )
    morph_df, delta_matrix, delta_keys, labels = cluster_hidden_features(
        delta_rows=delta_rows,
        morph_df=morph_df,
        n_clusters=args.n_hf,
        seed=args.seed,
    )
    print(f"[hidden] Clustered morph changes into {morph_df['hf_id'].nunique()} HF candidates.")

    field_df = top_field_contributions(delta_rows, morph_df, feature_names, top_n=args.top_fields)
    last_field_df = top_field_contributions(last_delta_rows, morph_df, feature_names, top_n=args.top_fields)
    physical_df = physical_relation_contributions(start_windows, final_windows, morph_df, physical_relations, top_n=args.top_fields)
    last_physical_df = physical_relation_contributions(start_windows, last_windows, morph_df, physical_relations, top_n=args.top_fields)
    group_df = group_delta_table(delta_rows, morph_df, groups)
    semantic_coupling_df, semantic_coupling_specs = semantic_coupling_contributions(
        start_windows=start_windows,
        end_windows=final_windows,
        morph_df=morph_df,
        feature_names=feature_names,
        preprocess_summary=preprocess_summary,
        top_n=args.top_relations,
        semantic_couplings_path=getattr(args, "semantic_couplings", None),
    )
    print(f"[hidden] Built {len(semantic_coupling_specs)} semantic coupling templates.")
    features = hidden_feature_summary(
        morph_df,
        field_df,
        group_df,
        semantic_coupling_df=semantic_coupling_df,
    )
    physical_validation = validate_physical_constraints(delta_rows, morph_df, feature_names)
    print(
        "[hidden] Physical validation: "
        f"validity={physical_validation['summary']['physical_validity']}, "
        f"physical_score={physical_validation['summary'].get('physical_score', 0.0):.4f}, "
        f"monotonic_violation_rate={physical_validation['summary']['monotonic_violation_rate']:.4f}, "
        f"large_shift_rate={physical_validation['summary'].get('large_shift_rate', 0.0):.4f}, "
        f"non_positive_risk_rate={physical_validation['summary'].get('non_positive_risk_rate', 0.0):.4f}"
    )

    morph_df.to_csv(output_dir / "morph_windows.csv", index=False, encoding="utf-8-sig")
    field_df.to_csv(output_dir / "field_contributions.csv", index=False, encoding="utf-8-sig")
    last_field_df.to_csv(output_dir / "last_step_field_contributions.csv", index=False, encoding="utf-8-sig")
    physical_df.to_csv(output_dir / "physical_relation_contributions.csv", index=False, encoding="utf-8-sig")
    last_physical_df.to_csv(output_dir / "last_step_physical_relation_contributions.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(output_dir / "group_deltas.csv", index=False, encoding="utf-8-sig")
    semantic_coupling_df.to_csv(output_dir / "semantic_couplings.csv", index=False, encoding="utf-8-sig")
    representative = (
        morph_df.sort_values(["hf_id", "cluster_distance"])
        .groupby("hf_id", as_index=False)
        .head(args.representatives)
    )
    representative.to_csv(output_dir / "representative_windows.csv", index=False, encoding="utf-8-sig")
    hidden_feature_payload = {
        "run_dir": str(run_dir),
        "method": f"vae_latent_{args.morph_method}_morphing",
        "selection_mode": args.selection_mode,
        "input_shape": [int(pred_ckpt["n_features"]), int(pred_ckpt["window_len"])],
        "groups": {k: [feature_names[i] for i in v] for k, v in groups.items()},
        "physical_relations": physical_relations,
        "semantic_coupling_templates": semantic_coupling_specs,
        "features": features,
    }
    with open(output_dir / "hidden_features.json", "w", encoding="utf-8") as f:
        json.dump(hidden_feature_payload, f, ensure_ascii=False, indent=2)
    with open(output_dir / "physical_validation.json", "w", encoding="utf-8") as f:
        json.dump(physical_validation, f, ensure_ascii=False, indent=2)
    evidence_payload = build_evidence_payload(
        run_dir=run_dir,
        features=features,
        field_df=field_df,
        morph_df=morph_df,
        feature_names=feature_names,
        physical_validation=physical_validation,
    )
    with open(output_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence_payload, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        output_dir / "morph_delta_matrix.npz",
        delta_matrix=delta_matrix,
        delta_keys=np.array(delta_keys, dtype=object),
        labels=np.asarray(labels, dtype=np.int32),
    )
    report_path, report_text = make_report(
        run_dir,
        features,
        field_df,
        last_field_df,
        physical_df,
        last_physical_df,
        morph_df,
        feature_names=feature_names,
        groups=groups,
        physical_validation=physical_validation,
    )
    print(f"[hidden] Saved report to: {report_path}")
    if getattr(args, "print_report", False):
        print("\n" + "=" * 80)
        print("HIDDEN FEATURE REPORT")
        print("=" * 80)
        print(report_text)
    if getattr(args, "print_json", False):
        print("\n" + "=" * 80)
        print("HIDDEN FEATURE JSON")
        print("=" * 80)
        print(json.dumps(hidden_feature_payload, ensure_ascii=False, indent=2))
    return {
        "output_dir": str(output_dir),
        "n_windows": int(len(morph_df)),
        "n_hidden_features": int(morph_df["hf_id"].nunique()),
        "report_path": str(report_path),
    }

