import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import WindowDataset
from .models import FaultTCN


def binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = None
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["auprc"] = None
    out["positive_rate"] = float(y_true.mean())
    return out


def threshold_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return {}

    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "best_f1": float(f1[best_idx]),
        "best_threshold": best_threshold,
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    for target_recall in [0.8, 0.9]:
        valid = np.where(recall[:-1] >= target_recall)[0]
        if len(valid) > 0:
            idx = valid[int(np.nanargmax(precision[:-1][valid]))]
            out[f"precision_at_recall_{int(target_recall * 100)}"] = float(precision[idx])
            out[f"threshold_at_recall_{int(target_recall * 100)}"] = float(thresholds[idx])
    return out


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    losses, ys, ps = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ys.append(y.detach().cpu().numpy())
            ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    return float(np.mean(losses)), np.vstack(ys), np.vstack(ps)


def predict_dataset(model, loader, device="cpu"):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in tqdm(loader, leave=False):
            x = x.to(device)
            logits = model(x)
            ys.append(y.numpy())
            ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.vstack(ys), np.vstack(ps)


def train_predictor(data_dir, output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[predictor] Loading processed data from: {data_dir}")
    train_ds = WindowDataset(data_dir, split="train")
    val_ds = WindowDataset(data_dir, split="val")
    n_features, window_len = train_ds[0][0].shape
    cfg = config["predictor"]
    print(
        f"[predictor] train_windows={len(train_ds):,}, val_windows={len(val_ds):,}, "
        f"n_features={n_features}, window_len={window_len}"
    )

    train_y = np.asarray([train_ds[i][1].numpy() for i in range(len(train_ds))]).reshape(-1)
    if cfg.get("positive_weight", "auto") == "auto":
        pos = max(float(train_y.sum()), 1.0)
        neg = max(float(len(train_y) - train_y.sum()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    else:
        pos_weight = torch.tensor([float(cfg.get("positive_weight", 1.0))], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FaultTCN(
        n_features=n_features,
        hidden_channels=int(cfg.get("hidden_channels", 96)),
        num_blocks=int(cfg.get("num_blocks", 5)),
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.15)),
    ).to(device)
    print(f"[predictor] Using device={device}, batch_size={int(cfg.get('batch_size', 256))}")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 256)), shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 256)), shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 256)), shuffle=False, num_workers=0)

    best_score = -1.0
    best_epoch = -1
    history = []
    patience = int(cfg.get("patience", 10))
    for epoch in range(int(cfg.get("max_epochs", 60))):
        print(f"[predictor] Epoch {epoch} started.")
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, y_true, y_prob = run_epoch(model, val_loader, loss_fn, None, device)
        metrics = binary_metrics(y_true, y_prob)
        score = metrics.get("auprc")
        if score is None or (isinstance(score, float) and math.isnan(score)):
            score = -val_loss
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "n_features": n_features,
                "window_len": window_len,
                "config": config,
                "metrics": row,
            }, output_dir / "model.best.pt")
        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    best_path = output_dir / "model.best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    _, train_prob = predict_dataset(model, train_eval_loader, device)
    val_true, val_prob = predict_dataset(model, val_loader, device)
    val_threshold_metrics = threshold_metrics(val_true, val_prob)

    val_pred_df = val_ds.metadata().copy()
    val_pred_df["true_label"] = val_true.reshape(-1)
    val_pred_df["pred_score"] = val_prob.reshape(-1)
    val_pred_df.to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    train_pred_df = train_ds.metadata().copy()
    train_pred_df["true_label"] = train_y.reshape(-1)
    train_pred_df["pred_score"] = train_prob.reshape(-1)
    train_pred_df.to_csv(output_dir / "train_predictions.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "threshold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_threshold_metrics, f, ensure_ascii=False, indent=2)

    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False, encoding="utf-8-sig")
    return {**history[-1], **val_threshold_metrics}
