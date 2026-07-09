import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .datasets import WindowDataset
from .models import WindowVAE, vae_loss


def select_by_label(dataset, wanted_label):
    if wanted_label is None:
        return dataset
    indices = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if int(y.item()) == int(wanted_label):
            indices.append(i)
    if not indices:
        raise ValueError(f"No generator samples found for label={wanted_label}")
    return Subset(dataset, indices)


def run_generator_epoch(model, loader, optimizer=None, device="cpu", beta=0.01, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    losses, recs, kls = [], [], []
    with torch.set_grad_enabled(train):
        for x, _ in tqdm(loader, leave=False):
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, beta=beta)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            recs.append(float(rec.cpu()))
            kls.append(float(kl.cpu()))
    return {
        "loss": float(np.mean(losses)),
        "reconstruction": float(np.mean(recs)),
        "kl": float(np.mean(kls)),
    }


def train_generator(data_dir, output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config["generator"]
    print(f"[generator] Loading processed data from: {data_dir}")
    base_ds = WindowDataset(data_dir, split="train")
    val_base_ds = WindowDataset(data_dir, split="val")
    train_ds = select_by_label(base_ds, cfg.get("train_on_label"))
    val_ds = select_by_label(val_base_ds, cfg.get("train_on_label"))
    n_features, window_len = train_ds[0][0].shape
    print(
        f"[generator] train_windows={len(train_ds):,}, "
        f"val_windows={len(val_ds):,}, n_features={n_features}, window_len={window_len}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WindowVAE(
        n_features=n_features,
        window_len=window_len,
        latent_dim=int(cfg.get("latent_dim", 32)),
        hidden_channels=int(cfg.get("hidden_channels", 64)),
    ).to(device)
    print(f"[generator] Using device={device}, batch_size={int(cfg.get('batch_size', 256))}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 256)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 256)), shuffle=False, num_workers=0)

    history = []
    best_loss = float("inf")
    best_epoch = -1
    patience = int(cfg.get("patience", 10))
    min_delta = float(cfg.get("min_delta", 1e-4))
    for epoch in range(int(cfg.get("max_epochs", 80))):
        print(f"[generator] Epoch {epoch} started.")
        train_metrics = run_generator_epoch(
            model,
            loader,
            optimizer=optimizer,
            device=device,
            beta=float(cfg.get("beta", 0.01)),
            grad_clip=float(cfg.get("grad_clip", 1.0)),
        )
        val_metrics = run_generator_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            beta=float(cfg.get("beta", 0.01)),
            grad_clip=float(cfg.get("grad_clip", 1.0)),
        )
        row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "reconstruction": train_metrics["reconstruction"],
            "kl": train_metrics["kl"],
            "val_loss": val_metrics["loss"],
            "val_reconstruction": val_metrics["reconstruction"],
            "val_kl": val_metrics["kl"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if row["val_loss"] < best_loss - min_delta:
            best_loss = row["val_loss"]
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "n_features": n_features,
                "window_len": window_len,
                "config": config,
                "metrics": row,
            }, output_dir / "vae.pt")
        if epoch - best_epoch >= patience:
            print(f"[generator] Early stopping at epoch {epoch}.")
            break

    if not (output_dir / "vae.pt").exists():
        torch.save({
            "model_state": model.state_dict(),
            "n_features": n_features,
            "window_len": window_len,
            "config": config,
            "metrics": history[-1],
        }, output_dir / "vae.pt")
    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False, encoding="utf-8-sig")
    return history[-1]
