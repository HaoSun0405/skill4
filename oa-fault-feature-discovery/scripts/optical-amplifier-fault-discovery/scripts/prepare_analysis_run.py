import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oa_fault.config import load_config
from oa_fault.datasets import WindowDataset
from oa_fault.hidden_features import load_predictor
from oa_fault.preprocessing import prepare_data
from oa_fault.train_predictor import predict_dataset


def copy_required_models(model_run, output_run):
    model_run = Path(model_run)
    output_run = Path(output_run)

    files = [
        (model_run / "predictor" / "model.best.pt", output_run / "predictor" / "model.best.pt"),
        (model_run / "generator" / "vae.pt", output_run / "generator" / "vae.pt"),
    ]
    for src, dst in files:
        if not src.exists():
            raise FileNotFoundError(f"Missing model artifact: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_predictions(output_run, batch_size=256, device="cpu"):
    output_run = Path(output_run)
    data_dir = output_run / "processed_data"
    predictor_dir = output_run / "predictor"
    predictor, checkpoint = load_predictor(predictor_dir / "model.best.pt", device)

    summary_path = data_dir / "preprocess_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        prep = json.load(f)
    window_shape = prep.get("window_shape") or []
    n_features = int(prep.get("n_features", window_shape[0] if len(window_shape) >= 1 else 0))
    window_len = int(window_shape[1] if len(window_shape) >= 2 else prep.get("window_len", 0))
    if int(checkpoint["n_features"]) != n_features or int(checkpoint["window_len"]) != window_len:
        raise ValueError(
            "Default predictor dimensions do not match processed data: "
            f"model=({checkpoint['window_len']}, {checkpoint['n_features']}), "
            f"data=({window_len}, {n_features})"
        )

    for split, filename in [
        ("train", "train_predictions.csv"),
        ("val", "validation_predictions.csv"),
        ("test", "test_predictions.csv"),
    ]:
        try:
            dataset = WindowDataset(data_dir, split=split)
        except ValueError:
            continue
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        y_true, y_prob = predict_dataset(predictor, loader, device)
        pred_df = dataset.metadata().copy()
        pred_df["true_label"] = np.asarray(y_true).reshape(-1)
        pred_df["pred_score"] = np.asarray(y_prob).reshape(-1)
        pred_df.to_csv(predictor_dir / filename, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an analysis run from raw parquet while loading predictor/generator from an existing default run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-run", required=True, help="Run directory containing predictor/ and generator/.")
    parser.add_argument("--output", required=True, help="New analysis run directory.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_run = Path(args.output)
    data_dir = output_run / "processed_data"
    output_run.mkdir(parents=True, exist_ok=True)

    print("[analysis-run] Preparing processed_data from raw parquet...")
    prep_summary = prepare_data(args.input, data_dir, load_config(args.config))

    print("[analysis-run] Copying default predictor/generator artifacts...")
    copy_required_models(args.model_run, output_run)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[analysis-run] Writing prediction details with device={device}...")
    write_predictions(output_run, batch_size=args.batch_size, device=device)

    run_summary = {
        "mode": "analysis_with_default_models",
        "input": str(Path(args.input).resolve()),
        "model_run": str(Path(args.model_run).resolve()),
        "output": str(output_run.resolve()),
        "preprocess_summary": prep_summary,
    }
    with open(output_run / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
