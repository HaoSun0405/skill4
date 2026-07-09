import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oa_fault.config import load_config
from oa_fault.preprocessing import prepare_data
from oa_fault.train_generator import train_generator
from oa_fault.train_predictor import train_predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    data_dir = output_dir / "processed_data"
    predictor_dir = output_dir / "predictor"
    generator_dir = output_dir / "generator"

    print("[1/4] Preparing data...")
    summary = prepare_data(args.input, data_dir, config)
    print("[2/4] Training predictor...")
    predictor_metrics = train_predictor(data_dir, predictor_dir, config)
    print("[3/4] Training generator...")
    generator_metrics = train_generator(data_dir, generator_dir, config)

    run_summary = {
        "processed_data": str(data_dir),
        "predictor_output": str(predictor_dir),
        "generator_output": str(generator_dir),
        "window_shape": summary["window_shape"],
        "n_features": summary["n_features"],
        "n_windows": summary["n_windows"],
        "predictor_metrics": predictor_metrics,
        "generator_metrics": generator_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
    print("[4/4] Done. Run summary:")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
