import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oa_fault.hidden_features import discover_hidden_features


def main():
    parser = argparse.ArgumentParser(
        description="Discover and explain hidden risk features from trained predictor/generator outputs."
    )
    parser.add_argument("--run", required=True, help="Run directory, e.g. outputs/run_001")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Window split to explain")
    parser.add_argument("--max-windows", type=int, default=1000, help="Maximum windows to morph")
    parser.add_argument("--max-baseline-windows", type=int, default=5000, help="Maximum low-risk windows for coupling baselines")
    parser.add_argument("--n-hf", type=int, default=5, help="Number of hidden feature candidates to cluster")
    parser.add_argument("--selection-mode", default="low_mid", choices=["low_mid", "high", "all"], help="Which windows to morph")
    parser.add_argument("--morph-method", default="gradient", choices=["gradient", "conservative-gradient", "multi-gradient", "risk-centroid"], help="Latent morph direction method")
    parser.add_argument("--max-direction-windows", type=int, default=5000, help="Maximum windows used to estimate risk-centroid direction")
    parser.add_argument("--morph-steps", type=int, default=12, help="Gradient morphing steps in VAE latent space")
    parser.add_argument("--step-size", type=float, default=0.12, help="Latent morph step size")
    parser.add_argument("--max-latent-norm", type=float, default=3.0, help="Maximum latent displacement from the original point")
    parser.add_argument("--target-logit-delta", type=float, default=3.0, help="Use the first morph step reaching this logit increase")
    parser.add_argument("--target-pred-score", type=float, default=None, help="Use the first morph step reaching this predicted risk")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for morphing")
    parser.add_argument("--top-fields", type=int, default=12, help="Top field contributions per HF")
    parser.add_argument("--top-relations", type=int, default=10, help="Legacy relation display hint; semantic couplings are generated high-recall and filtered by strong evidence.")
    parser.add_argument("--representatives", type=int, default=5, help="Representative windows per HF")
    parser.add_argument("--semantic-couplings", default=None, help="Optional agent-generated semantic coupling candidate JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--print-report", action="store_true", help="Print the hidden feature report to console")
    parser.add_argument("--print-json", action="store_true", help="Print the hidden feature JSON summary to console")
    args = parser.parse_args()

    summary = discover_hidden_features(args.run, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
