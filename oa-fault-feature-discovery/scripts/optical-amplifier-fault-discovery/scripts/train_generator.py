import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oa_fault.config import load_config
from oa_fault.preprocessing import prepare_data
from oa_fault.train_generator import train_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = args.data
    if data_dir is None:
        if args.input is None:
            raise ValueError("Provide either --data processed_dir or --input raw_table.parquet.")
        data_dir = f"{args.output}/processed_data"
        prepare_data(args.input, data_dir, config)
    train_generator(data_dir, args.output, config)


if __name__ == "__main__":
    main()
