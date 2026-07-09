from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    feature_cols_file = data_cfg.get("feature_cols_file")
    if feature_cols_file and not data_cfg.get("feature_cols"):
        feature_path = Path(feature_cols_file)
        if not feature_path.is_absolute():
            candidates = [path.parent / feature_path, path.parent.parent / feature_path]
            feature_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        with open(feature_path, "r", encoding="utf-8") as f:
            data_cfg["feature_cols"] = [
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
        data_cfg["feature_cols_file_resolved"] = str(feature_path)

    return config


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
