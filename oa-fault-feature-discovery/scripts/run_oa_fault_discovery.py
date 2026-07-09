import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_DIR = Path(
    os.environ.get("OA_FAULT_PROJECT_DIR", SKILL_DIR / "scripts" / "optical-amplifier-fault-discovery")
)
DEFAULT_MODEL_RUN = Path(
    os.environ.get("OA_FAULT_MODEL_RUN", SKILL_DIR / "models" / "default_run")
)
GENERATED_ARTIFACT_NAMES = {
    "processed_data",
    "predictor",
    "generator",
    "hidden_features",
    "agent_context.json",
    "run_summary.json",
    "tool_generated_report.md",
}


def run_command(cmd, cwd, verbose=False):
    if verbose:
        print("[skill-run] " + " ".join(str(x) for x in cmd), flush=True)
    if verbose:
        result = subprocess.run(cmd, cwd=str(cwd), text=True)
    else:
        result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if result.returncode != 0:
        if not verbose:
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            print(output[-4000:], flush=True)
        raise SystemExit(result.returncode)


def find_single_parquet(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".parquet":
        return path
    candidates = sorted(path.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet file found in: {path}")
    if len(candidates) > 1:
        names = "\n".join(f"- {p}" for p in candidates)
        raise ValueError(f"Multiple parquet files found. Please pass --input explicitly:\n{names}")
    return candidates[0]


def cleanup_analysis_artifacts(output_run):
    output_run = Path(output_run)
    if not output_run.exists():
        return
    for name in GENERATED_ARTIFACT_NAMES:
        child = output_run / name
        if child.is_dir():
            shutil.rmtree(child)
        elif child.exists():
            child.unlink()
    try:
        output_run.rmdir()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run optical amplifier fault feature discovery from a dataset parquet using default skill models."
    )
    parser.add_argument("--input", default=".", help="Parquet file or directory containing exactly one parquet file.")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--model-run", default=str(DEFAULT_MODEL_RUN))
    parser.add_argument("--config", default=None, help="Config path. Defaults to project configs/default.yaml.")
    parser.add_argument("--output", default=None, help="Analysis run output directory. Explicit custom outputs are kept unless --cleanup-artifacts is set.")
    parser.add_argument("--report", default=None, help="Final user-facing markdown report path.")
    parser.add_argument("--cleanup-artifacts", action="store_true", help="Remove internal evidence files after writing the final report. Default behavior for the auto-generated artifact directory.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep internal evidence files and agent_context.json for debugging.")
    parser.add_argument("--verbose", action="store_true", help="Print full output from internal project scripts.")
    parser.add_argument("--quick", action="store_true", help="Use configs/fake_quick.yaml for fake-data tests.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--n-hf", type=int, default=5)
    parser.add_argument("--selection-mode", default="low_mid", choices=["low_mid", "high", "all"])
    parser.add_argument("--morph-steps", type=int, default=50)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--max-latent-norm", type=float, default=5.0)
    parser.add_argument("--target-logit-delta", type=float, default=2.0)
    parser.add_argument("--morph-method", default="multi-gradient", choices=["gradient", "conservative-gradient", "multi-gradient", "risk-centroid"])
    parser.add_argument("--semantic-couplings", default=None, help="Optional agent-generated semantic coupling candidate JSON.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    model_run = Path(args.model_run)
    parquet_path = find_single_parquet(args.input).resolve()

    if args.config:
        config_path = Path(args.config)
    elif args.quick:
        config_path = project_dir / "configs" / "fake_quick.yaml"
    else:
        config_path = project_dir / "configs" / "default.yaml"

    if args.keep_artifacts and args.cleanup_artifacts:
        raise ValueError("Use only one of --keep-artifacts or --cleanup-artifacts.")

    output_run = Path(args.output) if args.output else parquet_path.parent / ".oa_fault_feature_discovery_artifacts"
    report_path = Path(args.report) if args.report else parquet_path.parent / "oa_fault_feature_report.md"
    keep_artifacts = args.keep_artifacts or (args.output is not None and not args.cleanup_artifacts)

    print("[progress 1/8] [#-------] identify dataset and paths", flush=True)
    if args.verbose:
        print(f"[skill-run] parquet={parquet_path}", flush=True)
        print(f"[skill-run] project_dir={project_dir}", flush=True)
        print(f"[skill-run] model_run={model_run}", flush=True)

    required = [
        project_dir / "scripts" / "prepare_analysis_run.py",
        project_dir / "scripts" / "discover_hidden_features.py",
        config_path,
        model_run / "predictor" / "model.best.pt",
        model_run / "generator" / "vae.pt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    print("[progress 2/8] [##------] check parquet, config, and environment", flush=True)
    print("[progress 3/8] [###-----] preprocess parquet and build windows", flush=True)
    print("[progress 4/8] [####----] load predictor from default_run and score current windows", flush=True)
    print("[progress 5/8] [#####---] load generator from default_run", flush=True)
    prepare_cmd = [
        sys.executable,
        str(project_dir / "scripts" / "prepare_analysis_run.py"),
        "--config",
        str(config_path),
        "--input",
        str(parquet_path),
        "--model-run",
        str(model_run),
        "--output",
        str(output_run),
    ]
    if args.cpu:
        prepare_cmd.append("--cpu")
    run_command(prepare_cmd, project_dir, verbose=args.verbose)

    print("[progress 6/8] [######--] morph windows toward higher predicted risk", flush=True)
    discover_cmd = [
        sys.executable,
        str(project_dir / "scripts" / "discover_hidden_features.py"),
        "--run",
        str(output_run),
        "--selection-mode",
        str(args.selection_mode),
        "--morph-method",
        str(args.morph_method),
        "--max-windows",
        str(args.max_windows),
        "--n-hf",
        str(args.n_hf),
        "--morph-steps",
        str(args.morph_steps),
        "--step-size",
        str(args.step_size),
        "--max-latent-norm",
        str(args.max_latent_norm),
        "--target-logit-delta",
        str(args.target_logit_delta),
    ]
    if args.semantic_couplings:
        discover_cmd.extend(["--semantic-couplings", str(Path(args.semantic_couplings))])
    if args.cpu:
        discover_cmd.append("--cpu")
    run_command(discover_cmd, project_dir, verbose=args.verbose)

    print("[progress 7/8] [#######-] check physical credibility", flush=True)
    print("[progress 8/8] [########] emit structured evidence for agent report", flush=True)
    internal_report = output_run / "tool_generated_report.md"
    if not internal_report.exists():
        raise FileNotFoundError(f"Missing generated markdown report: {internal_report}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(internal_report, report_path)
    if internal_report.resolve() != report_path.resolve():
        internal_report.unlink(missing_ok=True)

    hidden_dir = output_run / "hidden_features"
    agent_context = {
        "parquet_path": str(parquet_path),
        "final_report_path": str(report_path),
        "final_report_generated": True,
        "internal_artifacts_dir": str(output_run),
        "hidden_features_path": str(hidden_dir / "hidden_features.json"),
        "evidence_path": str(hidden_dir / "evidence.json"),
        "physical_validation_path": str(hidden_dir / "physical_validation.json"),
        "field_contributions_path": str(hidden_dir / "field_contributions.csv"),
        "semantic_couplings_path": str(hidden_dir / "semantic_couplings.csv"),
        "morph_windows_path": str(hidden_dir / "morph_windows.csv"),
        "report_instruction": (
            "The final user-facing markdown has already been generated deterministically at final_report_path. "
            "Do not rewrite or reformat the report. "
            "Do not look for or rely on markdown reports inside hidden_features; that directory is internal evidence only. "
            "Only summarize the report path, physical credibility, and 1-3 key findings for the user. "
            "Use evidence files only if the user explicitly asks for debugging details."
        ),
        "report_template": {
            "required_sections": [
                "分析概况",
                "总体结论",
                "候选前兆模式",
                "跨模式共性发现",
                "未展开模式说明",
                "建议",
            ],
            "pattern_sections": [
                "主要现象",
                "关键证据",
                "可能含义",
                "使用边界",
            ],
            "title_rule": "Use a business semantic title for each pattern. Do not use HF ids as titles.",
            "hf_id_rule": "HF ids are internal trace IDs only.",
            "risk_corr_thresholds": {
                "strong": 0.60,
                "strong_operator": ">",
            },
            "empty_result_wording": "本次分析未发现具备强相关证据的候选前兆模式。主要原因可能是 morph 后风险提升幅度较小，或字段/耦合变化与风险提升之间没有形成稳定关系。",
        },
    }
    context_path = output_run / "agent_context.json"
    if keep_artifacts:
        context_path.write_text(json.dumps(agent_context, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[skill-run] agent_context={context_path}", flush=True)
    else:
        cleanup_analysis_artifacts(output_run)
        print(f"[skill-run] final_report={report_path}", flush=True)
        print("[skill-run] internal_artifacts=removed; rerun with --keep-artifacts to retain debug evidence", flush=True)
    if args.verbose and keep_artifacts:
        print(json.dumps(agent_context, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
