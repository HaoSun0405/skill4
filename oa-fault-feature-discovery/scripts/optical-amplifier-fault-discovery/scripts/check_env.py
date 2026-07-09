import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_root():
    bundled = SCRIPT_DIR / "optical-amplifier-fault-discovery"
    if (bundled / "src").exists():
        return bundled
    project = SCRIPT_DIR.parent
    if (project / "src").exists():
        return project
    return bundled


PROJECT_ROOT = resolve_project_root()
SRC = PROJECT_ROOT / "src"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

REQUIRED = [
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "yaml",
    "tqdm",
]

PARQUET_ENGINES = ["pyarrow", "fastparquet"]


def has_module(name):
    return importlib.util.find_spec(name) is not None


def check_with_python(python_exe):
    code = r"""
import importlib.util, json, sys
required = ["numpy", "pandas", "sklearn", "torch", "yaml", "tqdm"]
parquet = ["pyarrow", "fastparquet"]
result = {
    "python": sys.executable,
    "missing": [name for name in required if importlib.util.find_spec(name) is None],
    "parquet_ok": any(importlib.util.find_spec(name) is not None for name in parquet),
}
print(json.dumps(result))
"""
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", code],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return {"python": str(python_exe), "error": str(exc)}
    if proc.returncode != 0:
        return {"python": str(python_exe), "error": (proc.stderr or proc.stdout).strip()}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"python": str(python_exe), "error": f"Could not parse check output: {exc}"}


def unique_paths(paths):
    seen = set()
    out = []
    for item in paths:
        if not item:
            continue
        path = Path(item).expanduser()
        try:
            key = str(path.resolve()).lower()
        except OSError:
            key = str(path).lower()
        if key not in seen and path.exists():
            seen.add(key)
            out.append(path)
    return out


def python_executables_for_env(env_root):
    return [
        Path(env_root) / "python.exe",
        Path(env_root) / "Scripts" / "python.exe",
        Path(env_root) / "bin" / "python",
    ]


def filesystem_roots():
    roots = []
    if os.name == "nt":
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            root = Path(f"{letter}:\\")
            if root.exists():
                roots.append(root)
    else:
        roots.append(Path("/"))
    return roots


def known_env_roots():
    roots = []

    for env_var in ("VIRTUAL_ENV", "CONDA_PREFIX", "WORKON_HOME"):
        value = os.environ.get(env_var)
        if value:
            roots.append(Path(value))

    conda_envs_path = os.environ.get("CONDA_ENVS_PATH")
    if conda_envs_path:
        for item in conda_envs_path.split(os.pathsep):
            if item:
                roots.append(Path(item))

    envs_file = Path.home() / ".conda" / "environments.txt"
    if envs_file.exists():
        try:
            roots.extend(Path(line.strip()) for line in envs_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        except OSError:
            pass

    for drive in filesystem_roots():
        roots.extend([
            drive / "Anaconda3",
            drive / "anaconda3",
            drive / "Miniconda3",
            drive / "miniconda3",
            drive / "Miniforge3",
            drive / "Mambaforge",
            drive / "conda",
            drive / "envs",
            drive / "venvs",
            drive / ".venvs",
        ])
        for container in ("py", "python", "Python", "tools", "Tools", "software", "Software", "programs", "Programs", "dev", "Dev"):
            base = drive / container
            roots.extend([
                base,
                base / "Anaconda3",
                base / "anaconda3",
                base / "Miniconda3",
                base / "miniconda3",
                base / "Miniforge3",
                base / "Mambaforge",
                base / "conda",
                base / "envs",
                base / "venvs",
                base / ".venvs",
            ])

    roots.extend([
        Path.home() / "Anaconda3",
        Path.home() / "Miniconda3",
        Path.home() / ".conda" / "envs",
        Path.home() / ".virtualenvs",
    ])

    return unique_paths(roots)


def candidates_from_env_roots():
    candidates = []
    for root in known_env_roots():
        candidates.extend(python_executables_for_env(root))

        envs_dir = root / "envs"
        if envs_dir.exists():
            try:
                for child in envs_dir.iterdir():
                    if child.is_dir():
                        candidates.extend(python_executables_for_env(child))
            except OSError:
                pass

        try:
            if root.name.lower() in {"envs", "venvs", ".venvs", "virtualenvs"}:
                for child in root.iterdir():
                    if child.is_dir():
                        candidates.extend(python_executables_for_env(child))
        except OSError:
            pass

        try:
            for child in root.glob("Python*"):
                if child.is_dir():
                    candidates.extend(python_executables_for_env(child))
        except OSError:
            pass
        try:
            for pattern in ("*conda*", "*Conda*", "*venv*", "*Venv*", "Python*"):
                for child in root.glob(pattern):
                    if child.is_dir():
                        candidates.extend(python_executables_for_env(child))
                        envs_dir = child / "envs"
                        if envs_dir.exists():
                            for env in envs_dir.iterdir():
                                if env.is_dir():
                                    candidates.extend(python_executables_for_env(env))
        except OSError:
            pass
    return candidates


def candidate_pythons():
    candidates = [Path(sys.executable)]
    for env_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env = os.environ.get(env_var)
        if env:
            candidates.extend([Path(env) / "Scripts" / "python.exe", Path(env) / "bin" / "python"])
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    try:
        proc = subprocess.run(["where", "python"], text=True, capture_output=True, timeout=10, check=False)
        if proc.returncode == 0:
            candidates.extend(Path(line.strip()) for line in proc.stdout.splitlines() if line.strip())
    except Exception:
        pass
    try:
        proc = subprocess.run(["conda", "env", "list", "--json"], text=True, capture_output=True, timeout=20, check=False)
        if proc.returncode == 0:
            payload = json.loads(proc.stdout)
            for env in payload.get("envs", []):
                candidates.extend([Path(env) / "python.exe", Path(env) / "Scripts" / "python.exe", Path(env) / "bin" / "python"])
    except Exception:
        pass
    candidates.extend(candidates_from_env_roots())
    for base in [Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent, PROJECT_ROOT]:
        candidates.extend([
            base / ".venv" / "Scripts" / "python.exe",
            base / "venv" / "Scripts" / "python.exe",
            base / ".venv" / "bin" / "python",
            base / "venv" / "bin" / "python",
        ])
    return unique_paths(candidates)


def print_result(result, prefix=""):
    if "error" in result:
        print(f"{prefix}{result['python']}: ERROR: {result['error']}")
        return
    missing = result.get("missing", [])
    parquet_ok = result.get("parquet_ok", False)
    status = "OK" if not missing and parquet_ok else "MISSING"
    details = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if not parquet_ok:
        details.append("parquet_engine=missing")
    detail_text = "" if not details else " (" + "; ".join(details) + ")"
    print(f"{prefix}{result['python']}: {status}{detail_text}")


def main():
    parser = argparse.ArgumentParser(description="Check the Python environment for the OA fault skill. This script never installs packages.")
    parser.add_argument("--python", default=None, help="Explicit Python executable to check.")
    parser.add_argument("--list-candidates", action="store_true", help="Show candidate Python interpreters discovered from PATH, conda, local venvs, and common environment directories across drives.")
    args = parser.parse_args()
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    target_python = Path(args.python) if args.python else Path(sys.executable)
    current_result = check_with_python(target_python)
    print(f"Project: {PROJECT_ROOT}")
    print("Environment check only. No packages were installed.")
    print_result(current_result, prefix="Active Python: ")
    ok = "error" not in current_result and not current_result.get("missing") and bool(current_result.get("parquet_ok"))
    if ok and not args.list_candidates:
        print("\nEnvironment check passed.")
        return
    if ok:
        print("\nEnvironment check passed. Listing discovered candidates because --list-candidates was provided.")
    else:
        print("\nEnvironment check failed for the active Python.")
        print("Before installing anything, check whether another existing Python/conda environment already has the dependencies.")
    good = []
    print("\nDiscovered Python candidates:")
    for candidate in candidate_pythons():
        result = check_with_python(candidate)
        print_result(result, prefix="  ")
        if "error" not in result and not result.get("missing") and result.get("parquet_ok"):
            good.append(result["python"])
    if good:
        print("\nFound existing environment(s) that appear usable. Run the skill with one of these Python executables:")
        wrapper = PROJECT_ROOT.parent / "run_oa_fault_discovery.py"
        for python_exe in good:
            print(f"  \"{python_exe}\" \"{wrapper}\" --input <data> --cpu")
        return
    print("\nNo usable existing environment was found from PATH/conda/local venv/cross-drive candidates.")
    print("Do not install automatically. Ask the user before installing dependencies, especially torch.")
    print("\nIf the user approves installation, suggested command for the active Python:")
    print(f"  \"{target_python}\" -m pip install -r \"{REQUIREMENTS}\"")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
