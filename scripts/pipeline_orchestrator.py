# run_pipeline.py
"""
Orchestrates the full pipeline with structured logging and selectable stages.
Usage examples:
    python run_pipeline.py --all
    python run_pipeline.py --train --metrics --cv
    python run_pipeline.py --hpt --finalize
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
VENV = ROOT / ".venv"
PY = VENV / "bin" / "python"

def run(cmd, check=True):
    print(f"\n>>> RUN: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if check and res.returncode != 0:
        sys.exit(res.returncode)

def ensure_setup():
    if not VENV.exists():
        run("python3 -m venv .venv")
    run(f"{VENV}/bin/pip install -U pip")
    run(f"{VENV}/bin/pip install -r requirements.txt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--hpt", action="store_true")
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        args.setup = args.train = args.metrics = args.cv = args.hpt = args.shap = args.smoke = args.finalize = True

    if args.setup:
        ensure_setup()

    if args.train:
        run(f"{PY} scripts/train_improved.py")

    if args.metrics:
        run(f"{PY} metrics_report.py")

    if args.cv:
        run(f"{PY} scripts/cv_evaluate.py")

    if args.hpt:
        run(f"{PY} scripts/hyperparam_search.py --trials 24 --early_rounds 50")

    if args.shap:
        run(f"{PY} scripts/shap_analysis_snippet.py")

    if args.smoke:
        run(f"{PY} scripts/inference_smoke_test.py")

    if args.finalize:
        tuned = ROOT / "models" / "fraud_detection_model_tuned.json"
        default = ROOT / "models" / "fraud_detection_model.json"
        final = ROOT / "models" / "fraud_detection_model_final.json"
        if tuned.exists():
            final.write_bytes(tuned.read_bytes())
            feats_src = Path(str(tuned) + ".features")
        elif default.exists():
            final.write_bytes(default.read_bytes())
            feats_src = Path(str(default) + ".features")
        else:
            print("No model found to finalize. Run training first.")
            sys.exit(1)
        feats_dst = Path(str(final) + ".features")
        if feats_src.exists():
            feats_dst.write_text(feats_src.read_text())
        meta = Path(str(final) + ".metadata.json")
        if not meta.exists():
            meta.write_text('{"decision_threshold": 0.5, "source": "run_pipeline.py finalize"}')
        print(f"Final model prepared at: {final}")

    if args.api:
        print("Starting API at http://127.0.0.1:8000/docs")
        run(f"{VENV}/bin/uvicorn scripts.predict:app --reload", check=False)

if __name__ == "__main__":
    main()
