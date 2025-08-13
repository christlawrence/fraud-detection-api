# metrics_report.py
import json
import os
from pathlib import Path

METRICS_PATH = Path("reports/figures/metrics_summary.json")

def load_metrics(path=METRICS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run training first.")
    return json.loads(path.read_text())

def pretty_report(metrics):
    print("=== Model Metrics Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\nNotes / sanity checks:")
    if metrics.get("auprc") is None:
        print(" - AUPRC not present in metrics JSON. Consider computing and saving AUPRC during training.")
    else:
        auprc = float(metrics["auprc"])
        print(f" - AUPRC = {auprc:.4f}")
        if auprc < 0.5:
            print("   WARNING: AUPRC is low for imbalanced task; model may rank poorly.")
    best_it = metrics.get("best_iteration")
    if best_it is None:
        print(" - best_iteration not recorded.")
    else:
        print(f" - best_iteration (0-based): {best_it}")

if __name__ == "__main__":
    m = load_metrics()
    pretty_report(m)