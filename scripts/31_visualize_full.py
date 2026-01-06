import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


REPORTS = Path("reports")
DATA = Path("data")
REPORTS.mkdir(parents=True, exist_ok=True)

LABELS = ["happy", "sad", "angry", "relaxed"]

def _norm(s):
    return str(s).strip().lower()

ALIASES = {
    "relaxed": "relaxed", "calm": "relaxed",
    "angry": "angry", "anger": "angry", "mad": "angry",
    "happy": "happy", "joy": "happy", "happiness": "happy", "joyful": "happy",
    "sad": "sad", "sadness": "sad", "unhappy": "sad"
}
TARGETS = [_norm(x) for x in LABELS]

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
sns.set_theme(style="whitegrid")

SKIP_KEYS = {"accuracy", "macro avg", "weighted avg", "samples avg", "micro avg"}

def safe_load_json(path: Path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    seps = [None, ",", ";", "\t", "|"]

    for enc in encodings:
        for sep in seps:
            try:
                opts = dict(encoding=enc, low_memory=False, **kwargs)
                if sep is None:
                    return pd.read_csv(path, sep=None, engine="python", **opts)
                else:
                    return pd.read_csv(path, sep=sep, **opts)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except Exception:
                pass

    for enc in encodings:
        for sep in seps:
            try:
                opts = dict(encoding=enc, low_memory=False, on_bad_lines="skip", **kwargs)
                if sep is None:
                    return pd.read_csv(path, sep=None, engine="python", **opts)
                else:
                    return pd.read_csv(path, sep=sep, engine="python", **opts)
            except Exception:
                continue

    raise RuntimeError(f"Could not robustly parse file: {path}")

def title_and_tight(ax, title):
    ax.set_title(title)
    plt.tight_layout()

def canon_report(rep: Dict) -> Dict:
    out = {}
    for k, v in (rep or {}).items():
        nk = _norm(k)
        if nk in {_norm(x) for x in SKIP_KEYS}:
            continue
        if not isinstance(v, dict):
            continue
        mapped = ALIASES.get(nk)
        if mapped in TARGETS:
            out[mapped] = v
    return out

def plot_overall_comparison():
    base = safe_load_json(REPORTS / "baseline_prompt_metrics.json")
    ft = safe_load_json(REPORTS / "finetuned_metrics.json")
    if not base:
        print("[skip] baseline_prompt_metrics.json not found.")
        return

    labels = ["Baseline"]
    acc = [base.get("accuracy", None)]
    f1m = [base.get("f1_macro", base.get("f1-macro", None))]

    if ft:
        labels.append("Fine-tuned")
        acc.append(ft.get("accuracy", None))
        f1m.append(ft.get("f1_macro", ft.get("f1-macro", None)))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, acc, width, label="Accuracy")
    ax.bar(x + width/2, f1m, width, label="Macro-F1")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    title_and_tight(ax, "Overall Performance")
    ax.legend()
    out = REPORTS / "02_overall_performance.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")

def plot_per_class_f1():
    base = safe_load_json(REPORTS / "baseline_prompt_metrics.json")
    ft = safe_load_json(REPORTS / "finetuned_metrics.json")
    if not base:
        print("[skip] baseline_prompt_metrics.json not found for per-class F1.")
        return

    base_rep = canon_report(base.get("report", {}))
    ft_rep = canon_report((ft or {}).get("report", {}))

    classes = TARGETS
    pretty = LABELS

    base_f1 = [base_rep.get(c, {}).get("f1-score", 0.0) for c in classes]
    ft_f1 = [ft_rep.get(c, {}).get("f1-score", 0.0) for c in classes] if ft else None

    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, base_f1, width, label="Baseline")
    if ft:
        ax.bar(x + width/2, ft_f1, width, label="Fine-tuned")
    ax.set_xticks(x); ax.set_xticklabels(pretty)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    title_and_tight(ax, "Per-class F1")
    ax.legend()
    out = REPORTS / "03_per_class_f1.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")


def plot_confusion_from_matrix_csv(matrix_csv, title, outfile):
    p = REPORTS / matrix_csv
    if not p.exists():
        print(f"[skip] {matrix_csv} not found.")
        return

    cm_df = pd.read_csv(p, index_col=0)  
    cm = cm_df.values

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

    out = REPORTS / outfile
    plt.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")

def main():
    print("== Visualizing results ==")
    plot_overall_comparison()
    plot_per_class_f1()

    plot_confusion_from_matrix_csv(
        "baseline_confusion_matrix.csv",
        "Confusion Matrix — Baseline",
        "04_confusion_baseline.png"
    )
    plot_confusion_from_matrix_csv(
        "finetuned_confusion_matrix.csv",
        "Confusion Matrix — Fine-tuned",
        "05_confusion_finetuned.png"
    )

    print("Done. Check the 'reports/' folder for PNGs.")

if __name__ == "__main__":
    main()
