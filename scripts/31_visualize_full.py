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

LABELS = ["relaxed", "angry", "happy", "sad"]

def _norm(s):
    return str(s).strip().lower()

ALIASES = {
    # relaxed
    "relaxed": "relaxed", "calm": "relaxed",
    # angry
    "angry": "angry", "anger": "angry", "mad": "angry",
    # happy
    "happy": "happy", "joy": "happy", "happiness": "happy", "joyful": "happy",
    # sad
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
    """
    Load JSON with explicit UTF-8 to avoid platform-default encoding surprises.
    """
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    """
    Read messy CSV/TSV safely:
      - Try several encodings
      - Try automatic delimiter sniffing (sep=None, engine='python')
      - Then try common delimiters: ',', ';', '\\t', '|'
      - Final fallback: permissive parse with on_bad_lines='skip'
    """
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
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
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

    raise RuntimeError(f"Could not robustly parse file: {path}. "
                       f"Try opening it and re-saving as UTF-8 CSV (comma or tab-delimited).")

def title_and_tight(ax, title):
    ax.set_title(title)
    plt.tight_layout()

def canon_report(rep: Dict) -> Dict:
    """
    Canonicalize a sklearn classification_report (output_dict=True) section:
    - normalize keys
    - remap aliases to our 4 targets
    - drop anything not in TARGETS
    """
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

def plot_class_distribution():
    seed = DATA / "seed.csv"
    if not seed.exists():
        print("[skip] data/seed.csv not found — no class distribution plot.")
        return
    df = read_csv_robust(seed)

    label_col = None
    for cand in ["label", "mood", "emotion", "target"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        print("[skip] No label column found in seed.csv for distribution plot.")
        return


    df["_canon"] = df[label_col].map(lambda s: ALIASES.get(_norm(s), None))
    df = df[df["_canon"].notna()] 

    order = TARGETS 
    pretty_map = {_norm(x): x for x in LABELS} 
    order_pretty = [pretty_map[c] for c in order]

    df["_pretty"] = df["_canon"].map(lambda c: pretty_map[c])
    plt.figure()
    ax = sns.countplot(x="_pretty", data=df, order=order_pretty)
    ax.set_xlabel("Class")
    ax.set_ylabel("# Samples")
    title_and_tight(ax, "Class Distribution in Dataset (seed)")
    out = REPORTS / "01_class_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")


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

    base_rep_raw = base.get("report", {})
    ft_rep_raw = (ft or {}).get("report", {})

    base_rep = canon_report(base_rep_raw)
    ft_rep = canon_report(ft_rep_raw)

    classes = TARGETS
    pretty = LABELS

    base_f1 = [base_rep.get(c, {}).get("f1-score", 0.0) for c in classes]
    ft_f1 = [ft_rep.get(c, {}).get("f1-score", 0.0) for c in classes] if ft else None

    print("[info] Using classes:", pretty)

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

def plot_confusion_from_preds(csv_path, title, outfile):
    p = REPORTS / csv_path
    if not p.exists():
        print(f"[skip] {csv_path} not found.")
        return
    df = read_csv_robust(p)
    if "true" not in df.columns or "pred" not in df.columns:
        print(f"[skip] {csv_path} must contain 'true' and 'pred' columns.")
        return


    map4 = lambda s: ALIASES.get(_norm(s), None)
    df = df.assign(true=df["true"].map(map4), pred=df["pred"].map(map4))
    df = df.dropna(subset=["true", "pred"])

    labels_for_cm = TARGETS
    pretty = LABELS
    cm = confusion_matrix(df["true"], df["pred"], labels=labels_for_cm)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pretty)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    out = REPORTS / outfile
    plt.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")


def make_examples_table(csv_path, outfile, n=10):
    p = REPORTS / csv_path
    if not p.exists():
        print(f"[skip] {csv_path} not found for examples table.")
        return
    df = read_csv_robust(p)

    map4 = lambda s: ALIASES.get(_norm(s), None)
    if "true" in df.columns:
        df["true"] = df["true"].map(map4)
    if "pred" in df.columns:
        df["pred"] = df["pred"].map(map4)

    text_cols = [c for c in ["lyric_text", "text", "title", "artist"] if c in df.columns]
    wrong = df[df.get("true") != df.get("pred")].copy() if "true" in df.columns and "pred" in df.columns else pd.DataFrame()
    if wrong.empty:
        wrong = df.sample(min(n, len(df)), random_state=42)
    else:
        wrong = wrong.sample(min(n, len(wrong)), random_state=42)

    cols = ["true", "pred"]
    if text_cols:
        cols = text_cols + cols
    table = wrong[cols]
    out_csv = REPORTS / outfile
    table.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv} (sampled {len(table)} rows)")

def main():
    print("== Visualizing results ==")
    plot_class_distribution()
    plot_overall_comparison()
    plot_per_class_f1()
    plot_confusion_from_preds("baseline_preds.csv", "Confusion Matrix — Baseline", "04_confusion_baseline.png")
    plot_confusion_from_preds("finetuned_preds.csv", "Confusion Matrix — Fine-tuned", "05_confusion_finetuned.png")
    make_examples_table("baseline_preds.csv", "06_examples_baseline.csv", n=10)
    make_examples_table("finetuned_preds.csv", "07_examples_finetuned.csv", n=10)
    print("Done. Check the 'reports/' folder for PNGs and CSVs.")

if __name__ == "__main__":
    main()
