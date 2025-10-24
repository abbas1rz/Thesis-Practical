import argparse
import re
import unicodedata
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CANON_LABELS = ["happy", "sad", "angry", "relaxed"]


def load_table(path_str: str, csv_encoding: str | None = None, csv_sep: str | None = None) -> pd.DataFrame:
    """
    Robustly load CSV/XLSX with encoding/separator tolerance.
    If csv_encoding or csv_sep is provided, those are used directly.
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)

    if csv_encoding or (csv_sep is not None):
        return pd.read_csv(
            p,
            sep=csv_sep,
            encoding=csv_encoding or "utf-8",
            engine="python",
            quotechar='"',
            doublequote=True,
            escapechar=None,
            on_bad_lines="error",
            dtype=str,
        )

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [",", ";", "\t", "|", None]  
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(
                    p,
                    sep=sep,
                    encoding=enc,
                    engine="python",
                    quotechar='"',
                    doublequote=True,
                    escapechar=None,
                    on_bad_lines="error",
                    dtype=str,
                )
            except Exception as e:
                last_err = e
                continue

    for enc in ["cp1252", "latin1"]:
        try:
            return pd.read_csv(
                p,
                sep=None,  # sniff
                encoding=enc,
                engine="python",
                quotechar='"',
                doublequote=True,
                escapechar=None,
                on_bad_lines="skip",
                dtype=str,
            )
        except Exception as e:
            last_err = e
            continue

    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                p,
                sep=None,
                encoding=enc,
                encoding_errors="ignore",
                engine="python",
                quotechar='"',
                doublequote=True,
                escapechar=None,
                on_bad_lines="skip",
                dtype=str,
            )
        except Exception as e:
            last_err = e
            continue

    raise last_err if last_err else RuntimeError("Failed to read CSV with all strategies.")


def normalize_label(x: str):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    syn = {
        # happy
        "happy": "happy", "joy": "happy", "joyful": "happy", "happiness": "happy",
        "positive": "happy", "glad": "happy", "cheerful": "happy", "upbeat": "happy",
        # sad
        "sad": "sad", "sadness": "sad", "melancholy": "sad", "blue": "sad",
        "down": "sad", "heartbroken": "sad", "lonely": "sad",
        # angry
        "angry": "angry", "anger": "angry", "mad": "angry", "aggressive": "angry",
        "rage": "angry", "furious": "angry", "rebellious": "angry",
        # relaxed
        "relaxed": "relaxed", "calm": "relaxed", "peaceful": "relaxed", "chill": "relaxed",
        "soothing": "relaxed", "serene": "relaxed", "tranquil": "relaxed", "content": "relaxed",
    }
    return syn.get(s, None)


# ---------- Artist helpers ----------
def canon_artist(a: str) -> str:
    a = unicodedata.normalize("NFKC", str(a)).lower().strip()
    a = re.sub(r"\s+", " ", a)
    a = re.sub(r"[^\w\s]", "", a) 
    return a

def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_csv", type=str, required=True,
                    help="Path to CSV/XLSX. Columns: lyrics/label (and optionally artist/title).")
    ap.add_argument("--train_size", type=float, default=0.80)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--test_size", type=float, default=0.10)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--require_all_classes", action="store_true",
                    help="Retry seeds until all 4 labels appear in each split.")
    ap.add_argument("--max_retries", type=int, default=30)
    ap.add_argument("--csv_encoding", type=str, default=None,
                    help="Force CSV encoding (e.g., cp1252).")
    ap.add_argument("--csv_sep", type=str, default=None,
                    help="Force CSV separator: ',', ';', '\\t', or '|'.")
    args = ap.parse_args()

    total = args.train_size + args.val_size + args.test_size
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Splits must sum to 1.0, got {total}")

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.seed_csv, csv_encoding=args.csv_encoding, csv_sep=args.csv_sep)
    print(f"Loaded {len(df)} rows from {args.seed_csv}")

    df.columns = [c.strip().lower() for c in df.columns]
    col_artist = next((c for c in df.columns if c in {"artist", "singer", "band"}), None)
    col_title  = next((c for c in df.columns if c in {"title", "song", "track"}), None)
    col_lyrics = next((c for c in df.columns if c in {"lyrics", "lyric_text", "text"}), None)
    col_label  = next((c for c in df.columns if c in {"mood", "label", "emotion", "target"}), None)

    if col_label is None:
        raise ValueError("Missing label column (one of: mood/label/emotion/target).")


    if col_lyrics:
        text = df[col_lyrics].astype(str)
    else:
        text = (df.get(col_title, "")).astype(str) + " by " + (df.get(col_artist, "")).astype(str)

    if col_artist is None:
        artist_series = "unknown_" + df[col_label].astype(str).fillna("").str.lower() + "_" + text.apply(short_hash)
    else:
        artist_series = df[col_artist].astype(str)

    out = pd.DataFrame({
        "song_id": np.arange(1, len(df) + 1),
        "artist": artist_series,
        "lyric_text": text.str.replace("\r\n", "\n").str.replace("\r", "\n"),
        "label": df[col_label].apply(normalize_label),
    })

    before = len(out)
    out = out.dropna(subset=["label"]).copy()
    out = out[out["lyric_text"].astype(str).str.strip().ne("")]
    after = len(out)
    print(f"Dropped {before - after} invalid rows.")

    out["artist"] = out["artist"].apply(canon_artist)
    out = out.drop_duplicates(subset=["lyric_text", "artist"]).reset_index(drop=True)

    if len(out) == 0:
        raise ValueError("No valid rows after cleaning. Check your label mapping and text columns.")

    def counts(d): return dict(d["label"].value_counts().sort_index())
    def has_all_classes(d): return set(d["label"].unique()) == set(CANON_LABELS)

    rng_seed = int(args.random_state)
    retries = 0
    while True:
        groups = out["artist"].values

        gss1 = GroupShuffleSplit(n_splits=1, train_size=args.train_size, random_state=rng_seed)
        train_idx, temp_idx = next(gss1.split(out, groups=groups))
        train_df = out.iloc[train_idx].reset_index(drop=True)
        temp_df  = out.iloc[temp_idx].reset_index(drop=True)

        temp_share = args.val_size + args.test_size
        if temp_share == 0:
            val_df = out.iloc[[]].copy()
            test_df = out.iloc[[]].copy()
        else:
            if args.val_size == 0:
                val_df = out.iloc[[]].copy()
                test_df = temp_df
            elif args.test_size == 0:
                val_df = temp_df
                test_df = out.iloc[[]].copy()
            else:
                val_share_in_temp = args.val_size / temp_share
                gss2 = GroupShuffleSplit(n_splits=1, train_size=val_share_in_temp, random_state=rng_seed)
                val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["artist"].values))
                val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
                test_df = temp_df.iloc[test_idx].reset_index(drop=True)

        ok = True
        if args.require_all_classes and len(train_df) and len(val_df) and len(test_df):
            ok = has_all_classes(train_df) and has_all_classes(val_df) and has_all_classes(test_df)

        if ok:
            break
        retries += 1
        if retries >= args.max_retries:
            print("Warning: reached max retries; proceeding with current split even if some classes are missing.")
            break
        rng_seed += 1


    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train.csv", index=False)
    val_df.to_csv(outdir / "val.csv", index=False)
    test_df.to_csv(outdir / "test.csv", index=False)

    print("\nSaved splits:")
    print(f"  train: {len(train_df)} rows  {counts(train_df)}")
    print(f"  val  : {len(val_df)} rows  {counts(val_df)}")
    print(f"  test : {len(test_df)} rows  {counts(test_df)}")

    def overlap(a, b): return len(set(a["artist"]) & set(b["artist"]))
    print("\nArtist leakage check (should be 0s):")
    print(f"  train ∩ val  = {overlap(train_df, val_df)}")
    print(f"  train ∩ test = {overlap(train_df, test_df)}")
    print(f"  val   ∩ test = {overlap(val_df, test_df)}")


if __name__ == "__main__":
    import sys
    default_args = [
        "--train_size", "0.8",
        "--val_size", "0.1",
        "--test_size", "0.1",
        "--random_state", "42",
        "--out_dir", "data",
        "--require_all_classes",
    ]
    if len(sys.argv) == 3 and sys.argv[1] == "--seed_csv":
        sys.argv.extend(default_args)
    main()
