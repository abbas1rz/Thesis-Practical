import re
import ast
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABELS = ["happy", "sad", "angry", "relaxed"]
ALLOWED = set(LABELS)

FEWSHOTS = [
    # happy
    {"lyrics": "Sunlight on the dashboard, windows down, we’re laughing at the rain.", "label": "happy"},
    {"lyrics": "The chorus lifts me higher, we dance like sparks in July.", "label": "happy"},
    # sad
    {"lyrics": "Another night alone, the room is heavy with the words we didn’t say.", "label": "sad"},
    {"lyrics": "The letters crumple in my hands; quiet tears salt the floor.", "label": "sad"},
    # angry 
    {"lyrics": "You betrayed me, I’m shouting at the walls till they crack.", "label": "angry"},
    {"lyrics": "I slam the door, my pulse is fire; everything burns tonight.", "label": "angry"},
    # relaxed
    {"lyrics": "Waves hush the shore, my thoughts drift like clouds.", "label": "relaxed"},
    {"lyrics": "Soft guitar under moonlight, I breathe out and let it go.", "label": "relaxed"},
]

SYSTEM = (
    "You are an expert emotion classifier for short song lyrics.\n"
    "Choose exactly ONE label from: 'happy', 'sad', 'angry', 'relaxed'.\n"
    "If the text is ambiguous, pick the best overall fit.\n"
    "Guidance:\n"
    "- 'angry' for aggressive, violent, hostile, loud, heated language (shout, rage, burn, fight).\n"
    "- 'relaxed' for calm, serene, slow, peaceful, gentle moods (waves, breeze, quiet, soft).\n"
    "- 'happy' for upbeat, joyful, celebratory moods (laugh, dance, party, bright).\n"
    "- 'sad' for sorrowful, lonely, grieving moods (tears, alone, ache, heartbroken).\n"
    "These four are equally valid; do not avoid 'angry' or 'relaxed' when cues appear.\n"
    "Output STRICTLY a single JSON object like: {\"label\": \"happy\"} — no extra text.\n\n"
    "Examples:\n" +
    "\n".join([f"Lyrics: {ex['lyrics']}\nAnswer: {{\"label\": \"{ex['label']}\"}}" for ex in FEWSHOTS]) +
    "\n\nNow classify the next example."
)

USER_TMPL = "Lyrics:\n{lyrics}\n"

def extract_json(text: str):
    """Grab first {...} block and parse as JSON/py-literal."""
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None

def normalize_label(lbl: str):
    """Map synonyms -> labels; return None if out-of-set."""
    if not isinstance(lbl, str):
        return None
    x = lbl.strip().lower()

    syn = {
        # happy
        "joy": "happy", "happiness": "happy", "joyful": "happy", "cheerful": "happy",
        "delighted": "happy", "elated": "happy", "glad": "happy", "ecstatic": "happy",
        # sad
        "sadness": "sad", "melancholy": "sad", "blue": "sad", "depressed": "sad",
        "down": "sad", "heartbroken": "sad", "sorrowful": "sad",
        # angry
        "anger": "angry", "mad": "angry", "furious": "angry", "rage": "angry",
        "irate": "angry", "wrath": "angry", "enraged": "angry", "hostile": "angry",
        # relaxed
        "calm": "relaxed", "chill": "relaxed", "chilled": "relaxed", "peaceful": "relaxed",
        "serene": "relaxed", "tranquil": "relaxed", "soothing": "relaxed",
        "laid-back": "relaxed", "laid back": "relaxed", "easygoing": "relaxed", "gentle": "relaxed",
    }
    x = syn.get(x, x)
    return x if x in ALLOWED else None


KEYWORD_FALLBACK = {
    "angry":  ["rage","angry","anger","furious","mad","scream","shout","yell","fight","punch","burn","war","blood","hate","violent"],
    "sad":    ["sad","tears","cry","wept","alone","lonely","broken","empty","blue","hurt","ache","sorrow","grief","loss"],
    "relaxed":["calm","relaxed","serene","peaceful","quiet","breeze","tide","slow","soothing","chill","laid back","laid-back","float","gentle","soft","lullaby"],
    "happy":  ["happy","smile","laugh","joy","joyful","bright","sun","celebrate","party","dance","cheerful","bliss"],
}

def keyword_guess(text: str):
    t = text.lower()
    scores = {k:0 for k in KEYWORD_FALLBACK}
    for lbl, kws in KEYWORD_FALLBACK.items():
        for w in kws:
            if w in t:
                scores[lbl] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def normalize_lyrics(s: str) -> str:
    """Light cleanup to stabilize prompting."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", " ")            
    s = re.sub(r"[ \t]+", " ", s)          
    s = re.sub(r"\s*\n\s*", "\n", s)       
    s = s.strip()
    return s

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--model", type=str, required=True,
                    help="e.g., meta-llama/Llama-3.2-3B-Instruct or meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--sample", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.35)   
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=int, default=0, help="GPU id (0=first GPU). Ignored when using device_map=auto.")
    ap.add_argument("--load_in_4bit", action="store_true", help="Load model with 4-bit quantization (bitsandbytes).")
    args = ap.parse_args()

    np.random.seed(args.seed)

    df = pd.read_csv(args.test_csv)
    if "label" not in df.columns or "lyric_text" not in df.columns:
        raise ValueError("CSV must contain 'lyric_text' and 'label' columns.")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    bad = sorted(set(df["label"]) - ALLOWED)
    if bad:
        raise ValueError(f"Found unexpected dataset labels: {bad}")

    before = len(df)
    df = df.dropna(subset=["lyric_text"]).reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with empty lyric_text.")

    if args.sample and args.sample > 0:
        df = df.sample(min(args.sample, len(df)), random_state=args.seed).reset_index(drop=True)

    import torch

    def build_generator(model_id: str, load_in_4bit: bool, device: int):
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",  
            )
            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
            )
            return tok, gen

        gen = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tok,
            device_map="auto",     
            torch_dtype="auto"
        )
        return tok, gen

    tok, gen = build_generator(args.model, args.load_in_4bit, args.device)

    def generate_fn(prompt: str) -> str:
        out = gen(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tok.eos_token_id,
        )[0]["generated_text"]
        return out

    y_true, y_pred = [], []
    preds_rows = []
    debug_lines = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        lyrics_raw = str(row.get("lyric_text", ""))
        lyrics = normalize_lyrics(lyrics_raw)[:3000]

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + SYSTEM +
            "\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n" + USER_TMPL.format(lyrics=lyrics) +
            "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        out_text = generate_fn(prompt)
        assistant_segment = out_text[len(prompt):]

        js = extract_json(assistant_segment)
        label = None
        if isinstance(js, dict):
            label = normalize_label(js.get("label"))

        if label is None:
            kg = keyword_guess(lyrics)
            if kg is not None:
                label = kg
            else:
                label = "relaxed" if any(
                    w in lyrics.lower() for w in ["calm","serene","peaceful","slow","soothing","quiet","gentle","soft"]
                ) else "happy"

        true_lbl = str(row.get("label", "")).strip().lower()
        y_true.append(true_lbl)
        y_pred.append(label)
        preds_rows.append({"lyrics": lyrics_raw, "true": true_lbl, "pred": label})

        if true_lbl in {"angry", "relaxed"}:
            debug_lines.append(
                "\n---\n"
                f"TRUE: {true_lbl}\n"
                f"PRED: {label}\n"
                f"LYRICS:\n{lyrics}\n"
                f"ASSISTANT_RAW:\n{assistant_segment}\n"
            )

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "baseline_prompt_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "f1_macro": f1m, "report": rep}, f, indent=2)

    pd.DataFrame(preds_rows).to_csv(reports_dir / "baseline_preds.csv", index=False)

    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    cm_df.to_csv(reports_dir / "baseline_confusion_matrix.csv")

    if debug_lines:
        with open(reports_dir / "debug_samples.txt", "w", encoding="utf-8") as f:
            f.writelines(debug_lines)

    print(json.dumps(
        {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1m, 4),
            "per_class_f1": {k: round(v.get("f1-score", 0.0), 4) for k, v in rep.items() if k in LABELS}
        },
        indent=2
    ))

if __name__ == "__main__":
    main()
