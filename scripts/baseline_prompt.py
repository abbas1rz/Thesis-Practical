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

    {"lyrics": "Sunlight on the dashboard, windows down, we’re laughing at the rain.", "label": "happy"},
    {"lyrics": "The chorus lifts me higher, we dance like sparks in July.", "label": "happy"},
    {"lyrics": "I’m madly in love, smiling every time you say my name.", "label": "happy"},
    {"lyrics": "Butterflies in my chest, I can’t stop grinning tonight.", "label": "happy"},


    {"lyrics": "Another night alone, the room is heavy with the words we didn’t say.", "label": "sad"},
    {"lyrics": "The letters crumple in my hands; quiet tears salt the floor.", "label": "sad"},
    {"lyrics": "I’m tired of being strong; the streets feel cold and empty.", "label": "sad"},
    {"lyrics": "It’s late and I still need you, even though you’re gone.", "label": "sad"},
    {"lyrics": "Tongue-tied and quiet, my heart sinks when I try to speak.", "label": "sad"},


    {"lyrics": "You betrayed me, I’m shouting at the walls till they crack.", "label": "angry"},
    {"lyrics": "I slam the door, my pulse is fire; everything burns tonight.", "label": "angry"},
    {"lyrics": "Say it to my face — I’m done swallowing lies.", "label": "angry"},
    {"lyrics": "Don’t tell me to calm down, you crossed the line.", "label": "angry"},


    {"lyrics": "Waves hush the shore, my thoughts drift like clouds.", "label": "relaxed"},
    {"lyrics": "Soft guitar under moonlight, I breathe out and let it go.", "label": "relaxed"},
    {"lyrics": "Pair of worn boots and a slow drive with the radio low.", "label": "relaxed"},
    {"lyrics": "Fireplace glowing, snow outside, I’m calm at home.", "label": "relaxed"},
    {"lyrics": "We sway slow under warm trees, letting the night breathe.", "label": "relaxed"},
    {"lyrics": "Folding chair by the water, watching the evening settle.", "label": "relaxed"},
]

SYSTEM = (
    "You are an expert emotion classifier for short song lyrics.\n"
    "Choose exactly ONE label from: 'happy', 'sad', 'angry', 'relaxed'.\n"
    "If the text is ambiguous, pick the best overall fit.\n"
    "Guidance:\n"
    "- 'angry' for aggressive, violent, hostile, loud, heated language (shout, rage, burn, fight). "
    "Words like 'madly' in romance do NOT mean angry.\n"
    "- 'relaxed' for calm, serene, slow, peaceful, gentle moods (waves, breeze, quiet, soft), including easygoing/cozy/mellow vibes.\n"
    "- 'happy' for upbeat, joyful, celebratory moods (laugh, dance, party, bright).\n"
    "- 'sad' for sorrowful, lonely, grieving moods (tears, alone, ache, heartbroken).\n"
    "Output STRICTLY a single JSON object like: {\"label\": \"happy\"} — no extra text.\n\n"
    "Examples:\n" +
    "\n".join([f"Lyrics: {ex['lyrics']}\nAnswer: {{\"label\": \"{ex['label']}\"}}" for ex in FEWSHOTS]) +
    "\n\nNow classify the next example."
)

USER_TMPL = "Lyrics:\n{lyrics}\n"


def extract_label(text: str):
    m = re.search(r'"label"\s*:\s*"([^"]+)"', text, re.I)
    if m:
        lbl = m.group(1).lower().strip()
        return lbl if lbl in ALLOWED else None
    m = re.search(r"\b(happy|sad|angry|relaxed)\b", text.lower())
    return m.group(1) if m else None

def normalize_lyrics(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    np.random.seed(args.seed)

    df = pd.read_csv(args.test_csv)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    if args.sample > 0:
        df = df.sample(min(args.sample, len(df)), random_state=args.seed)

    import torch

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
        )

    gen = pipeline("text-generation", model=model, tokenizer=tok)

    y_true, y_pred = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        lyrics = normalize_lyrics(row["lyric_text"])[:3000]

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + SYSTEM +
            "\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n" + USER_TMPL.format(lyrics=lyrics) +
            "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        out = gen(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            return_full_text=False,
        )[0]["generated_text"]

        label = extract_label(out)
        if label is None:
            label = "happy"

        y_true.append(row["label"])
        y_pred.append(label)


    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)


    with open(reports_dir / "baseline_prompt_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "f1_macro": f1m, "report": rep}, f, indent=2)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS],
    )
    cm_df.to_csv(reports_dir / "baseline_confusion_matrix.csv")


    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))
    print("Accuracy:", acc)
    print("Macro-F1:", f1m)

if __name__ == "__main__":
    main()
