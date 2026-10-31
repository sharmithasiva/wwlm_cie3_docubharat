"""Evaluate local AI4Bharat IndicTrans2 translation of DocuBharat answers.

Usage:  python scripts/eval_translation.py

Automatic metrics (no reference translations exist for these synthetic documents):
- number preservation: every number (date, amount, %) in the English sentence must
  appear in the translation (Indic digits count, see src/translation_checks.py);
- target-script share: fraction of letters written in the target language's script;
- latency per sentence on this CPU.
Adequacy/fluency need a fluent human reader: outputs are saved for manual review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.indic_translator import LANGUAGES, IndicTrans2Translator, TranslationConfig, status  # noqa: E402

SENTENCES = [
    "The examination fee for regular students is Rs. 1,800 per semester.",
    "The last date for payment of the examination fee without fine is 7 November 2026.",
    "Students must have a minimum attendance of 75% in each course to be eligible to appear for the examinations.",
    "The total intake is 180 seats across all four programmes.",
    "The DIT Entrance Test (DET) will be held on 7 March 2027 at Hyderabad, Bengaluru and Chennai.",
    "Selected students will receive Rs. 25,000 per annum for the duration of the course, up to a maximum of four years.",
    "The annual family income from all sources must not exceed Rs. 2,50,000.",
    "The total payable for a hostel resident is Rs. 1,12,000 per semester.",
    "Registration will be open from 1 December 2026 to 20 December 2026.",
    "Students should report to the examination hall 30 minutes before the start of each examination.",
]
TARGETS = {"hin_Deva": (0x0900, 0x097F), "tam_Taml": (0x0B80, 0x0BFF), "ben_Beng": (0x0980, 0x09FF)}
OUT = ROOT / "data" / "test_cases"


def script_share(text: str, block: tuple[int, int]) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(block[0] <= ord(ch) <= block[1] for ch in letters) / len(letters)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = TranslationConfig.from_env()
    available, reason = status(config)
    if not available:
        print(f"IndicTrans2 not available: {reason}")
        return 1
    translator = IndicTrans2Translator(config)
    rows = []
    for lang, block in TARGETS.items():
        for i, sentence in enumerate(SENTENCES, 1):
            t = translator.translate(sentence, lang)
            rows.append({
                "id": f"T{i:02d}-{lang}", "language": LANGUAGES[lang], "source": sentence, "translation": t.text,
                "numbers_preserved": not t.missing_numbers, "missing_numbers": ", ".join(t.missing_numbers),
                "target_script_share": round(script_share(t.text, block), 3), "seconds": round(t.seconds, 3),
            })
            print(f"[{lang}] T{i:02d} {'OK ' if not t.missing_numbers else 'LOST ' + str(t.missing_numbers)} "
                  f"{t.seconds:.2f}s :: {t.text}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "translation_results.csv", index=False, encoding="utf-8")
    metrics = {
        "model": "AI4Bharat IndicTrans2 distilled En->Indic (CTranslate2 int8, CPU)",
        "load_seconds": round(translator.load_seconds or 0, 2),
        "n": len(df),
        "number_preservation": round(df.numbers_preserved.mean(), 3),
        "by_language": {
            lang: {"number_preservation": round(g.numbers_preserved.mean(), 3),
                   "mean_target_script_share": round(g.target_script_share.mean(), 3),
                   "mean_seconds": round(g.seconds.mean(), 3)}
            for lang, g in df.groupby("language")
        },
    }
    (OUT / "translation_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
