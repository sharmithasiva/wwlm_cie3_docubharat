"""Run the 40-case DocuBharat evaluation and write results + metrics.

Usage:
    python scripts/run_evaluation.py                       # all systems that are configured
    python scripts/run_evaluation.py --systems extractive  # only the no-AI baseline
    python scripts/run_evaluation.py --offline             # score cached outputs only, no API calls

Outputs (data/test_cases/):
    outputs/<system>_<task>.json   raw model outputs (cache; delete a file to re-run that part)
    results.csv                    one row per (case, system): the required per-case table
    field_slots.csv                slot-level field matches
    metrics.json                   all metrics
The written analysis of these results is docs/evaluation_report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src import evaluation as ev  # noqa: E402

RESULTS_DIR = ROOT / "data" / "test_cases"
SYSTEMS = ("extractive", "groq", "patram")


def expected_text(case: dict) -> str:
    if not case["answerable"]:
        return "NOT IN DOCUMENT (system should say so)"
    return " AND ".join(" / ".join(group) for group in case["must"])


def make_clients(systems: list[str], offline: bool):
    clients = {}
    if offline:
        return clients
    if "groq" in systems:
        from src.llm_provider import create_provider

        clients["groq"] = create_provider()
    if "patram" in systems:
        from src.patram_client import create_client

        clients["patram"], reason = create_client()
        if clients["patram"] is None:
            print(f"Patram unavailable: {reason}")
    return clients


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--offline", action="store_true", help="score cached outputs only")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    systems = [s.strip() for s in args.systems.split(",") if s.strip() in SYSTEMS]

    cases = ev.load_cases()
    names = {c["doc"] for c in cases["qa"]} | {c["doc"] for c in cases["fields"]} | {c["doc"] for c in cases["summaries"]}
    docs = ev.load_documents(names)
    clients = make_clients(systems, args.offline)
    stopped: dict[str, str] = {}
    rows, slot_rows = [], []
    metrics: dict = {"systems": {}, "retrieval": {}, "notes": []}

    # ------------------------------------------------------------------ QA (28 cases)
    for system in systems:
        cache = ev.OutputCache(system, "qa")
        for case in cases["qa"]:
            d = docs[case["doc"]]
            if system == "extractive":
                run = lambda c=case, d=d: ev.qa_extractive(c, d)  # noqa: E731
            elif system == "groq" and clients.get("groq") and not d.doc.extraction.needs_ocr:
                run = lambda c=case, d=d: ev.qa_groq(c, d, clients["groq"])  # noqa: E731
            elif system == "groq" and d.doc.extraction.needs_ocr:
                run = lambda: {"found": False, "answer": "", "page": None, "grounded": None, "seconds": 0,  # noqa: E731
                               "error": None, "note": "scanned PDF: no text to send"}
            elif system == "patram" and clients.get("patram") and system not in stopped:
                run = lambda c=case, d=d: ev.qa_patram(c, d, clients["patram"])  # noqa: E731
            else:
                run = None
            try:
                record = cache.get_or_run(case["id"], run, offline=args.offline or run is None)
            except ev.StopSystem as exc:
                stopped[system] = str(exc)
                print(f"[{system}] stopped: {exc}")
                record = None
            if record is None:
                continue
            if system == "patram" and record.get("answer") and not record.get("error"):
                # Patram answers in free text: decide "found" with the current detector, not the
                # value saved at run time (an earlier detector missed "Not explicitly stated").
                record = {**record, "found": not ev.says_not_found(record["answer"])}
            correct, failure = ev.score_qa(case, record)
            rows.append({
                "test_id": case["id"], "task": "qa", "category": case["category"], "document": case["doc"],
                "question_or_task": case["question"], "expected": expected_text(case),
                "support_page": case.get("support_page"), "support_passage": case.get("support_text"),
                "system": system, "actual_output": record["answer"] or ("(not found)" if not record["found"] else ""),
                "page_used": record.get("page"), "grounded": record.get("grounded"),
                "correct": correct, "failure_category": failure, "error": record.get("error"),
            })
            print(f"[{system}] {case['id']}: {'OK ' if correct else 'BAD'} {failure}", flush=True)

    # ------------------------------------------------------------------ fields (6 cases)
    for system in systems:
        cache = ev.OutputCache(system, "fields")
        for case in cases["fields"]:
            d = docs[case["doc"]]
            if system == "extractive":
                got = ev.collect_rule_fields(d.fields)
                output = {k: sorted(map(str, v)) for k, v in got.items()}
            else:
                def run(d=d, system=system):
                    if system == "groq":
                        from src.llm_tasks import llm_extract_fields

                        r = ev.with_rate_limit_retry(lambda: llm_extract_fields(d.doc.pages, clients["groq"]))
                        return {"items": [[i.field, i.text] for i in r.items]}
                    from src.patram_client import PatramUnavailable
                    from src.patram_tasks import patram_fields

                    try:
                        r = patram_fields(clients["patram"], d.pdf, d.doc.pages, d.doc.extraction.pages_processed)
                    except PatramUnavailable as exc:
                        if "quota" in str(exc).lower():
                            raise ev.StopSystem(str(exc)) from exc
                        return {"items": [], "error": str(exc)}
                    return {"items": [[f.field, f.value] for f in r.fields], "raw": [a.text for a in r.answers]}

                available = clients.get(system) is not None and system not in stopped
                try:
                    record = cache.get_or_run(case["id"], run, offline=args.offline or not available)
                except ev.StopSystem as exc:
                    stopped[system] = str(exc)
                    record = None
                except Exception as exc:  # e.g. Groq error: score as empty now, retried on the next run
                    record = {"items": [], "error": f"{type(exc).__name__}: {getattr(exc, 'user_message', exc)}"}
                if record is None:
                    continue
                items = [tuple(i) for i in record["items"]]
                if system == "patram" and record.get("raw"):
                    # Re-parse the saved raw replies with the current parser (no new GPU calls).
                    from src.patram_tasks import page_text, parse_fields

                    items = [(f.field, f.value) for page, raw in enumerate(record["raw"], 1)
                             for f in parse_fields(raw, page, page_text(d.doc.pages, page))]
                got = ev.collect_model_fields(items)
                output = {**record, "items": [list(i) for i in items]}
            slots, unsupported = ev.score_fields(case["expected"], got, d.doc.full_text)
            matched = sum(s["matched"] for s in slots)
            for s in slots:
                slot_rows.append({"test_id": case["id"], "document": case["doc"], "system": system, **s})
            rows.append({
                "test_id": case["id"], "task": "fields", "category": "field_extraction", "document": case["doc"],
                "question_or_task": "Extract title, organisation, reference, key dates, amounts, contacts",
                "expected": json.dumps(case["expected"], ensure_ascii=False), "support_page": "all",
                "support_passage": None, "system": system,
                "actual_output": json.dumps(output, ensure_ascii=False, default=str)[:2000],
                "page_used": None, "grounded": None,
                "correct": matched == len(slots), "failure_category": f"{matched}/{len(slots)} slots exact",
                "error": output.get("error") if isinstance(output, dict) else None,
                "slots_matched": matched, "slots_total": len(slots),
                "unsupported_dates": len(unsupported["dates"]), "unsupported_amounts": len(unsupported["amounts"]),
            })
            print(f"[{system}] {case['id']}: {matched}/{len(slots)} slots", flush=True)

    # ------------------------------------------------------------------ summaries (6 cases)
    for system in systems:
        cache = ev.OutputCache(system, "summary")
        for case in cases["summaries"]:
            d = docs[case["doc"]]
            if system == "extractive":
                text = ev.rule_summary_text(d)
            else:
                def run(d=d, system=system):
                    if system == "groq":
                        from src.llm_tasks import llm_summarize

                        s = ev.with_rate_limit_retry(lambda: llm_summarize(d.doc.pages, clients["groq"]))
                        return {"text": "\n".join([s.purpose, *(i.text for i in s.all_items)])}
                    from src.patram_client import PatramUnavailable
                    from src.patram_tasks import patram_summary

                    try:
                        r = patram_summary(clients["patram"], d.pdf, d.doc.extraction.pages_processed)
                    except PatramUnavailable as exc:
                        if "quota" in str(exc).lower():
                            raise ev.StopSystem(str(exc)) from exc
                        return {"text": "", "error": str(exc)}
                    return {"text": "\n".join(a.text for a in r.answers)}

                available = clients.get(system) is not None and system not in stopped
                try:
                    record = cache.get_or_run(case["id"], run, offline=args.offline or not available)
                except ev.StopSystem as exc:
                    stopped[system] = str(exc)
                    record = None
                except Exception as exc:
                    record = {"text": "", "error": f"{type(exc).__name__}: {getattr(exc, 'user_message', exc)}"}
                if record is None:
                    continue
                text = record["text"]
            score = ev.score_summary(case, text, d.doc.full_text)
            covered = sum(f["covered"] for f in score["facts"])
            rows.append({
                "test_id": case["id"], "task": "summary", "category": "summarisation", "document": case["doc"],
                "question_or_task": "Summarise the document",
                "expected": "; ".join(f["fact"] for f in case["facts"]), "support_page": "all", "support_passage": None,
                "system": system, "actual_output": text[:2000], "page_used": None, "grounded": None,
                "correct": covered == len(score["facts"]) and not score["unsupported_numbers"],
                "failure_category": (f"{covered}/{len(score['facts'])} facts"
                                     + (f"; unsupported numbers {score['unsupported_numbers']}" if score["unsupported_numbers"] else "")),
                "error": None, "facts_covered": covered, "facts_total": len(score["facts"]),
                "unsupported_numbers": len(score["unsupported_numbers"]),
            })
            print(f"[{system}] {case['id']}: {covered}/{len(score['facts'])} facts, unsupported {score['unsupported_numbers']}", flush=True)

    # ------------------------------------------------------------------ retrieval (no model calls)
    for label, make_index in (("default_500w", lambda d: d.doc.index), ("small_80w", ev.small_chunk_index)):
        hits = defaultdict(list)
        for case in cases["qa"]:
            d = docs[case["doc"]]
            if d.doc.extraction.needs_ocr:
                continue
            index = make_index(d)
            for k in (1, 3):
                h = ev.retrieval_hit(case, index, k)
                if h is not None:
                    hits[k].append(h)
        metrics["retrieval"][label] = {f"hit@{k}": round(sum(v) / len(v), 3) for k, v in hits.items()} | {"n": len(hits[1])}

    # ------------------------------------------------------------------ metrics
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results.csv", index=False, encoding="utf-8")
    pd.DataFrame(slot_rows).to_csv(RESULTS_DIR / "field_slots.csv", index=False, encoding="utf-8")
    for system in systems:
        part = df[df.system == system]
        qa, fields, summ = part[part.task == "qa"], part[part.task == "fields"], part[part.task == "summary"]
        m: dict = {}
        if len(qa):
            answerable = qa[qa.category != "absent"]
            absent = qa[qa.category == "absent"]
            found = qa[(qa.actual_output != "(not found)") & qa.error.isna()]
            grounded_known = found[found.grounded.notna()]
            m["qa"] = {
                "n": len(qa), "accuracy": round(qa.correct.mean(), 3),
                "answerable_accuracy": round(answerable.correct.mean(), 3) if len(answerable) else None,
                "absent_correct_abstention": round(absent.correct.mean(), 3) if len(absent) else None,
                "by_category": {k: round(v, 3) for k, v in qa.groupby("category").correct.mean().items()},
                "failures": qa.failure_category.value_counts().to_dict(),
                "groundedness": round(grounded_known.grounded.astype(bool).mean(), 3) if len(grounded_known) else None,
                "groundedness_n": len(grounded_known),
            }
        if len(fields):
            m["fields"] = {
                "docs": len(fields), "slot_exact_match_accuracy": round(fields.slots_matched.sum() / fields.slots_total.sum(), 3),
                "slots_matched": int(fields.slots_matched.sum()), "slots_total": int(fields.slots_total.sum()),
                "unsupported_dates": int(fields.unsupported_dates.sum()), "unsupported_amounts": int(fields.unsupported_amounts.sum()),
            }
            slots = pd.DataFrame([r for r in slot_rows if r["system"] == system])
            m["fields"]["by_slot"] = {k: round(v, 3) for k, v in slots.groupby("slot").matched.mean().items()}
            m["fields"]["slot_relaxed_accuracy"] = round(slots.relaxed.mean(), 3)
            m["fields"]["by_slot_relaxed"] = {k: round(v, 3) for k, v in slots.groupby("slot").relaxed.mean().items()}
        if len(summ):
            m["summary"] = {
                "docs": len(summ), "fact_coverage": round(summ.facts_covered.sum() / summ.facts_total.sum(), 3),
                "summaries_with_unsupported_numbers": int((summ.unsupported_numbers > 0).sum()),
            }
        metrics["systems"][system] = m
    if stopped:
        metrics["notes"].append({"stopped": stopped})
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
