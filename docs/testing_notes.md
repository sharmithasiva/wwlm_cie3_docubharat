# Testing notes

Last run: 2026-10-31, Windows 11, Python 3.12.10, CPU only. **226 passed** (`python -m pytest`, ~27 s). Tests use no API keys, no GPU quota and no network:

- `tests/test_app.py` forces `PATRAM_ENABLED=false`, empties the keys and uses fake Patram, Groq and translator clients.
- The one real-model test (`test_indic_translator.py::test_real_model_translates_to_hindi`) runs only if the local IndicTrans2 model has been downloaded, and is skipped otherwise.

## Coverage of the required areas

| Area | Tests |
|---|---|
| PDF extraction; empty, malformed, encrypted, non-PDF, oversized, truncated files; scanned pages | `test_pdf_processor.py`, `test_robustness.py` |
| Cleaning, chunking, page mapping, sentence splitting | `test_chunker.py`, `test_robustness.py` |
| Retrieval relevance, Indic-script tokens | `test_retrieval.py`, `test_robustness.py` |
| Rule-based field extraction and summary | `test_field_extractor.py` |
| Unanswerable questions, source evidence, page citations | `test_qa_engine.py` |
| Hosted LLM: missing key, 401/403, 413, 429 (short and long `retry-after`), timeouts, network errors, 5xx, malformed or truncated replies, no secret in errors or logs | `test_llm_provider.py` |
| Grounded LLM tasks: verified/fabricated quotes, passage budget, delimiter injection | `test_llm_tasks.py` |
| BharatGen Patram client: page rendering, request shape, quota/auth/timeout/app errors, **waking-Space retry**, empty replies, number grounding, field parser on **real Patram replies** | `test_patram.py` |
| AI4Bharat IndicTrans2 wrapper: availability, int8 folder choice, segmentation, lost-number detection, load/translate failures | `test_indic_translator.py`, `test_translation_checks.py` |
| Streamlit UI run headlessly: start screen, sample loading, engines, fallbacks, scanned PDFs, translation on demand, caching, question box reset | `test_app.py` |
| Evaluation set integrity (support text on the stated page, expected values present) and scorer | `test_evaluation.py` |
| Markdown escaping, configuration parsing, pathological inputs (bounded time), 150-page limit | `test_robustness.py` |

## Bugs found by testing, live runs and review (fixed, each with a regression test)

| Bug | Found by |
|---|---|
| False reference numbers ("…notice. No payments…") | code review |
| "Rs" matched inside "hrs" | code review |
| 20,000-digit page took 19.7 s (regex backtracking) | adversarial test |
| Header/footer removal deleted repeated table values | code review |
| "Room 5." not split as a sentence | code review |
| Single-letter highlight terms | code review |
| Non-breaking hyphens broke quote verification (Groq) | live Groq run |
| Tamil "7,2026" wrongly flagged as missing numbers | live IndicTrans2 run |
| App tests called the real Patram Space (used GPU quota) | slow test run |
| Patram field replies in unexpected formats (one line, `;`, `.`, unlabelled) | live Patram run / evaluation |
| "Not explicitly stated" not recognised as abstention | evaluation review |
| Transient errors cached as final evaluation results | evaluation review |
| Upload `WriteTimeout` while the Space was waking up | live demo |
| Question box kept the previous document's question | live demo |

## Performance (measured)

| Scenario | Result |
|---|---|
| 100-page PDF: extract → clean → chunk → index | 0.84 s |
| ~50,000 words: fields + summary | 0.65 s |
| Extractive QA | ~3 ms/question |
| App process (no translation loaded) | ~87 MB (166 MB peak) |
| IndicTrans2 load / per sentence / app with model | 1.2–1.4 s / 0.3–0.7 s / 470–479 MB |
| Patram per question (Space awake) | 1.4–2.6 s GPU, 5–15 s wall clock; first call after sleep up to ~45 s |

## Not covered automatically

- Uploading through the browser's native file picker (the same code path is covered via the sample loader).
- Visual layout (checked manually; see `docs/screenshots/`).
- Live Groq/Patram behaviour. It is checked by `scripts/check_llm.py`, `scripts/check_patram_space.py` and the evaluation, not by unit tests.
