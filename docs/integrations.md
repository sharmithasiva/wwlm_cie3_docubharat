# External model integrations

DocuBharat's core (PDF extraction, BM25 retrieval, rule-based fields, extractive answers) runs locally with no model and no network access. Two optional integrations exist: a hosted LLM (Groq) and AI4Bharat IndicTrans2 translation running locally. Each can be switched off, and the core app keeps working without them.

Facts are marked **Verified** (read on the official page on 2026-10-29), **Observed** (seen in our own live requests) or **Unverified**.

## 1. Hosted LLM: Groq, `openai/gpt-oss-20b`

| Item | Detail |
|---|---|
| Official source | https://console.groq.com/docs/api-reference, https://console.groq.com/docs/models (Verified) |
| Model | `openai/gpt-oss-20b`, OpenAI's open-weight model, listed by Groq as a production model with a 131,072-token context (Verified) |
| Version | Groq does not expose a revision; the model id returned in each response is logged with the result (Observed: `openai/gpt-oss-20b`) |
| Inference method | HTTPS `POST https://api.groq.com/openai/v1/chat/completions` (OpenAI-compatible), `temperature=0`, `reasoning_effort=low`, strict `json_schema` structured output (Verified) |
| Authentication | `Authorization: Bearer <LLM_API_KEY>` from `.env` (Verified) |
| Cost and limits | Free plan without a card. Per-model limits are shown in the Groq console (`/settings/limits`); the public page lists Developer-plan limits only (e.g. 30 RPM / 8K TPM for this model), so free-plan numbers are **Unverified** |
| Data handling | Groq does not retain inference data by default. It may log inputs and outputs for up to 30 days for abuse monitoring, and you can opt out with Zero Data Retention (Verified, https://console.groq.com/docs/your-data) |
| Licence | gpt-oss weights are Apache 2.0 (per OpenAI; not re-verified here). Use of the API is governed by Groq's terms |
| Languages | Prompts and answers in English |
| Resource requirements (local) | None beyond `requests`; inference runs on Groq's servers |
| What is sent | Per question: the top 3 BM25 passages (≤ 9,000 characters) and the question. AI summary and AI fields: at most 9,000 characters of document text, only on a button click |
| Fallback | No key, any error, a timeout or a rate limit leads to an extractive answer with a plain-language notice. Retries are bounded (2), and waits longer than 10 s are reported instead of blocking |
| Grounding | The model must quote the passages; every quote is checked verbatim against the document (✅/⚠️ in the UI) |

## 2. AI4Bharat IndicTrans2: local translation of answers (CTranslate2, CPU)

This is DocuBharat's **genuine AI4Bharat integration**. It runs entirely on this laptop: no account, API or internet is needed at run time. (An earlier Bhashini-based client was removed at the user's request and never used live.)

| Item | Detail |
|---|---|
| Model | IndicTrans2 **distilled En→Indic** (`en-indic-dist`, the model published on HF as `ai4bharat/indictrans2-en-indic-dist-200M`), official **int8 CTranslate2** export (`ct2_int8_model`) |
| Organisation / source | AI4Bharat, https://github.com/AI4Bharat/IndicTrans2 (README "Download Models" table). Archive: `https://huggingface.co/datasets/ai4bharat/BPCC/resolve/main/additional/en-indic-dist.tar.gz` (gated with auto-approval; HF read token needed for the one-time download only) |
| Licence | MIT (model checkpoints and inference code) |
| Version | Archive streamed on 2026-10-29. Inference code vendored unmodified from commit `4f08e39cc6bf13cd62e2445dc725f22bff1a9219` (`third_party/indictrans2`) |
| Inference method | AI4Bharat's `inference.engine.Model(..., model_type="ctranslate2", device="cpu")` → `batch_translate(sentences, "eng_Latn", <FLORES code>)`. CTranslate2 4.8.4 is created with `compute_type="int8"`. Implemented in `src/indic_translator.py`, called from the Translate button in `app.py` |
| Authentication | None at run time. `HF_TOKEN` is used only by `scripts/setup_indictrans2.py` |
| Languages offered | Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu (the model covers all 22 scheduled languages) |
| Dependencies | `requirements-translation.txt` + `requirements-translation-nodeps.txt`: ctranslate2, sentencepiece, sacremoses, mosestokenizer, nltk, regex, indic-nlp-library, morfessor. **No PyTorch, no compiler, no GPU.** The venv grew by 86 MB |
| Disk | Streamed 2.78 GB; kept **218 MB** (`model.bin` 212 MB + vocabularies). The fairseq checkpoint (2.79 GB) and fp16 export (428 MB) are never stored |
| Memory (measured) | Stand-alone: 28 MB Python baseline → **453 MB peak**. Inside the Streamlit app (document loaded): 162 MB → **470 MB, peak 479 MB** |
| Speed (measured, CPU) | Model load **1.2 s**; **0.34–0.50 s per sentence**; 3-sentence answer 0.8 s; first click in the app including load 1.9 s |
| What is translated | Only the answer the user clicks Translate on. Quotes and source passages stay in English |
| Safeguards | English answer always shown; `missing_numbers` warns if a date or amount is lost; output is Markdown-escaped |
| Fallback | Packages missing, model not downloaded, or `TRANSLATION_ENABLED=false` → the sidebar explains why and nothing else changes. Load or translation errors → warning on that answer, English answer kept. Groq and all extractive features are independent of it |

**Observed quality (informal, not the Step 8 evaluation):** Hindi and Bengali kept "Rs. 1,800", "7 November 2026" and "75%" exactly. Tamil wrote the date as "நவம்பர் 7,2026": correct, but it initially tripped the number check, which was fixed to accept it. One Tamil sentence shifted "must have 75% attendance to be eligible" towards "eligible students must have 75% attendance", a meaning shift the number check cannot catch. This is why the English answer always stays visible.

Kept from the removed Bhashini client: `src/translation_checks.py` (number check, sentence segmentation), now used by this integration.

## 3. BharatGen Patram: document VQA on a private ZeroGPU Space (integrated)

This is DocuBharat's **genuine BharatGen integration** and the default answer engine when configured.

| Item | Detail |
|---|---|
| Model | `bharatgenai/patram-7b-instruct`, revision `patram1.1` (vision-language document model, ~8B parameters, BF16) |
| Organisation / licence | BharatGen; Apache 2.0; gated (access terms accepted by the user) |
| Where it runs | User's **private** Hugging Face ZeroGPU Space `sharmithas151005/docubharat-patram` (free tier). Space code in `patram_space/` follows the model card; the one deviation is torch 2.8.0 instead of 2.7.0, which ZeroGPU requires |
| Client | `src/patram_client.py` (`gradio_client==1.14.0`, endpoint `/ask`); tasks in `src/patram_tasks.py`; UI in `app.py` |
| What is sent | A PNG of **one page** (110 dpi) + the question. Summary and fields send up to `PATRAM_MAX_PAGES` (default 2) pages, only on a button click. Answers are cached per document/page/question |
| Page choice | The page of the best BM25 match (page 1 for scanned PDFs), or a page the user selects |
| Grounding | Patram returns no quotes, so every number in its answer is checked against the page's extracted text. Scanned pages are marked "not verifiable". The page image is shown with each answer |
| Scanned PDFs | Now usable: Patram reads the page image. Text-based features are skipped and the UI says so |
| Fallback | Space asleep, quota used up, timeout or auth error → plain message and the Groq or extractive answer instead. `PATRAM_ENABLED=false` turns it off |
| Licence of outputs | AI-generated; always labelled "BharatGen Patram" |

### Measured (live, 2026-10-29, synthetic sample documents)

| Task | Result | Time |
|---|---|---|
| Fee / circular number / scanned-page date (check script) | 3/3 correct | 1.4–2.1 s GPU, 5–9 s wall |
| In-app question "late fine and until when?" (auto page 1) | "The late fine is Rs. 500 and it is accepted until 14 November 2026." All 3 numbers found in the page text | 2.1 s GPU, 14.3 s wall (first call includes connecting) |
| In-app summary, 2 pages | Accurate; all 9 + 6 numbers found in the page text | 20.5 s for 2 calls |
| In-app fields, 2 pages | Patram ignored the one-item-per-line instruction and replied on one comma-separated line; the parser was extended and now reads its real reply as 11 labelled fields. **Content errors seen:** it gave the circular number as the *title*, and "All Heads of Departments and students…" as *eligibility* | 19.9 s for 2 calls |
| Scanned page question in-app | Answered from the image; marked not verifiable | 1.6 s GPU, 6.4 s wall |

**Quota:** free ZeroGPU accounts get 5 GPU-minutes/day. At roughly 3–6 s of GPU time per call (estimate), that is about 50–100 calls a day.
**Incident:** one test run before the fix called the real Space (the test fixture did not yet disable Patram) and used roughly 6 calls. Tests now set `PATRAM_ENABLED=false` and use a fake client.
