# Plan: BharatGen Patram on a free private ZeroGPU Space

Status (2026-10-29): **Space created by the user and verified working.** Private Space `sharmithas151005/docubharat-patram`, stage RUNNING on ZeroGPU (runtime reports hardware `zero-a10g`). It was created in the web UI, so no write token was shared.

## Test results (measured, `scripts/check_patram_space.py`)

| Page | Question | Patram answer | Correct | GPU generation | Wall clock |
|---|---|---|---|---|---|
| exam_circular p1 (digital) | Examination fee for regular students? | "The examination fee for regular students is Rs. 1,800 per semester." | ✅ | 2.14 s | 8.6 s |
| exam_circular p1 (digital) | Circular number? | "The circular number is KSU/COE/2026/41." | ✅ | 1.46 s | 6.2 s |
| scanned_library_notice p1 (**image only, no text layer**) | Until what date will the library remain open late? | "The library will remain open late until 15 December 2026." | ✅ | 1.44 s | 5.3 s |

- **Patram's custom code works on torch 2.8.0 + transformers 4.50.3**, so the main risk is cleared.
- Connecting took 5.2 s; the Space was already running.
- **Quota estimate (not measured directly):** ZeroGPU bills GPU time per `@spaces.GPU` call, including attach overhead. At roughly 3–6 s per call, 5 free minutes allow about **50–100 calls/day**.
- Client: `gradio_client==1.14.0`. Installing it downgraded `websockets` 17.3 → 15.0.1, which is within Streamlit's `>=12,<18`; the full test suite still passes.

## Why ZeroGPU (verified 2026-10-29, https://huggingface.co/docs/hub/spaces-zerogpu)

| Item | Detail |
|---|---|
| Who can host | Free accounts with a verified email and **older than 30 days**: up to 2 ZeroGPU Spaces (you confirmed your account qualifies) |
| GPU | Half an NVIDIA RTX Pro 6000 Blackwell, **48 GB VRAM**, BF16-capable. Patram is 8B parameters in BF16 (≈ 16 GB of weights), so it fits without quantisation |
| Daily quota | Free account: **5 GPU-minutes/day** (unauthenticated callers: 2 min). Resets 24 h after first use |
| Constraints | Gradio SDK only; torch **≥ 2.8.0**; Python 3.12.12 or 3.10.13; default 60 s per call (`@spaces.GPU(duration=...)`) |
| Rejected | Free Colab prohibits "web service offerings" and web-UI-driven use on free runtimes, and its T4 lacks BF16 and has only 16 GB. Kaggle has no documented way to serve an API (batch use only) |

## Files (`patram_space/`)

| File | Content |
|---|---|
| `app.py` | Loads Patram exactly as on its model card (`trust_remote_code`, bfloat16, `processor.process`, `generate_from_batch`, prompt `Question: … Answer based on the image.`). Exposes `/ask(image, question, max_new_tokens) → (answer, seconds)`. Caps question length, tokens (≤ 512) and image size (≤ 1600 px) to save quota |
| `requirements.txt` | torch 2.8.0, torchvision 0.23.0, transformers 4.50.3 (as the card), huggingface_hub 0.36.0, accelerate 0.34.2, einops 0.8.1 (as the card), spaces 0.51.3 |
| `README.md` | Space metadata: Gradio **5.50.0**, Python 3.12, Apache-2.0, model link |

Version decisions (checked on PyPI):

- **Gradio 6.x is unusable here:** it needs `huggingface_hub >= 1.16`, but Patram's `transformers==4.50.3` needs `huggingface_hub < 1.0`. Gradio 5.50.0 accepts `>= 0.33.5, < 2.0`, so `huggingface_hub 0.36.0` satisfies all three.
- **torch 2.8.0 instead of the card's 2.7.0** is forced by ZeroGPU. This is the **main risk**: Patram's custom model code has not been tested by BharatGen on torch 2.8.

## Known blocker found while preparing

Your current HF token can list Patram's files but gets **HTTP 403 when downloading them**. Either you haven't accepted Patram's access terms (separate from the BPCC dataset), or the token is fine-grained without "read access to public gated repos". Fix this first (step 1), or the Space cannot load the model.

## Steps

You do steps 1–6 in the browser. I do steps 7–8 after you tell me the Space is running.

1. Open https://huggingface.co/bharatgenai/patram-7b-instruct and accept the access terms.
2. Use a **read** token (classic "Read" type, or fine-grained with *Read access to contents of all public gated repos you can access*). Update `HF_TOKEN` in `.env` if you create a new one.
3. Create the Space at https://huggingface.co/new-space:
   - Name: `docubharat-patram`
   - SDK: **Gradio**
   - Hardware: **ZeroGPU**
   - Visibility: **Private**
   - Licence: apache-2.0
4. In the Space: **Files → Add file → Upload files**, then upload the three files from `patram_space/`. Replace the auto-generated README.
5. **Settings → Variables and secrets → New secret:** name `HF_TOKEN`, value = your read token.
6. Watch the **Logs** tab until it says "Running", or until it fails. If it fails, copy the error text and send it to me. Do not paste tokens.
7. Add `PATRAM_SPACE_ID=<your-username>/docubharat-patram` to `.env`. I then install `gradio_client==1.14.0` (small, pure Python; matches the Space's Gradio) and run `python scripts/check_patram_space.py`. That is **3 calls**: the fee and the circular number from the digital exam circular, and a date from the **scanned** library notice, which tests OCR-free reading.
8. I report: does it build on torch 2.8, are the answers right, and how many GPU-seconds does each call cost (→ calls per day)? **Only if all of that works** do I propose the DocuBharat integration (page images → Patram, current pipeline as fallback) and the evaluation design.

## If it fails

- **Build or import error on torch 2.8:** I'll read the log and try a minimal, documented fix (e.g. a compatible `accelerate`/`transformers` patch release). If Patram's code itself is incompatible, I stop and report.
- **Out of quota during testing:** wait for the 24 h reset; no cost is incurred on a free account.
- **Fallback:** a Kaggle notebook running Patram once over the evaluation pages, saving outputs for the report (no live app use).
