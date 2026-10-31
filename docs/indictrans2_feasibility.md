# Feasibility: local AI4Bharat IndicTrans2 translation on this laptop

Prepared 2026-10-29 **before** any download; the plan was then approved and carried out the same day. **Measured outcomes are in the 'Results after approval' section at the end and in docs/integrations.md.**

Labels: **Verified** = read in an official source or checked on PyPI/HF metadata today. **Measured** = measured on this laptop. **Estimate** = my calculation, to be replaced by measurements after approval. **Unverified** = not confirmable without the download.

## 1. Exact model

| Item | Detail | Status |
|---|---|---|
| Model | IndicTrans2 **distilled En→Indic** (`en-indic-dist`), the same model as `ai4bharat/indictrans2-en-indic-dist-200M` on HF. HF metadata lists it at about 0.3B parameters including embeddings | Verified |
| Organisation | AI4Bharat (IIT Madras) | Verified |
| Official source | https://github.com/AI4Bharat/IndicTrans2 ("Download Models" table, Distilled En-Indic row) | Verified |
| Artifact | `https://huggingface.co/datasets/ai4bharat/BPCC/resolve/main/additional/en-indic-dist.tar.gz`, linked from the official README | Verified |
| Access | The `ai4bharat/BPCC` dataset is gated (`gated: auto`). A free HF account, accepting the terms and a read token are required. No Bhashini account | Verified |
| Licence | Model checkpoints: **MIT** (README licence table). Inference code: MIT | Verified |
| Paper | Gala et al., *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages*, TMLR 2023, https://arxiv.org/abs/2305.16307 | Verified |

## 2. Windows compatibility (CPU only, no compiler)

| Dependency | Version / wheel | Status |
|---|---|---|
| `ctranslate2` | 4.8.4, `cp312-cp312-win_amd64` wheel (19.4 MB). Windows x64 wheels are official. CUDA is needed only for GPU use. The Visual C++ runtime is required (usually already present) | Verified (PyPI + CT2 install docs) |
| `sentencepiece` | 0.2.2 `cp312 win_amd64` wheel (1.2 MB) | Verified |
| `regex` | `cp312 win_amd64` wheel | Verified |
| `indic-nlp-library` 0.92, `sacremoses` 0.2.0, `mosestokenizer` 1.0.0, `nltk` 3.10.4 | pure-Python wheels | Verified |
| PyTorch / transformers / IndicTransToolkit / fairseq | **not needed**: the official `inference/engine.py` imports `ctranslate2` only on its CT2 path and never imports torch or IndicTransToolkit | Verified (source read) |
| Perl (used by `mosestokenizer`'s sentence splitter) | **not needed**: only `translate_paragraph` calls it; we call `batch_translate` with sentences split by DocuBharat itself | Verified (source read) |
| Microsoft C++ Build Tools | not needed: every package above has a binary wheel | Verified |

All checked with `pip download --only-binary=:all:` to a temp folder. Nothing was installed into the project environment.

## 3. Memory

| Component | Value | Status |
|---|---|---|
| Current Streamlit app (server process) | **87 MB now, 166 MB peak** after Groq answers and summaries | **Measured** |
| Free RAM on the laptop right now | 5.9 GB of 16.4 GB (other programs use the rest) | **Measured** |
| PyTorch | 0. Not used | Verified |
| CT2 model weights in RAM, loaded with `compute_type="int8"` | ≈ 0.3–0.4 GB (≈ 1 byte/parameter plus small float32 layers) | Estimate |
| Same model left at the default type on CPU | ≈ 1.1–1.2 GB, because float16 is converted to **float32** on x86 CPUs (CT2 quantization docs). We will force int8 | Verified behaviour, size Estimate |
| Transient peak while loading (reading a float16 checkpoint, then quantising) | ≈ 0.8–1.0 GB | Estimate |
| Translation working memory (beam 5, ≤ 160 input tokens per sentence) | < 0.1 GB | Estimate |
| ctranslate2 runtime + tokenisers + indic-nlp + sacremoses | ≈ 0.1–0.2 GB | Estimate |
| **Total app with translation loaded** | **steady ≈ 0.5–0.8 GB; peak while loading ≈ 1.0–1.4 GB** | Estimate. Fits easily in the 5.9 GB free |

The model is loaded **once per server** (cached as a shared resource), only when someone first clicks Translate, never at startup.

## 4. Disk space

| Item | Size | Status |
|---|---|---|
| Archive download (network transfer) | **2.78 GB** (2,778,571,691 bytes). This is the download size, not the installed size | Verified (HF API) |
| Archive contents | "3 directories, a fairseq model directory and 2 CT-ported model directories". A community guide loads `fairseq_model/ct2_fp16_model`; the second CT2 directory's name is not documented | Verified (README wording); names **Unverified** |
| Full extraction if everything were kept | probably ≥ 3 GB | Estimate |
| **What we would keep** | one CT2 directory + its `vocab/` folder ≈ **0.3–0.6 GB** | Estimate |
| Temporary disk during setup | ≈ 0 extra: the setup script streams the archive and extracts only the needed folder, without saving the 2.78 GB file | Design |
| Python packages installed | ≈ 25 MB of wheels, ≈ 60–120 MB installed | Wheels Verified; installed size Estimate |
| Free disk on C: | 86.9 GB | **Measured** |

## 5. Is conversion needed?

**No conversion is planned.** The official archive already contains CT2 checkpoints (README: "2 CT-ported model directories… provided only for efficient inference").

Risks and fallbacks:

- AI4Bharat's `install.sh` pins `ctranslate2==3.9.0`, so the checkpoints were most likely produced with CT2 3.x. CTranslate2 states that newer versions are backward compatible with previously converted models and that this "is rarely broken, even for major versions" (conversion docs). **Loading with 4.8.4 is therefore expected but Unverified.**
- **Fallback A:** `ctranslate2` 3.9.0 has no Python 3.12 wheel, but 3.19.0 and later do (Verified on PyPI). If 4.8.4 refuses the model, try the newest 3.x release; no Python downgrade is needed.
- **Fallback B (not recommended):** converting the fairseq checkpoint with `ct2-fairseq-converter` would need fairseq + PyTorch, and fairseq is impractical on Windows / Python 3.12. If A fails, I will stop and report rather than attempt this.
- The engine expects `vocab/model.SRC` and `vocab/model.TGT` inside the folder it is given (Verified, `engine.py`). Whether the CT2 folders already contain `vocab/` is **Unverified**; if not, the setup script copies the archive's vocab files in.

## 6. Quality and language coverage

| Item | Detail | Status |
|---|---|---|
| Direction | English → Indic (`en-indic`) | Verified |
| Languages | All 22 scheduled languages (26 language-script codes including English) | Verified (README) |
| Demo languages proposed | Hindi `hin_Deva`, Tamil `tam_Taml`, Bengali `ben_Beng`, Telugu `tel_Telu`, Marathi `mar_Deva`, Kannada `kan_Knda` | Codes Verified in the README language table |
| Quality | The distilled model trades some accuracy for size compared with the 1B model (paper). I will not quote BLEU/chrF numbers I have not checked. Step 8 will measure our own small set: number preservation, plus manual adequacy ratings on 10–15 answer sentences | Not yet measured |
| Known risk | Dates, amounts and names may be altered by MT. The existing `missing_numbers` check warns when a number disappears | Design |

## 7. How DocuBharat would call it

1. `requirements-translation.txt` (separate from `requirements.txt`) lists the 7 packages. The core install is unchanged.
2. `third_party/indictrans2/`: the minimum official inference files from the IndicTrans2 repo (`engine.py`, `flores_codes_map_indic.py`, `normalize_punctuation.py`, `normalize_regex_inference.py`), copied unmodified with AI4Bharat's MIT licence and the commit hash.
3. `scripts/setup_indictrans2.py` (run manually, once): reads `HF_TOKEN` from `.env`, streams the official archive over HTTPS, extracts only one CT2 folder + `vocab`, prints what it kept and its size, then loads the model and translates one test sentence.
4. `src/indic_translator.py`:
   - `IndicTrans2Translator.translate(text, "hin_Deva")` → `segment()` (existing) → official `Model(model_dir, device="cpu", model_type="ctranslate2").batch_translate(sentences, "eng_Latn", tgt)`.
   - The CT2 translator is created with `device="cpu"` and `compute_type="int8"`.
   - Translation runs only when the user clicks Translate on an answer; quotes and passages are never translated; `missing_numbers()` warns on lost dates or amounts.
5. **If anything fails**, the English answer stays and a plain message explains what is missing:
   - packages missing: "Translation not installed"
   - model folder missing: "Run the setup script"
   - CT2 load error (e.g. version incompatibility): "Translation model could not be loaded"
   - error during translation: shown for that answer only

   None of these affect PDF processing, BM25, rule-based extraction or Groq.

## 8. Rollback and switching it off

- `TRANSLATION_ENABLED=false` in `.env` hides the feature completely.
- Removing the model folder or uninstalling `requirements-translation.txt` leaves a working app (feature hidden).
- Groq stays exactly as it is. Tests run without the translation packages (an offline fake engine), plus one integration test that runs only when the model folder exists.

## 9. Hosted AI4Bharat options without Bhashini (checked today)

| Option | Finding | Status |
|---|---|---|
| HF Inference Providers | No IndicTrans2 deployment. The only provider-served `ai4bharat` model is `IndicBERTv2-MLM-only` (fill-mask), and free HF accounts get no monthly inference credits | Verified |
| HF Space `ai4bharat/IndicTrans3-beta` | Runtime stage **CONFIG_ERROR** (hardware `null`; torch incompatible with ZeroGPU). Model is a Gemma-3-based LLM under CC-BY-4.0. Not usable, and a Gradio Space API is not an official, stable service | Verified |
| HF Space `ai4bharat/IndicNER` | Runtime stage **RUNTIME_ERROR** (cannot download its own gated model) | Verified |
| models.ai4bharat.org | Interactive demo; no documented public API found | Verified (no API documentation found) |
| AIKosh (IndiaAI) IndicTrans2 page | Offers "Tryout" and "Download" only; no documented inference API | Verified |
| BharatGen on HF Inference Providers | No models served | Verified |

**Conclusion:** no officially documented, working hosted AI4Bharat/BharatGen inference exists without Bhashini. The local CT2 route is the only verified option that meets the constraints.

## 10. Minimal plan after approval (stops on first failure)

1. Install `requirements-translation.txt` (≈ 25 MB) and confirm `pip check`.
2. You accept the BPCC terms on HF and put `HF_TOKEN` in `.env` yourself.
3. Run the setup script: stream the archive, keep one CT2 folder + vocab, and **report the real folder names and sizes**.
4. Load with CT2 4.8.4 on CPU in a separate Python process; **measure** load time, peak RSS and per-sentence latency. If loading fails, try Fallback A once, then stop and report.
5. Add `src/indic_translator.py`, the UI button, offline tests and the integration test; run the full suite and browser check.
6. Update `docs/integrations.md` with the measured figures.

## References

- IndicTrans2 repository, README and licence table: https://github.com/AI4Bharat/IndicTrans2
- Inference engine source: https://github.com/AI4Bharat/IndicTrans2/blob/main/inference/engine.py
- Official install script (CT2 3.9.0 pin): https://github.com/AI4Bharat/IndicTrans2/blob/main/install.sh
- Checkpoint archive metadata: https://huggingface.co/api/datasets/ai4bharat/BPCC/tree/main/additional
- HF model card (parameter count): https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
- CTranslate2 installation: https://opennmt.net/CTranslate2/installation.html
- CTranslate2 conversion and backward compatibility: https://opennmt.net/CTranslate2/conversion.html
- CTranslate2 quantization (float16 → float32 on CPU, int8 support): https://opennmt.net/CTranslate2/quantization.html
- Paper: https://arxiv.org/abs/2305.16307
- Community setup guide (directory name evidence only, not authoritative): https://gist.github.com/Kishlay-notabot/a04e62a611b25bda413d284abbaaa254

## Results after approval (2026-10-29, measured)

| Question | Outcome |
|---|---|
| Archive layout | `en-indic-dist/ct2_int8_model` (model.bin 222.5 MB), `ct2_fp16_model` (427.8 MB), `fairseq_model/model/checkpoint_best.pt` (2,793 MB), `fairseq_model/vocab/{model,vocab}.{SRC,TGT}`. CT2 folders do **not** contain `vocab/`; the setup script moves `model.SRC`/`model.TGT` in |
| Conversion needed? | **No.** CTranslate2 4.8.4 loads the official int8 export directly; no fallback needed |
| Installed footprint | +86 MB packages; 218 MB model kept on disk |
| Memory | 453 MB peak stand-alone; 470–479 MB for the whole Streamlit app with the model loaded (estimate was 0.5–0.8 GB steady) |
| Speed | 1.2 s load; 0.34–0.50 s per sentence on CPU |
| Deviations from plan | `docopt` (a `mosestokenizer` dependency) is a pure-Python source package, not a wheel; `indic-nlp-library` installed with `--no-deps` to avoid Sphinx; `pip check` therefore lists `sphinx-argparse` and `sphinx-rtd-theme` as missing, which is intentional |
