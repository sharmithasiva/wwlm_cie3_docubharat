# Vendored: AI4Bharat IndicTrans2 inference engine

These files are copied **unmodified** from the official repository:

- Repository: https://github.com/AI4Bharat/IndicTrans2
- Commit: `4f08e39cc6bf13cd62e2445dc725f22bff1a9219` (2025-10-03)
- Files: `inference/__init__.py`, `engine.py`, `flores_codes_map_indic.py`, `indic_num_map.py`, `normalize_punctuation.py`, `normalize_regex_inference.py`
- Licence: MIT, Copyright (c) AI4Bharat (see `LICENSE`)

DocuBharat uses only the CTranslate2 path of `inference.engine.Model` (`model_type="ctranslate2"`) and calls `batch_translate`. The fairseq path (`custom_interactive.py`) is not included, and `translate_paragraph` (which needs Perl via `mosestokenizer`) is not called.

Model weights are not stored here. `scripts/setup_indictrans2.py` downloads them from AI4Bharat's official archive.
