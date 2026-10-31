"""Optional local translation of answers with AI4Bharat IndicTrans2 (CTranslate2, CPU).

Model: IndicTrans2 distilled En->Indic, from AI4Bharat's official archive (MIT),
fetched once by scripts/setup_indictrans2.py. Inference uses AI4Bharat's own engine
(third_party/indictrans2, unmodified, CTranslate2 path only) on the CPU in int8.

The model is loaded lazily on the first translation request, never at app start.
If the packages, the model folder or the model itself are unavailable, `status()`
explains why and the app keeps working in English.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.translation_checks import missing_numbers, segment

logger = logging.getLogger("docubharat.translation")

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "third_party" / "indictrans2"
DEFAULT_MODEL_DIR = ROOT / "models" / "indictrans2-en-indic-dist"
SOURCE_LANGUAGE = "eng_Latn"

# FLORES codes as used by AI4Bharat's engine (inference/flores_codes_map_indic.py).
LANGUAGES = {
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "mar_Deva": "Marathi",
    "guj_Gujr": "Gujarati",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "pan_Guru": "Punjabi",
    "ory_Orya": "Odia",
    "asm_Beng": "Assamese",
    "urd_Arab": "Urdu",
}
MODEL_LABEL = "AI4Bharat IndicTrans2 (distilled En→Indic, run locally with CTranslate2)"


class TranslationUnavailable(Exception):
    """Raised with a user-safe message when translation cannot run."""


@dataclass(frozen=True)
class TranslationConfig:
    enabled: bool = True
    model_dir: Path = DEFAULT_MODEL_DIR
    threads: int = 4

    @classmethod
    def from_env(cls) -> "TranslationConfig":
        enabled = os.getenv("TRANSLATION_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        model_dir = Path(os.getenv("INDICTRANS2_MODEL_DIR", "") or DEFAULT_MODEL_DIR)
        if not model_dir.is_absolute():
            model_dir = ROOT / model_dir
        try:
            threads = max(1, int(os.getenv("TRANSLATION_THREADS", "4")))
        except ValueError:
            threads = 4
        return cls(enabled=enabled, model_dir=model_dir, threads=threads)


@dataclass
class Translation:
    source_text: str
    text: str
    target_language: str
    seconds: float
    model_label: str = MODEL_LABEL
    missing_numbers: list[str] = field(default_factory=list)


def find_ct2_dir(model_dir: Path) -> Path | None:
    """The CTranslate2 folder to load: one holding model.bin and vocab/model.SRC + model.TGT.

    An int8 folder is preferred (smallest in memory); any other CT2 folder is
    quantised to int8 at load time instead.
    """
    if not model_dir.exists():
        return None
    candidates = [
        p.parent
        for p in model_dir.rglob("model.bin")
        if (p.parent / "vocab" / "model.SRC").exists() and (p.parent / "vocab" / "model.TGT").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: ("int8" not in p.name.lower(), str(p)))[0]


def status(config: TranslationConfig) -> tuple[bool, str]:
    """(available, reason). Cheap: never imports the model or reads weights."""
    if not config.enabled:
        return False, "Translation is turned off (TRANSLATION_ENABLED=false)."
    missing = [m for m in ("ctranslate2", "sentencepiece", "indicnlp", "sacremoses", "mosestokenizer")
               if importlib.util.find_spec(m) is None]
    if missing:
        return False, "Translation packages are not installed (see requirements-translation.txt)."
    if not (ENGINE_DIR / "inference" / "engine.py").exists():
        return False, "AI4Bharat inference engine files are missing (third_party/indictrans2)."
    if find_ct2_dir(config.model_dir) is None:
        return False, "The IndicTrans2 model is not downloaded yet. Run: python scripts/setup_indictrans2.py"
    return True, "ready"


def create_translator(config: TranslationConfig | None = None) -> tuple["IndicTrans2Translator | None", str]:
    """(translator, "ready") or (None, reason). Does not load the model yet."""
    config = config or TranslationConfig.from_env()
    available, reason = status(config)
    if not available:
        return None, reason
    return IndicTrans2Translator(config), reason


class IndicTrans2Translator:
    def __init__(self, config: TranslationConfig):
        available, reason = status(config)
        if not available:
            raise TranslationUnavailable(reason)
        self.config = config
        self.ct2_dir = find_ct2_dir(config.model_dir)
        self._model = None
        self._lock = threading.Lock()
        self.load_seconds: float | None = None

    def _load(self):
        """Create AI4Bharat's engine on the CPU with int8 weights (first use only)."""
        if str(ENGINE_DIR) not in sys.path:
            sys.path.insert(0, str(ENGINE_DIR))
        import ctranslate2
        from inference.engine import Model

        # The engine builds ctranslate2.Translator(ckpt_dir, device=device) itself. Wrapping the
        # constructor during __init__ adds int8 quantisation and a thread limit without editing
        # AI4Bharat's file, and avoids first loading the weights as float32.
        original = ctranslate2.Translator
        ctranslate2.Translator = functools.partial(
            original, compute_type="int8", intra_threads=self.config.threads
        )
        try:
            started = time.perf_counter()
            model = Model(str(self.ct2_dir), device="cpu", model_type="ctranslate2")
            self.load_seconds = time.perf_counter() - started
        finally:
            ctranslate2.Translator = original
        logger.info("IndicTrans2 loaded from %s in %.1fs", self.ct2_dir.name, self.load_seconds)
        return model

    def translate(self, text: str, target_language: str) -> Translation:
        if target_language not in LANGUAGES:
            raise TranslationUnavailable(f"Unsupported language: {target_language}")
        try:
            sentences = segment(text)
        except ValueError as exc:
            raise TranslationUnavailable("This answer is too long to translate.") from exc
        if not sentences:
            raise TranslationUnavailable("There is no text to translate.")
        with self._lock:  # the engine's tokenisers and CT2 model are shared by all sessions
            try:
                if self._model is None:
                    self._model = self._load()
                started = time.perf_counter()
                outputs = self._model.batch_translate(sentences, SOURCE_LANGUAGE, target_language)
                seconds = time.perf_counter() - started
            except TranslationUnavailable:
                raise
            except Exception as exc:
                logger.exception("IndicTrans2 translation failed")
                raise TranslationUnavailable("The translation model could not run. See the server log.") from exc
        translated = " ".join(o.strip() for o in outputs if o and o.strip())
        if not translated:
            raise TranslationUnavailable("The translation model returned no text.")
        return Translation(text, translated, target_language, seconds, missing_numbers=missing_numbers(text, translated))
