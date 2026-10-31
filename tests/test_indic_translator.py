"""Local IndicTrans2 wrapper: availability checks and behaviour with a fake engine (offline).

A real-model test runs only when the model has been downloaded with
scripts/setup_indictrans2.py; otherwise it is skipped.
"""

import importlib.util
from pathlib import Path

import pytest

from src import indic_translator
from src.indic_translator import (
    LANGUAGES,
    IndicTrans2Translator,
    TranslationConfig,
    TranslationUnavailable,
    find_ct2_dir,
    status,
)


def _fake_model_dir(tmp_path: Path, folder: str) -> Path:
    ct2 = tmp_path / "model" / "fairseq_model" / folder
    (ct2 / "vocab").mkdir(parents=True)
    (ct2 / "model.bin").write_bytes(b"x")
    (ct2 / "vocab" / "model.SRC").write_bytes(b"x")
    (ct2 / "vocab" / "model.TGT").write_bytes(b"x")
    return tmp_path / "model"


class FakeEngine:
    def __init__(self, outputs=None, error=None):
        self.outputs, self.error, self.calls = outputs, error, []

    def batch_translate(self, sentences, src, tgt):
        self.calls.append((list(sentences), src, tgt))
        if self.error:
            raise self.error
        return self.outputs if self.outputs is not None else [f"[{tgt}] {s}" for s in sentences]


@pytest.fixture
def translator(tmp_path, monkeypatch):
    model_dir = _fake_model_dir(tmp_path, "ct2_int8_model")
    monkeypatch.setattr(indic_translator, "status", lambda config: (True, "ready"))
    t = IndicTrans2Translator(TranslationConfig(model_dir=model_dir))
    return t


def _use_engine(t, engine, monkeypatch):
    monkeypatch.setattr(t, "_load", lambda: engine)
    return engine


# --------------------------------------------------------------------------- availability


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("TRANSLATION_ENABLED", "false")
    available, reason = status(TranslationConfig.from_env())
    assert not available and "turned off" in reason


def test_missing_model_folder(tmp_path):
    available, reason = status(TranslationConfig(model_dir=tmp_path / "nothing"))
    if importlib.util.find_spec("ctranslate2") is None:
        assert "not installed" in reason
    else:
        assert not available and "setup_indictrans2.py" in reason
    with pytest.raises(TranslationUnavailable):
        IndicTrans2Translator(TranslationConfig(model_dir=tmp_path / "nothing"))


def test_relative_model_dir_resolves_to_project(monkeypatch):
    monkeypatch.setenv("INDICTRANS2_MODEL_DIR", "models/x")
    assert TranslationConfig.from_env().model_dir == indic_translator.ROOT / "models" / "x"


def test_find_ct2_dir_prefers_int8_and_needs_vocab(tmp_path):
    root = _fake_model_dir(tmp_path, "ct2_fp16_model")
    assert find_ct2_dir(root).name == "ct2_fp16_model"
    _fake_model_dir(tmp_path, "ct2_int8_model")
    assert find_ct2_dir(root).name == "ct2_int8_model"
    (root / "fairseq_model" / "ct2_int8_model" / "vocab" / "model.TGT").unlink()
    assert find_ct2_dir(root).name == "ct2_fp16_model"  # incomplete folder is ignored


# --------------------------------------------------------------------------- translation behaviour


def test_sentences_are_sent_separately_with_flores_codes(translator, monkeypatch):
    engine = _use_engine(translator, FakeEngine(), monkeypatch)
    result = translator.translate("The fee is Rs. 1,800. Pay by 7 November 2026.", "hin_Deva")
    assert engine.calls == [(["The fee is Rs. 1,800.", "Pay by 7 November 2026."], "eng_Latn", "hin_Deva")]
    assert result.text.startswith("[hin_Deva]") and result.missing_numbers == []
    assert "IndicTrans2" in result.model_label


def test_lost_numbers_are_reported(translator, monkeypatch):
    _use_engine(translator, FakeEngine(outputs=["शुल्क देय है।"]), monkeypatch)
    assert translator.translate("The fee is Rs. 1,800.", "hin_Deva").missing_numbers == ["1,800"]


def test_model_is_loaded_once(translator, monkeypatch):
    loads = []
    engine = FakeEngine()
    monkeypatch.setattr(translator, "_load", lambda: loads.append(1) or engine)
    translator.translate("One.", "tam_Taml")
    translator.translate("Two.", "tam_Taml")
    assert loads == [1]


@pytest.mark.parametrize(
    "engine, message",
    [
        (FakeEngine(error=RuntimeError("Unsupported model binary version")), "could not run"),
        (FakeEngine(outputs=["", " "]), "no text"),
    ],
)
def test_engine_failures_become_user_messages(translator, monkeypatch, engine, message):
    _use_engine(translator, engine, monkeypatch)
    with pytest.raises(TranslationUnavailable, match=message):
        translator.translate("First. Second.", "hin_Deva")


def test_load_failure_is_reported(translator, monkeypatch):
    def broken_load():
        raise RuntimeError("cannot open model.bin")

    monkeypatch.setattr(translator, "_load", broken_load)
    with pytest.raises(TranslationUnavailable, match="could not run"):
        translator.translate("Hello.", "hin_Deva")


def test_unsupported_language_and_long_text(translator):
    with pytest.raises(TranslationUnavailable, match="Unsupported"):
        translator.translate("Hello.", "fra_Latn")
    with pytest.raises(TranslationUnavailable, match="too long"):
        translator.translate("Sentence. " * 50, "hin_Deva")


def test_language_codes_exist_in_ai4bharat_map():
    from third_party.indictrans2.inference.flores_codes_map_indic import flores_codes

    assert set(LANGUAGES) <= set(flores_codes)


# --------------------------------------------------------------------------- real model (optional)

_config = TranslationConfig.from_env()
_real_available = status(_config)[0]


@pytest.mark.skipif(not _real_available, reason="IndicTrans2 model not downloaded")
def test_real_model_translates_to_hindi():
    t = IndicTrans2Translator(_config)
    result = t.translate("The examination fee is Rs. 1,800 per semester.", "hin_Deva")
    assert result.text and any("ऀ" <= ch <= "ॿ" for ch in result.text)  # Devanagari output
