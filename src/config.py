"""Runtime limits and tunables, read from environment variables with safe defaults.

Nothing here requires an API key: the app must start with no configuration at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = 10
    max_pages: int = 100
    # A page with fewer extractable characters than this is treated as "low text"
    # (likely scanned or image-only).
    min_chars_per_page: int = 40
    chunk_words: int = 500
    chunk_overlap_words: int = 50
    top_k: int = 4

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            max_upload_mb=_env_int("DOCUBHARAT_MAX_UPLOAD_MB", cls.max_upload_mb),
            max_pages=_env_int("DOCUBHARAT_MAX_PAGES", cls.max_pages),
            min_chars_per_page=_env_int("DOCUBHARAT_MIN_CHARS_PER_PAGE", cls.min_chars_per_page),
            chunk_words=_env_int("DOCUBHARAT_CHUNK_WORDS", cls.chunk_words, minimum=50),
            chunk_overlap_words=_env_int("DOCUBHARAT_CHUNK_OVERLAP_WORDS", cls.chunk_overlap_words, minimum=0),
            top_k=_env_int("DOCUBHARAT_TOP_K", cls.top_k),
        )
        # Overlap must stay well below the chunk size or chunking never advances.
        if settings.chunk_overlap_words >= settings.chunk_words // 2:
            settings = replace(settings, chunk_overlap_words=settings.chunk_words // 4)
        return settings
