"""Check the hosted LLM configuration with one grounded question on a synthetic sample.

Usage (after putting LLM_API_KEY in .env):

    python scripts/check_llm.py

Sends at most one request: the top passages of data/sample_documents/exam_circular.pdf
(fictional content) plus one question. Prints the result; never prints the API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.llm_provider import LLMConfig, LLMError, create_provider  # noqa: E402
from src.llm_tasks import llm_answer  # noqa: E402
from src.pipeline import process_pdf  # noqa: E402

QUESTION = "What is the examination fee for regular students?"


def main() -> int:
    load_dotenv(ROOT / ".env")
    config = LLMConfig.from_env()
    print(f"Provider: {config.provider} | endpoint: {config.base_url} | model: {config.model}")
    print(f"API key set: {'yes' if config.api_key else 'NO'}")
    try:
        provider = create_provider(config)
    except LLMError as exc:
        print(f"Configuration problem: {exc.user_message}")
        return 1
    if provider is None:
        print("No LLM configured. Put LLM_API_KEY in .env (see .env.example).")
        return 1

    sample = ROOT / "data" / "sample_documents" / "exam_circular.pdf"
    doc = process_pdf(sample.read_bytes(), sample.name)
    evidence = doc.index.search(QUESTION, top_k=4)
    try:
        result = llm_answer(QUESTION, evidence, provider, pages=doc.pages)
    except LLMError as exc:
        print(f"Request failed: {exc.user_message}")
        return 1

    print(f"\nQ: {QUESTION}\nA ({result.model}): {result.answer}")
    for c in result.citations:
        status = f"verified on p. {c.page}" if c.verified else "NOT found in document"
        print(f"  quote [{c.passage_id}] {status}: {c.quote}")
    print(f"\nGrounded: {result.grounded}")
    return 0 if result.found and result.grounded else 2


if __name__ == "__main__":
    raise SystemExit(main())
