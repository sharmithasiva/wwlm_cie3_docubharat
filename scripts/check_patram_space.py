"""Test the private Patram ZeroGPU Space with synthetic sample pages.

Usage (after the Space is running and PATRAM_SPACE_ID + HF_TOKEN are in .env):

    python scripts/check_patram_space.py

Renders pages of the fictional sample PDFs to PNG, asks Patram questions whose answers
are known, and reports each answer, the Space's generation time, the wall-clock time
and whether the expected value appears. Every call uses ZeroGPU quota (free accounts:
5 GPU-minutes per day), so the default run makes only 3 calls. Never prints the token.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

SAMPLES = ROOT / "data" / "sample_documents"
CHECKS = [
    # (document, page, question, text expected in a correct answer)
    ("exam_circular.pdf", 1, "What is the examination fee for regular students?", "1,800"),
    ("exam_circular.pdf", 1, "What is the circular number?", "KSU/COE/2026/41"),
    ("scanned_library_notice.pdf", 1, "Until what date will the library remain open late?", "15 December"),
]


def render_page(pdf: Path, page_number: int, out_dir: Path, dpi: int = 110) -> Path:
    with pymupdf.open(pdf) as doc:
        pixmap = doc[page_number - 1].get_pixmap(dpi=dpi)
    path = out_dir / f"{pdf.stem}_p{page_number}.png"
    pixmap.save(path)
    return path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    env = dotenv_values(ROOT / ".env")
    space_id = (env.get("PATRAM_SPACE_ID") or "").strip()
    token = (env.get("HF_TOKEN") or "").strip()
    if not space_id or not token:
        print("Set PATRAM_SPACE_ID (e.g. your-username/docubharat-patram) and HF_TOKEN in .env.")
        return 1
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("gradio_client is not installed: python -m pip install gradio_client==1.14.0")
        return 1

    print(f"Connecting to {space_id} (a sleeping Space can take a few minutes to start)...")
    started = time.perf_counter()
    client = Client(space_id, hf_token=token, verbose=False)
    print(f"Connected in {time.perf_counter() - started:.1f}s\n")

    passed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, page, question, expected in CHECKS:
            image = render_page(SAMPLES / name, page, Path(tmp))
            started = time.perf_counter()
            try:
                answer, gen_seconds = client.predict(handle_file(str(image)), question, 128, api_name="/ask")
            except Exception as exc:  # report and continue: quota, cold start, or model errors
                print(f"[{name} p{page}] {question}\n  ERROR: {type(exc).__name__}: {str(exc)[:300]}\n")
                continue
            wall = time.perf_counter() - started
            ok = expected.lower().replace(" ", "") in str(answer).lower().replace(" ", "")
            passed += ok
            print(f"[{name} p{page}] {question}\n  Patram: {answer}\n"
                  f"  expected '{expected}': {'FOUND' if ok else 'not found'} | generation {gen_seconds}s | wall {wall:.1f}s\n")
    print(f"{passed}/{len(CHECKS)} answers contained the expected value.")
    return 0 if passed == len(CHECKS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
