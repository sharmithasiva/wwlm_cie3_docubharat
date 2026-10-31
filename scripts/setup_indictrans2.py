"""One-time setup: fetch AI4Bharat's official IndicTrans2 distilled En->Indic CT2 model.

Usage (after accepting the terms of https://huggingface.co/datasets/ai4bharat/BPCC
and putting HF_TOKEN=hf_... in .env):

    python scripts/setup_indictrans2.py

What it does:
- Streams the official archive (2.78 GB, linked from the IndicTrans2 README) over HTTPS
  without saving it, and keeps only the int8 CTranslate2 model and the two SentencePiece
  models (about 218 MB) in models/indictrans2-en-indic-dist/. Everything else is skipped.
- Lists every file in the archive with its size, so the real layout is documented.
- Uses tarfile's "data" filter, so archive paths cannot escape the target folder.
- Never prints the token.
"""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

ARCHIVE_URL = "https://huggingface.co/datasets/ai4bharat/BPCC/resolve/main/additional/en-indic-dist.tar.gz"
TERMS_URL = "https://huggingface.co/datasets/ai4bharat/BPCC"
EXPECTED_BYTES = 2_778_571_691
OUT_DIR = ROOT / "models" / "indictrans2-en-indic-dist"


class CountingReader:
    """File-like wrapper that reports download progress."""

    def __init__(self, raw):
        self.raw, self.read_bytes, self._next_report = raw, 0, 0
        self.started = time.perf_counter()

    def read(self, size=-1):
        chunk = self.raw.read(size)
        self.read_bytes += len(chunk)
        if self.read_bytes >= self._next_report:
            elapsed = max(time.perf_counter() - self.started, 1e-6)
            print(
                f"  downloaded {self.read_bytes / 1e9:5.2f} / {EXPECTED_BYTES / 1e9:.2f} GB "
                f"({self.read_bytes / 1e6 / elapsed:5.1f} MB/s)",
                flush=True,
            )
            self._next_report += 250_000_000
        return chunk


# Archive layout observed on 2026-10-29 (also recorded in manifest.json):
#   en-indic-dist/ct2_int8_model/{config.json, model.bin (222 MB), source_vocabulary.txt, target_vocabulary.txt}
#   en-indic-dist/ct2_fp16_model/...                    (428 MB, not needed: int8 is used on CPU)
#   en-indic-dist/fairseq_model/model/checkpoint_best.pt (2.79 GB, not needed)
#   en-indic-dist/fairseq_model/vocab/model.SRC, model.TGT (SentencePiece models the engine needs)
INT8_DIR = "ct2_int8_model"
SPM_FILES = ("model.SRC", "model.TGT")


def wanted(name: str) -> bool:
    """Keep the int8 CTranslate2 folder and the two SentencePiece models; skip everything else."""
    parts = name.split("/")
    return INT8_DIR in parts or (len(parts) >= 2 and parts[-2] == "vocab" and parts[-1] in SPM_FILES)


def place_vocab(out_dir: Path) -> Path:
    """AI4Bharat's engine reads <ckpt_dir>/vocab/model.SRC|TGT, but the archive stores them
    under fairseq_model/vocab; move them into the int8 folder."""
    ct2_dir = next(out_dir.rglob(INT8_DIR))
    (ct2_dir / "vocab").mkdir(exist_ok=True)
    for spm_file in SPM_FILES:
        source = next(p for p in out_dir.rglob(spm_file) if p.parent.parent.name != INT8_DIR)
        shutil.move(str(source), ct2_dir / "vocab" / spm_file)
    leftover = out_dir / ct2_dir.parent.name / "fairseq_model"
    if leftover.exists():
        shutil.rmtree(leftover)
    return ct2_dir


def main() -> int:
    token = (dotenv_values(ROOT / ".env").get("HF_TOKEN") or "").strip()
    if not token:
        print(f"HF_TOKEN is not set in .env. Accept the dataset terms at {TERMS_URL}, create a read token at "
              "https://huggingface.co/settings/tokens and add HF_TOKEN=... to .env.")
        return 1
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        print(f"{OUT_DIR} already exists and is not empty; remove it first to re-download.")
        return 1

    headers = {"Authorization": f"Bearer {token}"}
    probe = requests.get(ARCHIVE_URL, headers={**headers, "Range": "bytes=0-0"}, timeout=30, allow_redirects=True)
    if probe.status_code in (401, 403):
        print(f"Access denied (HTTP {probe.status_code}). Accept the terms at {TERMS_URL} with the same account.")
        return 1
    if probe.status_code not in (200, 206):
        print(f"Unexpected response while checking access: HTTP {probe.status_code}")
        return 1
    print("Access to the official archive confirmed. Streaming (nothing large is saved)...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    listing, kept = [], []
    with requests.get(ARCHIVE_URL, headers=headers, stream=True, timeout=60, allow_redirects=True) as response:
        response.raise_for_status()
        reader = CountingReader(response.raw)
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                listing.append({"name": member.name, "bytes": member.size, "type": "dir" if member.isdir() else "file"})
                if member.isfile() and wanted(member.name):
                    archive.extract(member, OUT_DIR, filter="data")
                    kept.append({"name": member.name, "bytes": member.size})
                    print(f"  kept    {member.name} ({member.size / 1e6:.1f} MB)", flush=True)
                elif member.isfile():
                    print(f"  skipped {member.name} ({member.size / 1e6:.1f} MB)", flush=True)

    ct2_dir = place_vocab(OUT_DIR)
    print(f"Model folder ready: {ct2_dir.relative_to(ROOT)}")

    manifest = {
        "source": ARCHIVE_URL,
        "licence": "MIT (AI4Bharat IndicTrans2 model checkpoints)",
        "downloaded_bytes": reader.read_bytes,
        "archive_listing": listing,
        "kept_files": kept,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    kept_mb = sum(k["bytes"] for k in kept) / 1e6
    print(f"\nDone. Streamed {reader.read_bytes / 1e9:.2f} GB, kept {len(kept)} files ({kept_mb:.0f} MB) in {OUT_DIR}")
    print("Full archive listing saved to manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
