---
title: DocuBharat Patram
emoji: 📄
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - bharatgenai/patram-7b-instruct
short_description: Private test Space serving BharatGen Patram for DocuBharat
---

# DocuBharat: BharatGen Patram (private test Space)

Serves [bharatgenai/patram-7b-instruct](https://huggingface.co/bharatgenai/patram-7b-instruct) (Apache 2.0) on ZeroGPU for the DocuBharat CS3235 project. It answers questions about one document page image at a time. API endpoint: `/ask`.

Setup notes:

- **Hardware:** ZeroGPU (selected in Space settings).
- **Visibility:** private.
- **Secret:** `HF_TOKEN`, a read token whose account has accepted Patram's access terms.
- **Environment:** the model card pins torch 2.7.0, but ZeroGPU requires torch ≥ 2.8.0, so this Space uses 2.8.0. `transformers` 4.50.3 matches the card.
