"""Private ZeroGPU Space serving BharatGen Patram for DocuBharat.

Model: bharatgenai/patram-7b-instruct (Apache 2.0, gated). Usage follows the official
model card: AutoProcessor/AutoModelForCausalLM with trust_remote_code, bfloat16,
processor.process(images=[image], text=prompt) and model.generate_from_batch(...).

Deviation from the card: it pins torch 2.7.0, but ZeroGPU supports torch >= 2.8.0 only,
so this Space uses torch 2.8.0. Whether Patram's custom code works there is exactly
what the first test of this Space checks.

API (for DocuBharat, via gradio_client): endpoint "/ask"
    inputs:  page image, question (str), max_new_tokens (int)
    outputs: answer (str), generation seconds (float)
Requires a Space secret HF_TOKEN with access to the gated Patram repository.
"""

import os
import time

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL_ID = "bharatgenai/patram-7b-instruct"
REVISION = os.getenv("PATRAM_REVISION", "patram1.1")  # model card default; also patram1.0 / patram1.2
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_QUESTION_CHARS = 1000
MAX_NEW_TOKENS = 512
MAX_IMAGE_SIDE = 1600  # keep page images moderate: less GPU time per call

# ZeroGPU: load at module level and place on "cuda" (emulated until a GPU is attached).
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=REVISION, trust_remote_code=True, torch_dtype=torch.bfloat16, token=HF_TOKEN
).to("cuda")
model.eval()


def _prepare_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    longest = max(image.size)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


@spaces.GPU(duration=60)
def ask(image: Image.Image, question: str, max_new_tokens: int = 256):
    if image is None:
        raise gr.Error("Please provide a document page image.")
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not question:
        raise gr.Error("Please provide a question.")
    max_new_tokens = int(min(max(int(max_new_tokens), 16), MAX_NEW_TOKENS))

    # Prompt format from the model card example: "Question: ... Answer based on the image."
    prompt = f"Question: {question} Answer based on the image."
    started = time.perf_counter()
    inputs = processor.process(images=[_prepare_image(image)], text=prompt)
    # From the card: float tensors (image patches) must match the model dtype; integer ids must not be cast.
    inputs = {
        k: (v.to("cuda").to(model.dtype) if torch.is_floating_point(v) else v.to("cuda")).unsqueeze(0)
        for k, v in inputs.items()
    }
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )
    generated = output[0, inputs["input_ids"].size(1):]
    answer = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, round(time.perf_counter() - started, 2)


demo = gr.Interface(
    fn=ask,
    inputs=[
        gr.Image(type="pil", label="Document page"),
        gr.Textbox(label="Question", max_lines=4),
        gr.Slider(16, MAX_NEW_TOKENS, value=256, step=16, label="Max new tokens"),
    ],
    outputs=[gr.Textbox(label="Answer"), gr.Number(label="Generation seconds")],
    title="DocuBharat: BharatGen Patram document VQA (private test Space)",
    description=f"{MODEL_ID} @ {REVISION}. Answers questions about a document page image.",
    api_name="ask",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
