from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------
# Public model configuration
# ---------------------------------------------------------------------

BASE_MODEL_ID = os.getenv(
    "BASE_MODEL_ID",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
)

# This should point to the public selected 25-step GRPO LoRA adapter.
# It can also be replaced with a local adapter folder.
ADAPTER_ID = os.getenv(
    "ADAPTER_ID",
    "Miladsaeedi70/smollm2-135m-scientific-grpo-lora",
)

MAX_PROMPT_LENGTH = int(
    os.getenv("MAX_PROMPT_LENGTH", "384")
)

SYSTEM_PROMPT = (
    "You are a scientific research assistant. "
    "Provide accurate, clear, concise, and task-focused answers. "
    "Acknowledge uncertainty when appropriate."
)


# ---------------------------------------------------------------------
# Device and model loading
# ---------------------------------------------------------------------

def select_device() -> str:
    """Select CUDA, Apple MPS, or CPU."""
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Any, str]:
    """Load the tokenizer, base model, and LoRA adapter once."""
    device = select_device()

    print(f"Loading base model: {BASE_MODEL_ID}")
    print(f"Loading adapter: {ADAPTER_ID}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_ID,
        is_trainable=False,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def generate_response(
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate a deterministic scientific response."""
    prompt = prompt.strip()

    if not prompt:
        return "Please enter a scientific question or instruction."

    try:
        tokenizer, model, device = load_model()
    except Exception as error:
        return (
            "The model could not be loaded.\n\n"
            f"Base model: {BASE_MODEL_ID}\n"
            f"Adapter: {ADAPTER_ID}\n\n"
            "Confirm that the adapter repository is public, or set "
            "ADAPTER_ID to a valid local adapter folder.\n\n"
            f"Error: {error}"
        )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
    ).to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_tokens = inputs["input_ids"].shape[1]
    response_tokens = generated[0, prompt_tokens:]

    response = tokenizer.decode(
        response_tokens,
        skip_special_tokens=True,
    ).strip()

    if not response:
        return "The model returned an empty response."

    return response


# ---------------------------------------------------------------------
# Local Gradio demo
# ---------------------------------------------------------------------

demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(
            label="Scientific prompt",
            lines=5,
            placeholder=(
                "Ask a scientific question or provide a scientific "
                "instruction..."
            ),
        ),
        gr.Slider(
            minimum=32,
            maximum=256,
            value=128,
            step=16,
            label="Maximum new tokens",
        ),
    ],
    outputs=gr.Textbox(
        label="Model response",
        lines=10,
    ),
    examples=[
        [
            (
                "Explain why spatial cross-validation is important "
                "in environmental machine learning."
            ),
            128,
        ],
        [
            (
                "Compare random cross-validation with temporal "
                "cross-validation."
            ),
            128,
        ],
        [
            (
                "Summarize the difference between model reproduction "
                "and spatial prediction."
            ),
            128,
        ],
    ],
    title="Scientific LLM — LoRA SFT + GRPO",
    description=(
        "A local Gradio demo for a SmolLM2-135M scientific assistant "
        "fine-tuned with LoRA SFT and groupwise-judge GRPO."
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        share=False,
        server_name="127.0.0.1",
    )
