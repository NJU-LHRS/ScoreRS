"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

from transformers import AutoProcessor, AutoTokenizer
import torch
import gc
from vllm import LLM

# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


def load_llm(model_name: str, model_path: str = None, device: str = "cuda:0"):

    # Force garbage collection
    gc.collect()
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if model_name == "internvl":
        if model_path is None:
            model_path = "OpenGVLab/InternVL2_5-8B"
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            disable_mm_preprocessor_cache=False,
            dtype="float16",
            device=device,
            gpu_memory_utilization=0.6,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return llm, processor
    elif model_name == "llava":
        if model_path is None:
            model_path = "llava-hf/llava-1.5-7b-hf"
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            disable_mm_preprocessor_cache=False,
            dtype="float16",
            device=device,
            gpu_memory_utilization=0.6,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return llm, tokenizer
    elif model_name == "llava_v1.6":
        if model_path is None:
            model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        llm = LLM(
            model=model_path,
            disable_mm_preprocessor_cache=False,
            dtype="float16",
            device=device,
            gpu_memory_utilization=0.6,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return llm, processor
    elif model_name == "qwen2-vl":
        if model_path is None:
            model_path = "Qwen/Qwen2-VL-7B-Instruct"
        llm = LLM(
            model=model_path,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            disable_mm_preprocessor_cache=False,
            dtype="float16",
            device=device,
            gpu_memory_utilization=0.6,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return llm, processor
    elif model_name == "qwen2-vl-r1":
        if model_path is None:
            model_path = "Qwen/Qwen2-VL-7B-Instruct"
        llm = LLM(
            model=model_path,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            disable_mm_preprocessor_cache=False,
            dtype="float16",
            device=device,
            gpu_memory_utilization=0.6,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return llm, processor
    else:
        raise ValueError(f"Model {model_name} not supported")


# InternVL
def run_internvl(question: str, processor: AutoProcessor):
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [processor.convert_tokens_to_ids(i) for i in stop_tokens]
    return prompt, stop_token_ids


# LLaVA-1.5
def run_llava(question: str, processor: AutoProcessor):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    stop_token_ids = None
    return prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question: str, processor: AutoProcessor):
    prompt = f"[INST] <image>\n{question} [/INST]"
    stop_token_ids = None
    return prompt, stop_token_ids


# Qwen2-VL
def run_qwen2_vl(question: str, processor: AutoProcessor):
    placeholder = "<|image_pad|>"

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    stop_token_ids = None
    return prompt, stop_token_ids


# Qwen2-VL-R1
def run_qwen2_vl_r1(question: str, processor: AutoProcessor):
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    placeholder = "<|image_pad|>"

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    stop_token_ids = None
    return prompt, stop_token_ids


model_example_map = {
    "internvl": run_internvl,
    "llava": run_llava,
    "llava-next": run_llava_next,
    "qwen2-vl": run_qwen2_vl,
    "qwen2-vl-r1": run_qwen2_vl_r1,
}
