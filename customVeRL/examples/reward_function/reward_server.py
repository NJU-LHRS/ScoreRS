# reward_server.py

import base64
import io
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoProcessor, Qwen2VLConfig
from qwen_reward import Qwen2Reward


MODEL_NAME_OR_PATH = ""  # your specific fine-tuned Qwen2Reward model path
REWARD_MODEL_DEVICE_ID = "1"  # Default CUDA device ID, "cpu" for CPU


# --- Pydantic Models ---
class RewardRequest(BaseModel):
    model: str = Field(
        default=MODEL_NAME_OR_PATH, description="Identifier for the model to use."
    )
    problem_description: str = Field(
        ...,
        description="The common problem description or question. May include <image> tag if image is relevant and your model expects it.",
    )
    image_data: Optional[str] = Field(
        None,
        description="Base64 encoded image data (e.g., 'data:image/jpeg;base64,...'). Optional.",
    )
    completion1: str = Field(..., description="The first completion/answer text.")
    completion2: str = Field(
        ...,
        description="The second completion/answer text (e.g., ground truth or alternative).",
    )


class ScoreItem(BaseModel):
    completion_id: str
    score: float


class RewardResponse(BaseModel):
    model: str
    scores: List[ScoreItem]


class ModelContext:
    def __init__(self):
        self.model: Optional[Qwen2Reward] = None
        self.processor: Optional[AutoProcessor] = None
        self.device: Optional[str] = None


model_ctx = ModelContext()

# --- FastAPI App ---
app = FastAPI(title="Qwen2Reward API")


def decode_image(image_data_str: Optional[str]) -> Optional[Image.Image]:
    if not image_data_str:
        return None
    try:
        image_source = image_data_str
        if image_data_str.startswith("data:image"):
            header, encoded = image_data_str.split(",", 1)
            image_source = encoded

        image_bytes = base64.b64decode(image_source)
        image = Image.open(io.BytesIO(image_bytes))
        # Ensure image is in RGB format, as most vision models expect
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid or unsupported image_data: {e}"
        )


def prepare_chat_messages(
    problem: str, completion: str, image_present: bool
) -> List[Dict[str, Any]]:
    """
    Prepares a list of messages in the format expected by Qwen-VL's apply_chat_template.
    If an image is present, a placeholder {"type": "image"} is added to the user content.
    The actual image tensor is handled by the processor during the __call__ method.
    """
    user_content_list: List[Dict[str, Any]] = [{"type": "text", "text": problem}]
    if image_present:
        # This signals to `apply_chat_template` to make space/syntax for an image.
        # The actual image PIL object is passed to the processor's `images` argument later.
        user_content_list.append({"type": "image"})

    return [
        {"role": "user", "content": user_content_list},
        {"role": "assistant", "content": completion},
    ]


@app.on_event("startup")
async def startup_event():
    if torch.cuda.is_available() and REWARD_MODEL_DEVICE_ID != "cpu":
        model_ctx.device = f"cuda:{REWARD_MODEL_DEVICE_ID}"
    else:
        model_ctx.device = "cpu"

    print(f"Attempting to load model on device: {model_ctx.device}")

    try:
        model_ctx.processor = AutoProcessor.from_pretrained(
            MODEL_NAME_OR_PATH, trust_remote_code=True
        )
        # Ensure padding is on the right, consistent with how rewards are often calculated (last token of non-padded sequence)
        model_ctx.processor.tokenizer.padding_side = "right"

        config = Qwen2VLConfig.from_pretrained(
            MODEL_NAME_OR_PATH, trust_remote_code=True
        )
        config.ranked_candidate_num = 1  # For scoring, not ranking loss calculation
        config.pad_token_id = model_ctx.processor.tokenizer.pad_token_id

        # Determine torch_dtype based on device
        dtype = torch.bfloat16 if model_ctx.device != "cpu" else torch.float32

        model_ctx.model = Qwen2Reward.from_pretrained(
            MODEL_NAME_OR_PATH,
            config=config,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map=model_ctx.device,
            trust_remote_code=True,
        )
        model_ctx.model.eval()
        for param in model_ctx.model.parameters():
            param.requires_grad = False
        print(
            f"Successfully loaded Qwen2Reward model '{MODEL_NAME_OR_PATH}' and processor on device '{model_ctx.device}'."
        )
    except Exception as e:
        print(f"FATAL: Failed to load model or processor during startup: {e}")
        # Depending on server setup, you might want the app to exit or enter a degraded state.
        # For now, it will raise an error on first request if model is None.
        model_ctx.model = None
        model_ctx.processor = None
        # raise RuntimeError(f"Failed to load model: {e}") # Or handle more gracefully


# --- API Endpoints ---
@app.post("/v1/rewards", response_model=RewardResponse)
async def get_rewards_endpoint(request: RewardRequest):
    if not model_ctx.model or not model_ctx.processor or not model_ctx.device:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Check server logs for loading errors.",
        )

    pil_image = decode_image(request.image_data)

    # 1. Construct message lists for applying chat template
    messages_for_comp1 = prepare_chat_messages(
        request.problem_description, request.completion1, pil_image is not None
    )
    messages_for_comp2 = prepare_chat_messages(
        request.problem_description, request.completion2, pil_image is not None
    )

    # 2. Apply chat template to get formatted text strings
    try:
        # `add_generation_prompt=False` because inputs are full exchanges, not prompts for generation
        text_comp1 = model_ctx.processor.tokenizer.apply_chat_template(
            messages_for_comp1, tokenize=False, add_generation_prompt=False
        )
        text_comp2 = model_ctx.processor.tokenizer.apply_chat_template(
            messages_for_comp2, tokenize=False, add_generation_prompt=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error applying chat template: {e}. Check message structure and problem description content (e.g. <image> tags).",
        )

    texts_batch = [text_comp1, text_comp2]
    # If an image is present, it's used for both comparisons.
    # The processor expects a list of images, one for each text in the batch.
    images_batch = [pil_image, pil_image] if pil_image else None

    # 3. Tokenize using the processor (handles text and images)
    try:
        inputs = model_ctx.processor(
            text=texts_batch,
            images=images_batch,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model_ctx.device)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing inputs with processor: {e}"
        )

    # 4. Get model outputs (reward scores)
    with torch.inference_mode():
        # Autocast for mixed precision if on CUDA and model supports bfloat16/float16
        # Determine autocast device type ("cuda" or "cpu")
        autocast_device_type = (
            model_ctx.device.split(":")[0] if "cuda" in model_ctx.device else "cpu"
        )
        autocast_enabled = autocast_device_type == "cuda"
        autocast_dtype = (
            torch.bfloat16 if autocast_enabled else torch.float32
        )  # bfloat16 for CUDA, float32 for CPU

        with torch.autocast(
            device_type=autocast_device_type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            try:
                outputs = model_ctx.model(**inputs, return_dict=True, return_loss=False)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                # Optionally log input shapes for debugging:
                # print(f"Input shapes: {{k: v.shape if hasattr(v, 'shape') else type(v) for k,v in inputs.items()}}")
                raise HTTPException(
                    status_code=500, detail=f"Error during model computation: {e}"
                )

    raw_scores = outputs.values  # `values` field from Qwen2RewardOutput
    if raw_scores is None:
        raise HTTPException(
            status_code=500, detail="Model output did not contain 'values'."
        )

    raw_scores = raw_scores.flatten().to(torch.float32)

    if raw_scores.shape[0] != 2:
        raise HTTPException(
            status_code=500,
            detail=f"Expected 2 scores from the model, but received {raw_scores.shape[0]}. Input batch size was {len(texts_batch)}.",
        )

    score1 = raw_scores[0].item()
    score2 = raw_scores[1].item()

    return RewardResponse(
        model=request.model if request.model else MODEL_NAME_OR_PATH,
        scores=[
            ScoreItem(completion_id="completion1", score=score1),
            ScoreItem(completion_id="completion2", score=score2),
        ],
    )
