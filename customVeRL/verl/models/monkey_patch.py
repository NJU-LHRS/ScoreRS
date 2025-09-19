from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .transformers.flash_attention_utils import flash_attention_forward
from .transformers.qwen2_vl import qwen2_vl_attn_forward


def apply_ulysses_patch(model_type: str) -> None:
    if model_type in ("llama", "gemma", "gemma2", "mistral", "qwen2"):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    elif model_type in ("qwen2_vl", "qwen2_5_vl"):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLFlashAttention2,
        )
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLFlashAttention2,
        )

        Qwen2VLFlashAttention2.forward = qwen2_vl_attn_forward
        Qwen2_5_VLFlashAttention2.forward = qwen2_vl_attn_forward
    else:
        raise NotImplementedError(
            f"Model architecture {model_type} is not supported yet."
        )
