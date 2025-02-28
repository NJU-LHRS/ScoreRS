import pathlib
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from peft import PeftModel
from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.utils import logging

from .lhrsbot_constant import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .lhrsbot_utils import (
    LayerNormFp32,
    ResidualAttentionBlock,
    get_2d_sincos_pos_embed,
)

logger = logging.get_logger(__name__)


class LHRSVisionModal(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()

        self.vision_model_name = vision_encoder
        self.encoder = SiglipVisionModel.from_pretrained(self.vision_model_name)
        self.encoder.vision_model.encoder.layers = (
            self.encoder.vision_model.encoder.layers[:-1]
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.vision_model_name
        )
        self.extract_stage = [
            self.encoder.config.num_hidden_layers // 3,
            self.encoder.config.num_hidden_layers // 3 * 2,
            self.encoder.config.num_hidden_layers - 1,
        ]

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def unfreeze_encoder(self):
        self.encoder.requires_grad_(True)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.encoder.dtype

    @property
    def device(self):
        return self.encoder.device

    @property
    def config(self):
        return self.encoder.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    def forward(
        self,
        pixel_values,
    ):
        outputs = self.encoder(
            pixel_values.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        if hasattr(self, "extract_stage"):
            image_embeds = []
            for _, stage in enumerate(self.extract_stage):
                current_hidden_states = outputs.hidden_states[stage]
                image_embeds.append(current_hidden_states)
            image_embeds = torch.cat(image_embeds, dim=1)
            return image_embeds.to(pixel_values.dtype)
        else:
            return outputs.last_hidden_state.to(pixel_values.dtype)


class VisionPerceiver(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.num_query = config.bridge_num_query
        self.num_layers = config.bridge_num_layers
        self.num_attention_heads = config.bridge_num_attention_heads
        self.encoder_hidden_size = config.vision_encoder_hidden_size
        self.stage_num = config.bridge_stage_num
        self.split_part = config.bridge_split_part
        self.max_size = config.bridge_max_size
        self.embed_dim = config.bridge_embed_dim
        self.num_patches = (
            int(config.bridge_num_patches**0.5),
            int(config.bridge_num_patches**0.5),
        )
        self.use_moe = config.bridge_use_moe
        self.num_experts = config.bridge_num_experts
        self.num_selects = config.bridge_num_selects
        self.output_size = config.hidden_size

        self.query = nn.Parameter(torch.zeros(1, self.num_query, self.embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02, mean=0.0)

        if self.encoder_hidden_size != self.embed_dim:
            self.in_proj = nn.Linear(self.encoder_hidden_size, self.embed_dim)
        else:
            self.in_proj = nn.Identity()

        self.layers = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=self.embed_dim,
                    n_head=self.num_attention_heads,
                    is_cross_attention=True,
                    norm_layer=LayerNormFp32,
                    use_moe=self.use_moe,
                    num_experts=self.num_experts,
                    num_selects=self.num_selects,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.layernorm_query = LayerNormFp32(self.embed_dim)
        self.layernorm_kv = LayerNormFp32(self.embed_dim)
        self.layernorm_post = LayerNormFp32(self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.output_size)

        self._set_2d_pos_embed(self.max_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _set_2d_pos_embed(self, max_size, device="cpu"):
        pos_embed = (
            torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size))
            .float()
            .to(device)
        )
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(
        self,
        image_embs: torch.Tensor,
    ) -> torch.Tensor:
        image_embs = self.in_proj(image_embs)

        query_tokens = self.query.expand(image_embs.size(0), -1, -1)

        if isinstance(self.stage_num, int):
            stage1_query, stage2_query, stage3_query = torch.split(
                query_tokens, self.num_query // self.stage_num, dim=1
            )
        else:
            stage1_query, stage2_query, stage3_query = torch.split(
                query_tokens, self.stage_num, dim=1
            )

        stage1_image, stage2_image, stage3_image = torch.split(
            image_embs, self.split_part, dim=1
        )

        all_tokens = []
        pos_embed = (
            self.pos_embed[: self.num_patches[0], : self.num_patches[1], :]
            .reshape(self.num_patches[0] * self.num_patches[1], -1)
            .to(image_embs.dtype)
        )
        pos_embed = pos_embed.unsqueeze(0).expand(image_embs.size(0), -1, -1)
        pos_embed = pos_embed.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)
        for sub_token, sub_image in zip(
            [stage1_query, stage2_query, stage3_query],
            [stage1_image, stage2_image, stage3_image],
        ):
            sub_token = self.layernorm_query(sub_token)
            sub_image = self.layernorm_kv(sub_image)

            sub_image = sub_image.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)
            sub_token = sub_token.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)

            for layer in self.layers:
                sub_token = layer(sub_token, sub_image + pos_embed, sub_image)

            sub_token = sub_token.permute(1, 0, 2)  # (L, B, D) -> (B, L, D)
            all_tokens.append(sub_token)

        query_tokens = torch.cat(all_tokens, dim=1)
        query_tokens = self.layernorm_post(query_tokens)
        out = self.out_proj(query_tokens)
        return out

    def load_state_dict(self, state_dict, **kwrags):
        msg = super().load_state_dict(state_dict, strict=False)

        if len(msg.missing_keys) > 0:
            assert self.use_moe
            layer_up_weight = "layers.{}.mlp.c_fc.weight"
            layer_up_bias = "layers.{}.mlp.c_fc.bias"
            layer_down_weight = "layers.{}.mlp.c_proj.weight"
            layer_down_bias = "layers.{}.mlp.c_proj.bias"

            for layer_idx in range(len(self.layers)):
                up_weight = state_dict[layer_up_weight.format(layer_idx)]
                up_bias = state_dict[layer_up_bias.format(layer_idx)]
                down_weight = state_dict[layer_down_weight.format(layer_idx)]
                down_bias = state_dict[layer_down_bias.format(layer_idx)]

                for expert_idx in range(self.num_experts):
                    self.layers[layer_idx].mlp.calculator.experts.weight_up[
                        expert_idx
                    ].data = up_weight
                    self.layers[layer_idx].mlp.calculator.experts.bias_up[
                        expert_idx
                    ].data = up_bias
                    self.layers[layer_idx].mlp.calculator.experts.weight_down[
                        expert_idx
                    ].data = down_weight.mT
                    self.layers[layer_idx].mlp.calculator.experts.bias_down[
                        expert_idx
                    ].data = down_bias

    def freeze_encoder(self):
        self.requires_grad_(False)

    def unfreeze_encoder(self):
        self.requires_grad_(True)


class LHRSMetaModel:
    def __init__(self, config):
        super(LHRSMetaModel, self).__init__(config)

        if hasattr(config, "vision_encoder"):
            self.initialize_vision_encoder(config)

    def get_vision_encoder(self):
        vision_encoder = getattr(self, "rgb", None)
        return vision_encoder

    def get_rgb_pooler(self):
        rgb_pooler = getattr(self, "rgb_pooler", None)
        return rgb_pooler

    def get_image_processor(self):
        encoder = self.get_vision_encoder()
        if encoder is not None:
            return encoder.image_processor
        return None

    def initialize_vision_encoder(self, model_args):
        if self.get_vision_encoder() is None:
            rgb = LHRSVisionModal(self.config.vision_encoder)
            if getattr(model_args, "rgb_freeze", False):
                rgb.freeze_encoder()
            else:
                rgb.unfreeze_encoder()

            self.rgb = rgb

        self.config.vision_encoder_hidden_size = self.rgb.hidden_size
        self.config.bridge_split_part = [
            self.rgb.num_patches,
            self.rgb.num_patches,
            self.rgb.num_patches,
        ]
        self.config.bridge_max_size = 64
        self.config.bridge_num_patches = self.rgb.num_patches
        self.config.bridge_embed_dim = self.rgb.hidden_size

        if getattr(self, "rgb_pooler", None) is None:
            rgb_pooler = VisionPerceiver(self.config)

            if getattr(model_args, "rgb_pooler_freeze", False):
                rgb_pooler.freeze_encoder()
            else:
                rgb_pooler.unfreeze_encoder()

            self.rgb_pooler = rgb_pooler

        if getattr(model_args, "rgb_ckpt_path", None) is not None:
            self.rgb.load_state_dict(model_args.rgb_ckpt_path)

        if getattr(model_args, "pooler_ckpt_path", None) is not None:
            self.rgb_pooler.load_state_dict(model_args.pooler_ckpt_path)


class LHRSMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        model = self.get_model().get_vision_encoder()
        return model

    def get_rgb_pooler(self):
        model = self.get_model().get_rgb_pooler()
        return model

    def get_image_processor(self):
        encoder = self.get_vision_encoder()
        if encoder is not None:
            return encoder.image_processor
        return None

    def encode_images(self, pixel_values):
        encode_image = self.get_vision_encoder()(pixel_values)
        pooled_image = self.get_rgb_pooler()(encode_image)
        return pooled_image

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        vision_encoder = self.get_vision_encoder()
        if vision_encoder is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def custom_load_state_dict(self, state_dict_path, strict=True):
        ckpt = torch.load(state_dict_path, map_location="cpu")
        if "model" in ckpt.keys():
            ckpt = ckpt["model"]
        text_path = pathlib.Path(state_dict_path).parent / "TextLoRA"

        logger.info(f"Loading RGB encoder.")
        msg = self.get_vision_encoder().load_state_dict(ckpt["rgb_ckpt"], strict=strict)
        logger.info(
            f"After loading RGB encoder: Missing: {msg.missing_keys}. Unexpected: {msg.unexpected_keys}"
        )

        other_ckpt = ckpt["other_ckpt"]
        self.get_rgb_pooler().load_state_dict(other_ckpt["rgb_pooler"])
        del ckpt

        if text_path.exists():
            logger.info(f"Loadding LoRA parameters.")
            self = PeftModel.from_pretrained(
                self,
                text_path,
                is_trainable=False,
                torch_dtype=torch.float16,
            )
            self = self.merge_and_unload()


class LHRSMetaConfig:
    vision_encoder: str = "google/siglip-so400m-patch14-384"
    bridge_num_query: int = 272
    bridge_num_layers: int = 6
    bridge_num_attention_heads: int = 8
    bridge_stage_num: List[int] = [112, 96, 64]
    bridge_use_moe: bool = True
    bridge_num_experts: int = 4
    bridge_num_selects: int = 2
