#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
import os
import re
import shutil
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import ImageFile
from src.model.llava_constant import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

ImageFile.LOAD_TRUNCATED_IMAGES = True


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        print(x)

        return x + self.proj(x)


class CLIPVisionTower(nn.Module):
    def clip_interpolate_embeddings(self, image_size=600, patch_size=14):
        """This function helps interpolating positional embeddings during checkpoint loading,
        especially when you want to apply a pre-trained model on images with different resolution.

        Args:
            image_size (int): Image size of the new model.
            patch_size (int): Patch size of the new model.
            model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
            interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
            reset_heads (bool): If true, not copying the state of heads. Default: False.

        Returns:
            OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
        """
        # Shape of pos_embedding is (1, seq_length, hidden_dim)
        state_dict = (
            self.vision_tower.vision_model.embeddings.position_embedding.state_dict()
        )
        pos_embedding = state_dict["weight"]
        pos_embedding = pos_embedding.unsqueeze(0)
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(
                f"Unexpected position embedding shape: {pos_embedding.shape}"
            )

        new_seq_length = (image_size // patch_size) ** 2 + 1

        # Need to interpolate the weights for the position embedding.
        # We do this by reshaping the positions embeddings to a 2d grid, performing
        # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
        if new_seq_length != seq_length:
            # The class token embedding shouldn't be interpolated so we split it up.
            seq_length -= 1
            new_seq_length -= 1
            pos_embedding_token = pos_embedding[:, :1, :]
            pos_embedding_img = pos_embedding[:, 1:, :]

            # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            torch._assert(
                seq_length_1d * seq_length_1d == seq_length,
                "seq_length is not a perfect square!",
            )

            # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding_img = pos_embedding_img.reshape(
                1, hidden_dim, seq_length_1d, seq_length_1d
            )
            new_seq_length_1d = image_size // patch_size

            # Perform interpolation.
            # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
            new_pos_embedding_img = nn.functional.interpolate(
                pos_embedding_img,
                size=new_seq_length_1d,
                mode="bicubic",
                align_corners=True,
            )

            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
            new_pos_embedding_img = new_pos_embedding_img.reshape(
                1, hidden_dim, new_seq_length
            )

            # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
            new_pos_embedding = torch.cat(
                [pos_embedding_token, new_pos_embedding_img], dim=1
            )[0]
            state_dict["weight"] = new_pos_embedding
            self.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(
                new_seq_length + 1, hidden_dim
            )
            self.vision_tower.vision_model.embeddings.position_embedding.load_state_dict(
                state_dict
            )
            self.vision_tower.vision_model.embeddings.image_size = image_size
            self.vision_tower.vision_model.embeddings.patch_size = patch_size
            self.vision_tower.vision_model.embeddings.position_ids = torch.arange(
                new_seq_length + 1
            ).expand((1, -1))

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.vision_tower_name
            )
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)
            self.clip_interpolate_embeddings(image_size=504, patch_size=14)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.clip_interpolate_embeddings(image_size=504, patch_size=14)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # print(image.shape)
                # import pdb; pdb.set_trace()
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )

                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                # print(image_features.shape)

                image_features.append(image_feature)
        else:
            # print(images.shape)
            # import pdb; pdb.set_trace()
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # print(image_features.shape)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class GeoChatMetaModel:

    def __init__(self, config):
        super(GeoChatMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
            # print(mm_projector)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class GeoChatMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids[:half_len]
                )
                cur_input_embeds_2 = self.get_model().embed_tokens(
                    cur_input_ids[half_len:]
                )
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2],
                    dim=0,
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                print(mm_projector_weights)
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


class GeoChatConfig(LlamaConfig):
    model_type = "geochat"


class GeoChatLlamaModel(GeoChatMetaModel, LlamaModel):
    config_class = GeoChatConfig

    def __init__(self, config: LlamaConfig):
        super(GeoChatLlamaModel, self).__init__(config)


class GeoChatLlamaForCausalLM(LlamaForCausalLM, GeoChatMetaForCausalLM):
    config_class = GeoChatConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GeoChatLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = (
            self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images
            )
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("geochat", GeoChatConfig)
AutoModelForCausalLM.register(GeoChatConfig, GeoChatLlamaForCausalLM)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )

    if "openai" not in vision_tower:
        vision_tower = "openai/" + vision_tower
    return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if "geochat" in model_name.lower():
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading Geochat from base model...")
            model = GeoChatLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )

            print("Loading additional GeoChat weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id, filename=filename, subfolder=subfolder
                    )
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(
                    model_path, "non_lora_trainables.bin"
                )
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        elif model_base is not None:
            # this may be mm projector only
            print("Loading GeoChat from base model...")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = GeoChatLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
            )

            mm_projector_weights = torch.load(
                os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
            )
            mm_projector_weights = {
                k: v.to(torch.float16) for k, v in mm_projector_weights.items()
            }
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            print("Loading GeoChat......")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = GeoChatLlamaForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )

    image_processor = None

    if "geochat" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len