# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import math
import os
import sys
import time
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import safetensors
import torch
from accelerate.utils import DistributedDataParallelKwargs
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerControl,
    TrainerState,
)
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS, TRAINING_ARGS_NAME
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, is_peft_available
from trl import PPOConfig, PPOTrainer
from trl.core import (
    WANDB_PADDING,
    PPODecorators,
    convert_to_scalar,
    logprobs_from_logits,
    stack_dicts,
    stats_to_np,
)
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import (
    AverageMeter,
    count_parameters,
    get_current_device,
    get_logits_processor,
)
from ..callbacks import (
    FixValueHeadModelCallback,
    PPOSaveIntermediateCallback,
    SaveProcessorCallback,
)
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import (
    dump_layernorm,
    get_rewards_from_server,
    replace_model,
    restore_layernorm,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class HackTrainer(Trainer):
    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self.processing_class


class CustomPPOTrainer(PPOTrainer, HackTrainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[List["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
        )
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
            init_kl_coef=finetuning_args.init_kl_coef,
            gamma=finetuning_args.gamma,
            lam=finetuning_args.lam,
            kl_penalty=finetuning_args.kl_penalty,
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(
                    find_unused_parameters=training_args.ddp_find_unused_parameters
                )
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = (
                training_args.deepspeed_plugin
            )
            if ppo_config.log_with is not None:
                logger.warning_rank0(
                    "PPOTrainer cannot use external logger when DeepSpeed is enabled."
                )
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = (
                backward_batch_size
                * finetuning_args.ppo_buffer_size
                * training_args.world_size
            )
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id],
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self.is_fsdp_enabled = (
            getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        )
        callbacks = (
            DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        )
        self.callback_handler = CallbackHandler(
            callbacks,
            self.accelerator.unwrap_model(self.model),
            self.tokenizer,
            self.optimizer,
            self.lr_scheduler,
        )
        if self.args.max_steps > 0:
            logger.info_rank0(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if finetuning_args.custom_reward_model or not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(
                        reward_model.pretrained_model, "is_loaded_in_4bit", False
                    )
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(
                    self.reward_model, evaluation_mode=True
                )

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_old_version, self.accelerator
            )
            self.add_callback(BAdamCallback)

        if finetuning_args.save_ppo_stats:
            self.add_callback(
                PPOSaveIntermediateCallback(
                    ppo_save_interval=finetuning_args.ppo_stats_save_interval,
                    ppo_log_interval=finetuning_args.ppo_stats_log_interval,
                )
            )
            self.save_ppo_stats = True
            self.ppo_log_interval = finetuning_args.ppo_stats_log_interval
        else:
            self.save_ppo_stats = False

    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self.processing_class

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (
            (PreTrainedModel,)
            if not is_peft_available()
            else (PreTrainedModel, PeftModel)
        )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if getattr(self, "processing_class", None) is not None:
            self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += [
                "labels",
                "query",
                "response",
                "label",
                "images",
                "videos",
            ]

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError(
                "`resume_from_checkpoint` will be supported in the future version."
            )

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        logger.info_rank0(
            "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {:,}".format(
                total_train_batch_size
            )
        )
        logger.info_rank0(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}"
        )
        logger.info_rank0(
            f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}"
        )
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(
            f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}"
        )

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            if "pixel_values" in batch:
                batch_pixel_values = []
            else:
                batch_pixel_values = None

            if "image_grid_thw" in batch:
                batch_image_grid_thw = []
            else:
                batch_image_grid_thw = None

            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                if "pixel_values" in batch and batch["pixel_values"].dim() == 2:
                    pixel_values = batch["pixel_values"]

                    image_token_start = 0
                    image_token_end = 0
                    for i in range(0, idx):
                        # number of image tokens in the previous batches
                        image_token_start += batch["image_grid_thw"][i].prod()

                    for i in range(0, idx + self.config.mini_batch_size):
                        image_token_end += batch["image_grid_thw"][i].prod()

                    pixel_values = pixel_values[image_token_start:image_token_end, :]
                    current_mini_batch = {
                        **batch[idx : idx + self.config.mini_batch_size],
                    }
                    current_mini_batch["pixel_values"] = pixel_values
                    batch_pixel_values.append(pixel_values)
                    batch_image_grid_thw.append(current_mini_batch["image_grid_thw"])
                else:
                    current_mini_batch = batch[idx : idx + self.config.mini_batch_size]

                    if "pixel_values" in current_mini_batch:
                        batch_pixel_values.append(current_mini_batch["pixel_values"])

                    if "image_grid_thw" in current_mini_batch:
                        batch_image_grid_thw.append(
                            current_mini_batch["image_grid_thw"]
                        )

                mini_batch_queries, mini_batch_responses = self.get_inputs(
                    current_mini_batch
                )
                mini_batch_rewards = self.get_rewards(
                    mini_batch_queries,
                    mini_batch_responses,
                    current_mini_batch["pixel_values"],
                    current_mini_batch["image_grid_thw"],
                )
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # Run PPO step
            self.model.train()
            stats = self.step(
                queries,
                responses,
                rewards,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
            )
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(
                        queries, skip_special_tokens=True
                    )
                    batch["response"] = self.tokenizer.batch_decode(
                        responses, skip_special_tokens=True
                    )
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            if (
                self.save_ppo_stats
                and self.state.global_step % self.ppo_log_interval == 0
            ):
                stats_dict = {
                    "prompt": self.tokenizer.decode(
                        queries[0], skip_special_tokens=True
                    ),
                    "response": self.tokenizer.decode(
                        responses[0], skip_special_tokens=True
                    ),
                    "response_tokens": [
                        self.tokenizer.decode(token) for token in responses[0]
                    ],
                    "logprobs": stats["objective/logprobs"][0][
                        stats["ppo/masks"][0].astype(bool)
                    ].tolist(),
                    "ref_logprobs": stats["objective/ref_logprobs"][0][
                        stats["ppo/masks"][0].astype(bool)
                    ].tolist(),
                    "values": stats["ppo/values"][0][
                        stats["ppo/masks"][0].astype(bool)
                    ].tolist(),
                    "token_rewards": stats["ppo/token_rewards"][0][
                        stats["ppo/masks"][0].astype(bool)
                    ].tolist(),
                    "reward": rewards[0].cpu().item(),
                }
                self.state.current_ppo_stat = stats_dict

            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if (
                self.is_local_process_zero()
                and (step + 1) % self.args.logging_steps == 0
            ):
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(
                        self.args.output_dir,
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
                    )
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                training_args
            )
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self,
        training_args: "Seq2SeqTrainingArguments",
        num_training_steps: int,
        optimizer: "torch.optim.Optimizer",
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(
        self, batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
        r"""
        Generates model's responses given queries.
        """
        if (
            batch["input_ids"].size(0) == 1
        ):  # handle llama2 ppo with gradient accumulation > 1
            start_index = (
                (batch["input_ids"][0] != self.tokenizer.pad_token_id)
                .nonzero()[0]
                .item()
            )
            for k, v in batch.items():
                if k != "pixel_values" and k != "image_grid_thw":
                    batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(
            self.model, self.accelerator
        ) as unwrapped_model:
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = (
                self.accelerator.unwrap_model(self.model)
            )
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_output: "torch.Tensor" = unwrapped_model.generate(
                generation_config=self.generation_config,
                logits_processor=get_logits_processor(),
                **batch,
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (
                (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            )
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif (
                self.tokenizer.eos_token_id == self.tokenizer.pad_token_id
            ):  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses

    def prepare_model_inputs(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
    ):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [
                    {"input_ids": q, "attention_mask": torch.ones_like(q)}
                    for q in queries
                ]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [
                    {"input_ids": r, "attention_mask": torch.ones_like(r)}
                    for r in responses
                ]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [
                    {
                        "input_ids": ids,
                        "attention_mask": torch.ones_like(ids),
                    }
                    for ids in input_ids
                ]
            ).to(self.current_device)

            if pixel_values is not None or image_grid_thw is not None:
                if isinstance(pixel_values, list):
                    pixel_values = torch.stack(pixel_values)
                if isinstance(image_grid_thw, list):
                    image_grid_thw = torch.stack(image_grid_thw)

                input_data["pixel_values"] = pixel_values
                input_data["image_grid_thw"] = image_grid_thw

        input_data.pop("labels", None)  # we don't want to compute LM losses
        return input_data

    @torch.no_grad()
    def get_rewards(
        self,
        queries: List["torch.Tensor"],
        responses: List["torch.Tensor"],
        pixel_values: List["torch.Tensor"] = None,
        image_grid_thw: List["torch.Tensor"] = None,
    ) -> List["torch.Tensor"]:
        r"""
        Computes scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.finetuning_args.reward_model_type == "api":
            if pixel_values is not None or image_grid_thw is not None:
                raise NotImplementedError(
                    "API reward model does not support image inputs."
                )
            else:
                token_ids = [
                    torch.cat((q, r), dim=-1).tolist()
                    for q, r in zip(queries, responses)
                ]
                messages = self.tokenizer.batch_decode(
                    token_ids, skip_special_tokens=False
                )
                return get_rewards_from_server(self.reward_model, messages)

        batch: Dict[str, "torch.Tensor"] = self.prepare_model_inputs(
            queries, responses, pixel_values, image_grid_thw
        )
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = (
            self.accelerator.unwrap_model(self.model)
        )

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(
            reward_model, self.accelerator
        ), self.amp_context:  # support bf16
            outputs: "torch.Tensor" = reward_model(
                **batch, return_dict=True, use_cache=False
            )
            if "values" in outputs:
                values = outputs.values
            else:
                values = outputs[-1]

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        if values.dim() == 2:
            rewards = values.gather(
                dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1)
            )
        else:
            rewards = values

        return rewards.float().detach()  # use fp32 type

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: Dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
        pixel_values: Optional["torch.Tensor"] = None,
        image_grid_thw: Optional["torch.Tensor"] = None,
    ) -> Tuple[
        "torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"
    ]:
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        if pixel_values is not None and not isinstance(pixel_values, list):
            pixel_values = [pixel_values]
        if image_grid_thw is not None and not isinstance(image_grid_thw, list):
            image_grid_thw = [image_grid_thw]

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs : (i + 1) * fbs]
                for key, value in model_inputs.items()
            }
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            if pixel_values is not None:
                input_kwargs["pixel_values"] = pixel_values[i]
            if image_grid_thw is not None:
                input_kwargs["image_grid_thw"] = image_grid_thw[i]

            with self.amp_context:  # support bf16
                logits, _, values = model(
                    **input_kwargs, return_dict=True, use_cache=False
                )

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat(
                        (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                    )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = (
                        masks[j, start:end] * response_masks_batch[j][start:end]
                    )

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
        pixel_values: Optional[List[torch.FloatTensor]] = None,
        image_grid_thw: Optional[List[torch.FloatTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.
            pixel_values (List[`torch.FloatTensor`], *optional*):
                List of tensors containing pixel values of the images.
            image_grid_thw (List[`torch.FloatTensor`], *optional*):
                List of tensors containing the image grid of shape (3, H, W).

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = (
                self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            )
            if self.config.use_score_norm:
                scores = (
                    scores - self.running.mean.to(**tensor_to_kwargs)
                ) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(
                scores.float(), -self.config.score_clip, self.config.score_clip
            ).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["decoder_input_ids"],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                        pad_first=pad_first,
                    )
                )
                model_inputs["decoder_attention_mask"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["decoder_attention_mask"],
                        dim=1,
                        pad_index=0,
                        pad_first=pad_first,
                    )
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(
                    logits_or_none, None, gather=False
                )
                ref_full_logprobs = logprobs_from_logits(
                    ref_logits_or_none, None, gather=False
                )

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, all_logprobs, ref_logprobs, masks
                )
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(
                values, rewards, masks
            )
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = (
                    backward_batch_start + self.config.backward_batch_size
                )
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(
                    0, self.config.backward_batch_size, self.config.mini_batch_size
                ):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[
                        mini_batch_start:mini_batch_end
                    ]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [
                            batch_dict["responses"][i] for i in mini_batch_inds
                        ],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {
                            k: mini_batch_dict[k] for k in model_inputs_names
                        }

                        current_pixel_values, current_image_grid_thw = (
                            self.construct_pixel_values_select(
                                pixel_values, mini_batch_inds, image_grid_thw
                            )
                        )

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            pixel_values=current_pixel_values,
                            image_grid_thw=current_image_grid_thw,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], WANDB_PADDING
        )
        train_stats["policy/ratio"] = torch.flatten(
            train_stats["policy/ratio"]
        ).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        if self.finetuning_args.save_ppo_stats:
            stats["ppo/token_rewards"] = rewards.detach().cpu().numpy()
            stats["ppo/values"] = values.detach().cpu().numpy()
            stats["ppo/masks"] = masks.detach().cpu().numpy()

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(
                    self.model
                )  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(
                    self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
                )
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = (
                self.accelerator.unwrap_model(self.model)
            )
            self._save(output_dir, state_dict=unwrapped_model.state_dict())

    def construct_pixel_values_select(
        self, pixel_values, mini_batch_inds, image_grid_thw=None
    ):
        if pixel_values is None:
            return None, None

        if pixel_values[0].dim() == 3:
            return torch.cat(pixel_values, dim=0)[mini_batch_inds], None
        else:
            assert image_grid_thw is not None
            convert_grid = torch.cat(image_grid_thw)
            convert_pixel = torch.cat(pixel_values)
            flatten_pixel_values = []

            current_start = 0
            for i in range(convert_grid.size(0)):
                current_grid = convert_grid[i]
                current_pixel = convert_pixel[
                    current_start : current_start + current_grid.prod()
                ]
                flatten_pixel_values.append(current_pixel)
                current_start += current_grid.prod()

            return_grid = convert_grid[mini_batch_inds]
            return_pixel = torch.cat([flatten_pixel_values[i] for i in mini_batch_inds])
            return return_pixel, return_grid
