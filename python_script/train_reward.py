import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import wandb
from packaging import version
from PIL import Image
from src.custom_trainer.CustomTrainer.EpochBasedTrainer import EpochBasedTrainer
from src.custom_trainer.CustomTrainer.IterBasedTrainer import IterBasedTrainer
from src.custom_trainer.CustomTrainer.optimizer import build_optimizer
from src.custom_trainer.CustomTrainer.utils import (
    CustomTrainerConfigError,
    DataConfig,
    ModelConfig,
    TrainConfig,
    barrier,
    deepspeed_init_distributed,
    get_default_device,
    get_fsdp_wrap_policy,
    init_distributed,
    seed_all,
    setup_logger,
)
from src.dataset.dataset_class import Qwen2RewardDataset
from src.model.qwen_reward import Qwen2Reward
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoProcessor, DataCollatorWithPadding
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger("train")


@dataclass
class CustomDataConfig(DataConfig):
    image_root: Optional[str] = ""
    target_root: Optional[str] = ""
    max_pixel_base: Optional[int] = 1280


@dataclass
class CustomModelConfig(ModelConfig):
    base_model_path: Optional[str] = ""
    reward_model_ckpt_path: Optional[str] = None
    ranked_candidate_num: Optional[int] = 2
    train_vision_model: Optional[bool] = False
    train_text_model: Optional[bool] = False
    train_value_head: Optional[bool] = True


@dataclass
class CustomTrainConfig(TrainConfig):
    model: CustomModelConfig = field(default_factory=CustomModelConfig)
    data: CustomDataConfig = field(default_factory=CustomDataConfig)


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        flattened_features = []
        batch_pixel_values = []
        batch_image_grid_thw = []

        if "pixel_values" in features[0]:
            batch_pixel_values = [feature["pixel_values"] for feature in features]
        if "image_grid_thw" in features[0]:
            batch_image_grid_thw = [feature["image_grid_thw"] for feature in features]
        for feature in features:
            for idx, candidate_ids in enumerate(feature["input_ids"]):
                flattened_feature = {
                    "input_ids": candidate_ids,
                    "attention_mask": feature["attention_mask"][idx],
                }

                # Copy other features that aren't nested
                for key, value in feature.items():
                    if key not in [
                        "input_ids",
                        "attention_mask",
                        "pixel_values",
                        "image_grid_thw",
                    ]:
                        flattened_feature[key] = value

                flattened_features.append(flattened_feature)

        batch = super().__call__(flattened_features)

        if len(batch_pixel_values) > 0:
            batch["pixel_values"] = torch.cat(batch_pixel_values)
        if len(batch_image_grid_thw) > 0:
            batch["image_grid_thw"] = torch.cat(batch_image_grid_thw)

        return batch


def main(config: TrainConfig):
    logger.info(f"Creating model")

    processor = AutoProcessor.from_pretrained(
        config.model.base_model_path, max_pixels=config.data.max_pixel_base * 28 * 28
    )
    processor.tokenizer.padding_side = "right"

    model_config = Qwen2VLConfig.from_pretrained(config.model.base_model_path)
    model_config.ranked_candidate_num = config.model.ranked_candidate_num
    model_config.pad_token_id = processor.tokenizer.pad_token_id
    model = Qwen2Reward.from_pretrained(
        config.model.reward_model_ckpt_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    for _, param in model.named_parameters():
        param.requires_grad = False

    if config.model.train_vision_model:
        for _, param in model.visual.named_parameters():
            param.requires_grad = True

    if config.model.train_text_model:
        for _, param in model.model.named_parameters():
            param.requires_grad = True

    if config.model.train_value_head:
        for _, param in model.reward_head.named_parameters():
            param.requires_grad = True

    logger.info(f"Creating dataset")
    train_dataset = Qwen2RewardDataset(
        image_root=config.data.image_root,
        data_root=config.data.target_root,
        transform=processor,
    )

    logger.info(f"Dataset length: {len(train_dataset)}")
    logger.info(f"Creating dataloader")

    if not config.is_distribute:
        sampler = None
    else:
        sampler = torch.utils.data.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.device_train_batch_size,
        shuffle=True if sampler is None else False,
        sampler=sampler,
        pin_memory=config.data.pin_memory,
        collate_fn=CustomDataCollatorWithPadding(tokenizer=processor.tokenizer),
        num_workers=config.data.num_workers,
        drop_last=config.data.drop_last,
    )

    if (
        hasattr(model, "gradient_checkpointing_enable")
        and config.activation_checkpointing
    ):
        model.gradient_checkpointing_enable()

    if config.fsdp.enabled:
        if hasattr(model, "get_fsdp_wrap_policy"):
            wrap_policy = model.get_fsdp_wrap_policy()
        elif hasattr(model, "block"):
            wrap_policy = get_fsdp_wrap_policy(type(model.block))
        else:
            wrap_policy = True

        torch.cuda.set_device(f"cuda:{config.local_rank}")

        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            # This prevents any parameters from being initialized twice
            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=get_default_device())

            param_init_fn = dummy_init_fn
        else:
            param_init_fn = None

        model = FSDP(
            model,
            sharding_strategy=config.fsdp.sharding_strategy,
            mixed_precision=config.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=config.fsdp.use_orig_params,
            limit_all_gathers=True,
            device_id=config.local_rank,
            param_init_fn=param_init_fn,
        )

        optimizer = build_optimizer(
            model,
            name=config.optimizer.name,
            lr=config.optimizer.learning_rate,
            wd=config.optimizer.weight_decay,
            filter_bias_and_bn=config.optimizer.decay_norm_and_bias,
        )

    elif config.deepspeed.enabled:
        import deepspeed

        if config.optimizer.name == "adamw":
            parameter = None
            optimizer = None
        else:
            parameter = None
            optimizer = build_optimizer(
                model,
                name=config.optimizer.name,
                lr=config.optimizer.learning_rate,
                wd=config.optimizer.weight_decay,
                filter_bias_and_bn=config.optimizer.decay_norm_and_bias,
                use_shear=config.model.use_shear or config.model.use_lag_shear,
                shear_lr=config.model.shear_lr,
            )

        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=config.deepspeed_init,
            optimizer=optimizer if optimizer is not None else None,
            model_parameters=parameter if parameter is not None else None,
        )
    else:
        optimizer = build_optimizer(
            model,
            name=config.optimizer.name,
            lr=config.optimizer.learning_rate,
            wd=config.optimizer.weight_decay,
            filter_bias_and_bn=config.optimizer.decay_norm_and_bias,
        )

    share_args = {
        "model": model,
        "wandb": config.wandb.enabled,
        "optimizer": optimizer,
        "lr_scheduler": config.scheduler,
        "data_loader": train_loader,
        "work_dir": config.save_folder,
        "max_num_checkpoints": config.checkpoint.save_num_checkpoints_to_keep,
        "log_period": config.console_log_interval,
        "ckpt_period": config.checkpoint.save_interval,
        "clip_grad_norm": config.max_grad_norm,
        "enable_amp": config.autocast_precision != torch.float32,
        "accelerator": config.accelerator,
        "cumulative_iters": config.gradient_accumulation_steps,
        "eval_data_loader": None,
        "is_distributed": config.is_distribute,
        "deepspeed": config.deepspeed.enabled,
        "fsdp": config.fsdp.enabled,
        "torch_compile": config.compile,
        "dtype": config.autocast_precision,
        "save_ckpt_by": config.checkpoint.save_strategy,
        "eval_data_loader": None,
    }

    if config.run_strategy == "step":
        trainer = IterBasedTrainer(max_iters=config.run_duration, **share_args)
    else:
        trainer = EpochBasedTrainer(max_epochs=config.run_duration, **share_args)

    if config.evaluators is not None:
        raise NotImplementedError("Evaluator is not implemented")

    if config.load_path is not None:
        resume_path = config.load_path
    else:
        resume_path = None

    trainer.train(load_checkpoint=resume_path)

    final_ckpt_dir = Path(trainer.ckpt_dir)
    if hasattr(model, "save_pretrained"):
        final_ckpt_dir = final_ckpt_dir / "final"
        final_ckpt_dir.mkdir(exist_ok=True, parents=True)
        model.save_pretrained(final_ckpt_dir)
        processor.save_pretrained(final_ckpt_dir)
        processor.tokenizer.save_pretrained(final_ckpt_dir)


if __name__ == "__main__":
    try:
        if "--local_rank" in sys.argv[1]:
            config_path, other_args = sys.argv[2], sys.argv[3:]
        else:
            config_path, other_args = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise CustomTrainerConfigError(
            f"Usage: [--local_rank] {sys.argv[0]} CONFIG_PATH [OTHER_ARGS]"
        )

    config = CustomTrainConfig.load(config_path, other_args)

    if config.deepspeed.enabled:
        config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
        config.is_distribute = config.world_size > 1
    else:
        config.rank, config.local_rank, config.world_size = init_distributed()
        config.is_distribute = config.world_size > 1

    setup_logger("train", output=config.save_folder, rank=config.rank)
    seed_all(config.seed)

    if config.rank == 0:
        save_path = Path(config.save_folder) / "config.yaml"
        if save_path.is_file() and not config.save_overwrite:
            raise CustomTrainerConfigError(
                f"{save_path} already exists, use save_overwrite=true to overwrite"
            )
        else:
            logger.info(f"Saving config to {save_path}")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            config.save(save_path)
        del save_path

    barrier()

    if (
        config.wandb is not None
        and config.wandb.enabled
        and (config.rank == 0 or not config.wandb.rank_zero_only)
    ):
        if config.wandb.wandb_dir is not None:
            wandb_dir = Path(config.wandb.wandb_dir)
        else:
            wandb_dir = Path(config.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project=config.wandb.project,
            group=config.wandb.group,
            name=config.wandb.name,
            tags=config.wandb.tags,
            job_type=config.wandb.job_type,
            config=config.asdict(exclude=["wandb"]),
        )

    barrier()
    main(config)
