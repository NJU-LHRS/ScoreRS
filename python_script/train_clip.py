import logging
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

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
from src.dataset.dataset_class import ImageTextDataset
from src.evaluation.clip_cls import EpochCLIPClsEvalHook, IterCLIPClsEvalHook
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.datasets import ImageFolder
from transformers import (
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizerFast,
    default_data_collator,
)

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger("train")


@dataclass
class CustomDataConfig(DataConfig):
    image_path: Optional[str] = ""
    csv_path: Optional[str] = ""
    val_data_path: Optional[str] = ""
    filter_record: Optional[bool] = False
    save_top_percent: Optional[float] = 0.3
    filter_column: Optional[str] = "clip_score"
    image_column: Optional[str] = "filename"
    target_column: Optional[str] = "title"


@dataclass
class CustomModelConfig(ModelConfig):
    model_name: Optional[str] = ""


@dataclass
class CustomTrainConfig(TrainConfig):
    model: CustomModelConfig = field(default_factory=CustomModelConfig)
    data: CustomDataConfig = field(default_factory=CustomDataConfig)


def hack_hf_image_transform(image_processor: CLIPImageProcessor, image: Image.Image):
    image = image_processor(image, return_tensors="pt")["pixel_values"].squeeze()
    return image


def main(config: TrainConfig):
    logger.info(f"Creating model")

    model = CLIPModel.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        config.model.model_name,
    )
    tokenizer = CLIPTokenizerFast.from_pretrained(
        config.model.model_name,
    )

    logger.info(f"Creating dataset")
    if "skyscript" in config.data.image_path.lower():
        train_dataset = ImageTextDataset(
            data_root=config.data.image_path,
            target_root=config.data.csv_path,
            target_type="csv",
            transform=image_processor,
            tokenizer=tokenizer,
            filter_record=config.data.filter_record,
            save_top_percent=config.data.save_top_percent,
            filter_column=config.data.filter_column,
            target_column=config.data.target_column,
            image_column=config.data.image_column,
        )
    elif "remoteclip" in config.data.image_path.lower():
        train_dataset = ImageTextDataset(
            data_root=config.data.image_path,
            target_root=config.data.csv_path,
            target_type="csv",
            transform=image_processor,
            tokenizer=tokenizer,
            filter_record=config.data.filter_record,
            save_top_percent=config.data.save_top_percent,
            filter_column=config.data.filter_column,
            target_column=config.data.target_column,
            image_column=config.data.image_column,
        )

    eval_loader = {}

    for val_data_path in Path(config.data.val_data_path).glob("*_Image"):
        dataset = ImageFolder(
            val_data_path,
            transform=partial(hack_hf_image_transform, image_processor),
        )
        classes_name = dataset.classes

        data_name = val_data_path.name.split("_")[0]
        eval_loader[data_name] = {
            "dataloader": torch.utils.data.DataLoader(
                dataset,
                batch_size=config.device_eval_batch_size,
                shuffle=False,
                sampler=None,
                pin_memory=config.data.pin_memory,
                num_workers=config.data.num_workers,
            ),
            "classes_name": classes_name,
        }

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
        collate_fn=default_data_collator,
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
        "eval_data_loader": eval_loader,
    }

    if config.run_strategy == "step":
        trainer = IterBasedTrainer(max_iters=config.run_duration, **share_args)
    else:
        trainer = EpochBasedTrainer(max_epochs=config.run_duration, **share_args)

    if config.evaluators is not None:
        if config.evaluators.eval_strategy == "step":
            trainer.register_hook(
                [
                    IterCLIPClsEvalHook(
                        evaluators=config.evaluators.type,
                        period=config.evaluators.eval_interval,
                        tokenizer=tokenizer,
                    )
                ]
            )
        else:
            trainer.register_hook(
                [
                    EpochCLIPClsEvalHook(
                        evaluators=config.evaluators.type,
                        period=config.evaluators.eval_interval,
                        tokenizer=tokenizer,
                    )
                ]
            )

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
