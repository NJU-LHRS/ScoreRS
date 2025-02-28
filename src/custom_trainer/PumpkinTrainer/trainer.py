import logging
import os
import os.path as osp
import time
import weakref
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader

from .hook import (
    DeepSpeedHook,
    EpochCheckpointerHook,
    Fp16OptimizerHook,
    GradientCumulativeFp16OptimizerHook,
    GradientCumulativeOptimizerHook,
    HookBase,
    IterCheckpointerHook,
    LoggerHook,
    OptimizerHook,
)
from .hook.lr_scheduler_hook import *
from .utils import (
    MetricStroge,
    gather,
    get_rank,
    get_world_size,
    is_main_process,
    symlink,
)
from .utils.config_parser import (
    CompilerConfig,
    SaveStrategy,
    SchedulerConfig,
    SchedulerType,
    SchedulerUnits,
)

logger = logging.getLogger("train")

SCHEDULER_MAPPING = {
    SchedulerType.step_scheduler: StepLrUpdaterHook,
    SchedulerType.exp_scheduler: ExpLrUpdaterHook,
    SchedulerType.inv_scheduler: InvLrUpdaterHook,
    SchedulerType.cosine_annealing: CosineAnnealingLrUpdaterHook,
    SchedulerType.flat_cosine_annealing: FlatCosineAnnealingLrUpdaterHook,
    SchedulerType.cosine_restart: CosineRestartLrUpdaterHook,
    SchedulerType.cyclic_lr: CyclicLrUpdaterHook,
    SchedulerType.one_cycle: OneCycleLrUpdaterHook,
}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: SchedulerConfig,
        data_loader: DataLoader,
        work_dir: str = "work_dir",
        max_num_checkpoints: int = None,
        log_period: int = 50,
        ckpt_period: int = 1,
        clip_grad_norm: Optional[float] = None,
        enable_amp: bool = False,
        wandb: bool = False,
        accelerator: str = "cpu",
        cumulative_iters: int = 1,
        eval_data_loader: List[DataLoader] = None,
        is_distributed: bool = False,
        torch_compile: CompilerConfig = None,
        deepspeed: bool = False,
        fsdp: bool = False,
        dtype: str = "float32",
        save_ckpt_by: SaveStrategy = SaveStrategy.step,
    ):
        """
        Args:
            model (torch.nn.Module)
            optimizer (torch.optim.Optimizer)
            lr_scheduler (optim.lr_scheduler._LRScheduler)
            data_loader (torch.utils.data.DataLoader): Training data loader.
            work_dir (str): The working directory to save checkpoints and logs.
                Defaults to "work_dir".
            max_num_checkpoints (int): The maximum number of checkpoints to save.
                If None, save all checkpoints. Defaults to None.
            log_period (int): The period (iter-based) to log. Defaults to 50.
            clip_grad_norm (float): Max norm of the gradients. If <= 0, will not clip gradients.
                Defaults to 0.
            enable_amp (bool): Enable the Automatic Mixed Precision (AMP) training.
                Defaults to False.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.eval_data_loader = eval_data_loader
        self.work_dir = work_dir
        self.save_ckpt_by = save_ckpt_by
        self.ckpt_period = ckpt_period
        self.metric_storage = MetricStroge()

        self.inner_iter: int = 0  # [0, epoch_len - 1]
        self.wandb = wandb
        self.torch_compile = torch_compile
        self.deepspeed = deepspeed
        self.fsdp = fsdp

        self.dtype = dtype

        if accelerator == "cpu":
            self.device = torch.device(accelerator)
            self.autocast_type = "cpu"
        elif accelerator == "gpu":
            if is_distributed:
                self.device = torch.device(get_rank())
            else:
                self.device = torch.device("cuda")
            self.autocast_type = "cuda"
        elif accelerator == "mps":
            self.device = torch.device("mps")
            self.autocast_type = "cpu"
        else:
            raise NotImplementedError

        self._hooks: List[HookBase] = []
        self._data_iter = iter(data_loader)
        self._max_num_checkpoints = max_num_checkpoints
        self._log_period = log_period
        self._clip_grad_norm = clip_grad_norm
        self._enable_amp = enable_amp
        self._cumulative_iters = cumulative_iters
        self._is_distributed = is_distributed

    @property
    def registered_hook_names(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.__class__.__name__ for h in self._hooks]

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def epoch_len(self) -> int:
        return len(self.data_loader)

    @property
    def max_iters(self) -> int:
        raise NotImplementedError

    @property
    def cur_stat(self) -> int:
        raise NotImplementedError

    @property
    def cur_iter(self) -> int:
        raise NotImplementedError

    @property
    def start_iter(self) -> int:
        raise NotImplementedError

    @property
    def ckpt_dir(self) -> str:
        return osp.join(self.work_dir, "checkpoints")

    @property
    def tb_log_dir(self) -> str:
        return osp.join(self.work_dir, "tf_logs")

    @property
    def log_file(self) -> str:
        return osp.join(self.work_dir, "log.txt")

    @property
    def model_or_module(self) -> nn.Module:
        if isinstance(self.model, (DataParallel, DistributedDataParallel)) or self.torch_compile is not None:
            return self.model.module
        return self.model

    @property
    def hook_info(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.__class__.__name__ for h in self._hooks]

    def log(self, *args, **kwargs):
        self.metric_storage.update(*args, **kwargs)

    def load_cur_stat(self, value):
        raise NotImplementedError

    def get_specific_hooks(self) -> List[HookBase]:
        raise NotImplementedError

    def _prepare_for_training(self) -> None:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        split_line = "-" * 50
        logger.info(
            f"\n{split_line}\n"
            f"Work directory: {self.work_dir}\n"
            f"Checkpoint directory: {self.ckpt_dir}\n"
            f"Tensorboard directory: {self.tb_log_dir}\n"
            f"Log file: {self.log_file}\n"
            f"{split_line}"
        )

        if self.deepspeed:
            optimizer_hook = DeepSpeedHook()
        elif self._cumulative_iters > 1 and self._enable_amp:
            optimizer_hook = GradientCumulativeFp16OptimizerHook(
                grad_clip=self._clip_grad_norm, cumulative_iters=self._cumulative_iters
            )
        elif self._enable_amp:
            optimizer_hook = Fp16OptimizerHook(grad_clip=self._clip_grad_norm)
        elif self._cumulative_iters > 1:
            optimizer_hook = GradientCumulativeOptimizerHook(
                grad_clip=self._clip_grad_norm, cumulative_iters=self._cumulative_iters
            )
        else:
            optimizer_hook = OptimizerHook(grad_clip=self._clip_grad_norm)

        if self.lr_scheduler.name == SchedulerType.cosine_annealing:
            lr_scheduler_hook = SCHEDULER_MAPPING[self.lr_scheduler.name](
                by_epoch=self.lr_scheduler.units == SchedulerUnits.by_epoch,
                warmup=self.lr_scheduler.t_method,
                warmup_iters=self.lr_scheduler.t_warmup,
                warmup_ratio=self.lr_scheduler.t_factor,
                min_lr=self.lr_scheduler.t_min,
            )
        elif self.lr_scheduler.name == SchedulerType.step_scheduler:
            steps = [int(self.max_iters * x) for x in self.lr_scheduler.step]
            print(steps)
            lr_scheduler_hook = SCHEDULER_MAPPING[self.lr_scheduler.name](
                by_epoch=self.lr_scheduler.units == SchedulerUnits.by_epoch,
                warmup=self.lr_scheduler.t_method,
                warmup_iters=self.lr_scheduler.t_warmup,
                warmup_ratio=self.lr_scheduler.t_factor,
                min_lr=self.lr_scheduler.t_min,
                step=steps,
                gamma=self.lr_scheduler.gamma,
            )

        hooks = self.get_specific_hooks()
        hooks.insert(0, optimizer_hook)
        hooks.insert(1, lr_scheduler_hook)

        self.register_hook(hooks)
        logger.info(f"Registered default hooks for all processes: {self.hook_info}")

    def _set_to_device(self) -> None:
        if not self.deepspeed and not self.fsdp:
            self.model.to(self.device)
            if self._is_distributed:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[get_rank()])

        logger.info("Using %s for training " % self.device)

    def _build_default_hook(self) -> List[HookBase]:
        """
        Build default hooks for the trainer. Should be implemented in subclasses.
        """
        raise NotImplementedError

    def register_hook(self, hooks: List[Optional[HookBase]]) -> None:
        """Register hooks to the trainer.

        The hooks are executed in the order they are registered.

        Args:
            hooks (list[HookBase]): List of hooks.
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
            if self._hooks and isinstance(self._hooks[-1], LoggerHook):
                self._hooks.insert(len(self._hooks) - 1, h)
            else:
                self._hooks.append(h)

    def build_ckpt_hook(self):
        if self.save_ckpt_by == "epoch":
            return EpochCheckpointerHook(self.ckpt_period, self._max_num_checkpoints)
        else:
            return IterCheckpointerHook(self.ckpt_period, self._max_num_checkpoints)

    def save_checkpoint(self, file_name: str) -> None:
        """Save "epoch", "model", "optimizer", "lr_scheduler", "metric_storage",
        "hooks" (optional), "grad_scaler" (optional).

        Args:
            filename (str): The name of the file to save.
        """
        if self.deepspeed:
            data = {
                "cur_stat": self.cur_stat,
                "num_gpus": get_world_size(),
                "metric_storage": self.metric_storage,
            }
        else:
            data = {
                "cur_stat": self.cur_stat,
                "num_gpus": get_world_size(),
                "optimizer": self.optimizer.state_dict(),
                "metric_storage": self.metric_storage,
            }

            if hasattr(self.model_or_module, "custom_save_checkpoint"):
                data["model"] = self.model_or_module.custom_save_checkpoint(file_name)
            elif hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(self.ckpt_dir)
            else:
                data["model"] = self.model_or_module.state_dict()

        hook_states = {h.class_name: h.state_dict() for h in self._hooks if h.checkpointable}
        if hook_states:
            data["hooks"] = hook_states

        file_path = osp.join(self.ckpt_dir, file_name)
        logger.info(f"Saving checkpoint to {file_path}")

        if self.deepspeed:
            self.model.save_checkpoint(self.ckpt_dir, file_name, client_state=data)
        else:
            torch.save(data, file_path)

        if not self.deepspeed:
            dst_file = osp.join(self.ckpt_dir, "latest.pth")
            symlink(file_name, dst_file)

    def load_checkpoint(self, path: str = "", checkpoint: Dict[str, Any] = None):
        """Load the given checkpoint.

        Args:
            checkpoint (dict): The checkpoint to load.
            path (str): Path to the checkpoint. If empty, will not load anything.
                `checkpoint` and `path` can only be specified one.
        """
        assert (checkpoint is not None) ^ (path != "")

        if self.deepspeed:
            _, checkpoint = self.model.load_checkpoint(path)
        else:
            if path:
                logger.info(f"Loading checkpoint from {path} ...")
                checkpoint = torch.load(path, map_location="cpu")

            num_gpus = get_world_size()
            ckpt_num_gpus = checkpoint["num_gpus"]
            assert num_gpus == ckpt_num_gpus, (
                f"You are trying to load a checkpoint trained with {ckpt_num_gpus} GPUs, "
                f"but currently only have {num_gpus} GPUs."
            )

            # load model
            if not hasattr(self.model_or_module, "custom_load_state_dict"):
                incompatible = self.model_or_module.load_state_dict(checkpoint["model"], strict=False)
            else:
                incompatible = self.model_or_module.custom_load_state_dict(path, strict=False)
            if incompatible is not None and incompatible.missing_keys:
                logger.warning(
                    "Encounter missing keys when loading model weights:\n" f"{incompatible.missing_keys}"
                )
            if incompatible is not None and incompatible.unexpected_keys:
                logger.warning(
                    "Encounter unexpected keys when loading model weights:\n" f"{incompatible.unexpected_keys}"
                )

            # load optimizer
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # load epoch
        self.load_cur_stat(checkpoint["cur_stat"] + 1)
        
        # load metric_storage
        self.metric_storage = checkpoint["metric_storage"]
        for key in self.metric_storage._history.keys():
            sum_value = self.metric_storage._history[key]._sum
            if isinstance(sum_value, torch.Tensor):
                self.metric_storage._history[key]._sum = sum_value.detach().cpu().item()

        # load hooks
        hook_states = checkpoint.get("hooks", {})
        hook_names = [h.class_name for h in self._hooks if h.checkpointable]
        missing_keys = [name for name in hook_names if name not in hook_states]
        unexpected_keys = [key for key in hook_states if key not in hook_names]
        if missing_keys:
            logger.warning(f"Encounter missing keys when loading hook state dict:\n{missing_keys}")
        if unexpected_keys:
            logger.warning(f"Encounter unexpected keys when loading hook state dict:\n{unexpected_keys}")

        for key, value in hook_states.items():
            for h in self._hooks:
                if h.class_name == key and h.checkpointable:
                    h.load_state_dict(value)
                    break

    def _call_hooks(self, stage: str) -> None:
        for h in self._hooks:
            getattr(h, stage)()

    def _log_iter_metrics(
        self, loss_dict: Dict[str, torch.Tensor], data_time: float, iter_time: float, lr: float
    ) -> None:
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict.update(data_time=data_time, iter_time=iter_time)
        # gather metrics among all workers for logging
        all_metrics_dict = gather(metrics_dict)

        if is_main_process():
            self.log(self.cur_iter, lr=lr, smooth=False)

            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            self.log(self.cur_iter, data_time=data_time)

            iter_time = np.max([x.pop("iter_time") for x in all_metrics_dict])
            self.log(self.cur_iter, iter_time=iter_time)

            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}

            if "total_loss" in metrics_dict.keys():
                loss_value = metrics_dict.pop("total_loss")
            elif "loss" in metrics_dict.keys():
                loss_value = metrics_dict.pop("loss")
            else:
                loss_value = sum(metrics_dict.values())
                
            self.log(self.cur_iter, total_loss=loss_value)
            if len(metrics_dict) >= 1:
                self.log(self.cur_iter, **metrics_dict)

    def train_on_iter(self) -> None:
        """Train one iteration.

        .. Note::

            Standard PyTorch LR scheduler is epoch-based and called at the end of epoch.
            However, our scheduler is iteration-based, so it should be called after every iteration.

        Subclass :class:`Code.Trainer` and implement your :meth:`train_one_iter`
        to do something fancier.
        """
        iter_start_time = time.perf_counter()
        lr_this_iter = self.lr

        ######################
        # 1. Load batch data #
        ######################
        # we choose to read data by iterator instead of `for data in data_loader`
        # in order to calculate the data loading time
        start = time.perf_counter()
        batch = next(self._data_iter)
        data_time = time.perf_counter() - start

        #####################
        # 2. Calculate loss #
        #####################
        with torch.autocast(device_type=self.autocast_type, enabled=self._enable_amp, dtype=self.dtype):
            batch = self.put_input_to_device(batch)
            outputs = self.model(**batch, return_loss=True)

            loss_dict = {}
            loss_dict["total_loss"] = outputs.loss
            self.loss_dict = loss_dict

        if isinstance(self.loss_dict, torch.Tensor):
            self.loss_dict = {"total_loss": self.loss_dict}

        ###########################
        # 3. Log Metrics
        ###########################
        self._log_iter_metrics(self.loss_dict, data_time, time.perf_counter() - iter_start_time, lr_this_iter)

    def train(self, load_checkpoint: str = None) -> None:
        """
        Start training.
        Parameters
        ----------
        load_checkpoint : str, checkpoint path if resume from checkpoint
        """
        self._prepare_for_training()
        self._set_to_device()
        if load_checkpoint is not None:
            self.load_checkpoint(path=load_checkpoint)

        self._call_hooks("before_train")
        self.sub_classes_train()
        self._call_hooks("after_train")

    def log_trainable_params(self):
        """
        Log trainable parameters.
        """
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # Convert to MB
        total_params = total_params / 1e6
        logger.info(f"Total trainable parameters: {total_params:.2f}M")

    def put_input_to_device(self, modal_input: Union[torch.Tensor, List, Dict, Tuple]):
        if isinstance(modal_input, tuple):
            modal_input = tuple([item.to(device=self.device) for item in modal_input])
        elif isinstance(modal_input, Dict):
            for key, item in modal_input.items():
                if isinstance(item, torch.Tensor):
                    modal_input[key] = item.to(device=self.device)
        else:
            modal_input = modal_input.to(self.device)
        return modal_input

    def sub_classes_train(self):
        """
        True implementation of training process. Depends on the task, this function should be implemented in subclass.
        """
        raise NotImplementedError
