import logging
from typing import List

from .hook import DistributedHook, HookBase, LoggerHook
from .trainer import Trainer
from .utils import collect_env, is_main_process

logger = logging.getLogger("train")


class EpochBasedTrainer(Trainer):
    def __init__(self, max_epochs: int, **kwargs):
        """
        Args:
            max_epochs (int): Total training epochs.
        """
        super().__init__(**kwargs)
        self.max_epochs = max_epochs

        self.epoch = 0
        self.start_epoch = 0
        self.inner_iter = 0

        if is_main_process() or self.deepspeed:
            self.register_hook(self._build_default_hook())
            logger.info(f"Registered default hooks for main process: {self.registered_hook_names}")

        logger.info("Environment info:\n" + collect_env())

    @property
    def cur_stat(self) -> int:
        return self.cur_iter

    @property
    def max_iters(self) -> int:
        return self.max_epochs * self.epoch_len

    @property
    def cur_iter(self) -> int:
        return self.epoch * self.epoch_len + self.inner_iter

    @property
    def start_iter(self) -> int:
        return self.start_epoch * self.epoch_len

    def _build_default_hook(self) -> List[HookBase]:
        return [
            self.build_ckpt_hook(),
            LoggerHook(self._log_period, tb_log_dir=self.tb_log_dir, use_wandb=self.wandb) if is_main_process() else None,
        ]

    def load_cur_stat(self, value):
        epoch = value // self.epoch_len
        inner_iter = value % self.epoch_len
        self.epoch = epoch
        self.start_epoch = epoch
        self.inner_iter = inner_iter

    def get_specific_hooks(self) -> List[HookBase]:
        return [DistributedHook()]

    def _train_one_epoch(self) -> None:
        self.model.train()
        for self.inner_iter in range(self.inner_iter, self.epoch_len):
            self._call_hooks("before_iter")
            self.train_on_iter()
            self._call_hooks("after_iter")
        self._data_iter = iter(self.data_loader)
        self.inner_iter = 0

    def sub_classes_train(self):
        logger.info(f"Start training from epoch {self.start_epoch} to {self.max_epochs}.")
        self.log_trainable_params()
        for self.epoch in range(self.start_epoch, self.max_epochs):
            self._call_hooks("before_epoch")
            self._train_one_epoch()
            self._call_hooks("after_epoch")
