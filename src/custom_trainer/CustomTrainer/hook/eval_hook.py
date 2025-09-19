import logging
from typing import List

import torch
import wandb
from tqdm import tqdm

from ..utils import get_rank
from .hookbase import HookBase
from .logger_hook import LoggerHook

logger = logging.getLogger("train")


class EvalHook(HookBase):
    """Run an evaluation function periodically.

    It is executed every ``period`` epochs and after the last epoch.
    """

    func_mapping = {
        "perplexity": "perplexity_eval_func",
    }

    def __init__(self, period: int, evaluators: List[str]):
        """
        Args:
            period (int): The period to run ``eval_func``. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_func (callable): A function which takes no arguments, and
                returns a dict of evaluation metrics.
        """
        super(EvalHook, self).__init__()
        self._period = period
        self._eval_func_bank = []
        for evals in evaluators:
            assert evals in self.func_mapping, f"evaluators {evals} not be supported"
            self._eval_func_bank.append(getattr(self, self.func_mapping[evals]))

    def _eval_func(self):
        for eval_func in self._eval_func_bank:
            eval_func()

        self.trainer.model.train()

    def perplexity_eval_func(self):
        self.trainer.model.eval()
        total_tokens = 0
        total_ppl = 0

        for batch in tqdm(
            self.trainer.eval_data_loader, desc="Evaluating", disable=get_rank() != 0
        ):
            batch = self.trainer.put_input_to_device(batch)
            new_tokens = batch["input_ids"].shape[1]
            inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
            with (
                torch.autocast(
                    device_type=self.trainer.autocast_type,
                    enabled=self.trainer._enable_amp,
                    dtype=self.trainer.dtype,
                )
                and torch.inference_mode()
            ):
                outputs = self.trainer.model(**inputs)
            loss = outputs.loss
            ppl = loss.exp().item()

            total_ppl += ppl * new_tokens
            total_tokens += new_tokens

        ppl = total_ppl / total_tokens
        logger.info(f"{self.trainer.cur_iter}: ppl: {ppl:.4f}")
        for hook in self.trainer._hooks:
            if isinstance(hook, LoggerHook):
                hook._tb_writer.add_scalar("eval/ppl", ppl)
                if hook.wandb:
                    wandb.log({"eval/ppl": ppl})
                break


class EpochEvalHook(EvalHook):
    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._eval_func()


class IterEvalHook(EvalHook):
    def after_iter(self):
        if self.every_n_iters(self._period) or self.is_last_iter():
            self._eval_func()
