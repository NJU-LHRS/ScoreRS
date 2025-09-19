import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
import inspect

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


RewardFunction = Callable[[str, str], RewardScore]


class FunctionRewardManager:
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(
                f"Reward function file {config.reward_function} not found."
            )

        spec = importlib.util.spec_from_file_location(
            "custom_reward_fn", config.reward_function
        )
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(
                f"Module {module} does not have function {config.reward_function_name}."
            )

        reward_fn: RewardFunction = getattr(module, config.reward_function_name)
        print(
            f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`."
        )
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_fn_params = inspect.signature(reward_fn).parameters
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(
        self, data: DataProto
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            if "type" in data_item.non_tensor_batch:
                problem = data_item.non_tensor_batch["problem"]
                reward_type = data_item.non_tensor_batch["type"]
                images = data_item.non_tensor_batch["image"]

            if (
                "reward_type" in self.reward_fn_params
                and "type" in data_item.non_tensor_batch
            ):
                score = self.reward_fn(
                    response_str,
                    ground_truth,
                    reward_type=reward_type,
                    problem=problem,
                    images=images,
                )
            else:
                score = self.reward_fn(response_str, ground_truth)
            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
