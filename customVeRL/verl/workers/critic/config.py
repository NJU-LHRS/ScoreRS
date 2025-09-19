from dataclasses import dataclass, field

from ..actor.config import FSDPConfig, ModelConfig, OffloadConfig, OptimConfig


@dataclass
class CriticConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    micro_batch_size_per_device_for_update: int = 4
    micro_batch_size_per_device_for_experience: int = 16
    max_grad_norm: float = 1.0
    cliprange_value: float = 0.5
    ppo_epochs: int = 1
    padding_free: bool = False
    ulysses_sequence_parallel_size: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    """auto keys"""
    global_batch_size_per_device: int = field(default=-1, init=False)
