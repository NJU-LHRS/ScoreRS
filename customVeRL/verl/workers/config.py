from dataclasses import dataclass, field

from .actor import ActorConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig
from .critic import CriticConfig
from .reward import RewardConfig
from .rollout import RolloutConfig


__all__ = [
    "ActorConfig",
    "CriticConfig",
    "FSDPConfig",
    "ModelConfig",
    "OptimConfig",
    "RefConfig",
    "RewardConfig",
    "RolloutConfig",
    "WorkerConfig",
]


@dataclass
class WorkerConfig:
    hybrid_engine: bool = True
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def post_init(self):
        self.ref.micro_batch_size_per_device_for_experience = (
            self.actor.micro_batch_size_per_device_for_experience
        )
        self.ref.padding_free = self.actor.padding_free
        self.ref.ulysses_sequence_parallel_size = (
            self.actor.ulysses_sequence_parallel_size
        )
        self.ref.use_torch_compile = self.actor.use_torch_compile
