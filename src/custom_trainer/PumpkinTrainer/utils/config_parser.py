from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from .error_message import CustomTrainerConfigError
from .misc import PathOrStr, StrEnum, is_url

__all__ = [
    "ActivationCheckpointingStrategy",
    "CompilerConfig",
    "ModelConfig",
    "OptimizerType",
    "OptimizerConfig",
    "SchedulerType",
    "SchedulerConfig",
    "DataConfig",
    "EvaluatorConfig",
    "TrainConfig",
    "SpeedMonitorConfig",
    "WandbConfig",
    "CompilerConfig",
    "WandbConfig",
    "FSDPPrecision",
    "FSDPWrapStrategy",
    "FSDPConfig",
    "CheckpointType",
    "SaveStrategy",
]

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


class BaseConfig:
    @classmethod
    def _register_resolvers(cls, validate_paths: bool = True):
        # Expands path globs into a list.
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = sorted(glob(path))
                if not matches and validate_paths:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        # Chooses the first path in the arguments that exists.
        def path_choose(*paths) -> str:
            for path in paths:
                if is_url(path) or Path(path).exists():
                    return path
            if validate_paths:
                raise FileNotFoundError(", ".join(paths))
            else:
                return ""

        # Finds the latest checkpoint in a folder.
        def path_last_checkpoint(path) -> str:
            from .misc import auto_resume_helper

            latest_checkpoint = auto_resume_helper(path)
            if latest_checkpoint is None:
                if validate_paths:
                    raise FileNotFoundError(
                        f"Could not find a latest checkpoint at {path}"
                    )
                else:
                    return ""
            else:
                return str(latest_checkpoint)

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)
        om.register_new_resolver(
            "path.last_checkpoint", path_last_checkpoint, replace=True
        )

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        """
        Update the legacy config settings whose schemas have undergone backwards-incompatible changes.
        """
        return config

    @classmethod
    def new(cls: Type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise CustomTrainerConfigError(str(e))

    @classmethod
    def load(
        cls: Type[C],
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        cls._register_resolvers(validate_paths=validate_paths)
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))
            if key is not None:
                raw = raw[key]  # type: ignore
            raw = cls.update_legacy_settings(raw)
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise CustomTrainerConfigError(str(e))

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out


@dataclass
class ModelConfig(BaseConfig):
    pass


class OptimizerType(StrEnum):
    lionw = "lionw"
    adamw = "adamw"
    adanp = "adanp"


@dataclass
class OptimizerConfig(BaseConfig):
    name: OptimizerType = OptimizerType.adamw
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)

    decay_norm_and_bias: bool = False
    decay_embeddings: bool = False
    metrics_log_interval: Optional[int] = None

    def __post_init__(self):
        self.betas = tuple(self.betas)  # type: ignore[assignment]

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "name") and new_config.name == "decoupled_lionw":
                new_config.name = "lionw"
                if hasattr(new_config, "eps"):
                    del new_config.eps

        return new_config


class SchedulerType(StrEnum):
    step_scheduler = "step_scheduler"
    exp_scheduler = "exp_scheduler"
    inv_scheduler = "inv_scheduler"
    cosine_annealing = "cosine_annealing"
    flat_cosine_annealing = "flat_cosine_annealing"
    cosine_restart = "cosine_restart"
    cyclic_lr = "cyclic_lr"
    one_cycle = "one_cycle"


class SchedulerUnits(StrEnum):
    by_epoch = "epoch"
    by_step = "step"
    by_token = "token"


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_annealing
    units: SchedulerUnits = SchedulerUnits.by_step

    # basic for warmup
    t_warmup: Union[int, float] = 100
    t_method: Optional[Union[str]] = "linear"
    t_factor: Optional[float] = 0.1
    t_min: Optional[float] = 0.000001

    # step_scheduler
    step: List[float] = field(default_factory=lambda: [1.0])
    gamma: float = 0.1

    grad_clip_warmup_steps: Optional[Union[int, float]] = None
    """
    The warmup period for which the max grad norm (or norm ratio) will be set to its
    warmup value of `max_grad_norm * grad_clip_warmup_factor`.
    """

    grad_clip_warmup_factor: Optional[float] = None
    """
    The ratio of the max allowed gradient norm (or norm ratio) for clipping during the warmup period
    vs after the warmup period.
    """


@dataclass
class DataConfig(BaseConfig):
    data: Optional[str] = None
    num_workers: int = 4
    drop_last: bool = False
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0


class EvaluatorType(StrEnum):
    perplexity = "perplexity"
    clip_cls = "clip_cls"


class EvaluatorStrategy(StrEnum):
    epoch = "epoch"
    step = "step"


@dataclass
class EvaluatorConfig(BaseConfig):
    type: List[EvaluatorType] = field(
        default_factory=lambda: [EvaluatorType.perplexity]
    )
    eval_interval: int = 1000
    eval_strategy: EvaluatorStrategy = EvaluatorStrategy.step


@dataclass
class WandbConfig(BaseConfig):
    enabled: bool = False
    wandb_dir: Optional[str] = None
    project: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["watching"])
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1


@dataclass
class SpeedMonitorConfig(BaseConfig):
    window_size: int = 100
    gpu_flops_available: Optional[Union[float, int]] = None


@dataclass
class CompilerConfig(BaseConfig):
    mode: Optional[str] = None
    """
    The mode to compile the model in. At the moment this can be "default",
    "reduce-overhead" (useful for smaller models/batches), or "max-autotune"
    (the fastest for larger models, but takes a long time to compile).
    """

    fullgraph: bool = False
    """
    Whether it is OK to break model into several subgraphs when compiling.
    Note that this is not compatible with FSDP.
    """

    backend: str = "inductor"
    """
    The backend to use.
    """


class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    """
    Wrap each block with its own FSDP instance.
    """

    by_block_and_size = "by_block_and_size"
    """
    Like 'by_block' but `wte` and `ff_out` will be wrapped separately as well.
    """

    by_block_group = "by_block_group"
    """
    Wrap each block group together into its own FSDP instance.
    This requires :attr:`~ModelConfig.block_group_size` to be bigger than 1.
    """

    by_block_group_and_size = "by_block_group_and_size"
    """
    Like 'by_block_group' but `wte` and `ff_out` will be wrapped separately as well.
    """

    size_based = "size_based"
    """
    Used PyTorch's default size-based auto wrap policy.
    """

    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_five = "one_in_five"


class FSDPPrecision(StrEnum):
    pure = "pure"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, ``reduce_dtype``,
    and ``buffer_dtype`` all set to the autocast precision data type.
    """

    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """


@dataclass
class FSDPConfig(BaseConfig):
    enabled: bool = False
    use_orig_params: bool = True
    """
    This must be ``True`` if using ``compile`` or you want to track the parameter norm during training.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    wrapping_strategy: Optional[FSDPWrapStrategy] = None
    """
    The wrapping strategy to use. If ``None``, the default, the model is wrapped with a single top-level
    FSDP instance.
    """

    precision: FSDPPrecision = FSDPPrecision.pure


@dataclass
class DeepSpeedConfig(BaseConfig):
    enabled: bool = False
    """
    Whether to use DeepSpeed for training.
    """

    stage: int = 0
    """
    The DeepSpeed stage to use.
    """

    offload_optimizer: bool = False
    """
    Whether to offload the optimizer to the CPU.
    """

    offload_param: bool = False
    """
    Whether to offload the model parameters to the CPU.
    """


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"


class ShardedCheckpointerType(StrEnum):
    torch_new = "torch_new"
    torch_legacy = "torch_legacy"
    local = "local"


class TrainStrategy(StrEnum):
    epoch = "epoch"
    step = "step"


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    """
    Checkpoint every transformer layer.
    """

    one_in_two = "one_in_two"
    """
    Checkpoint one in two transformer layers.
    """

    one_in_three = "one_in_three"
    """
    Checkpoint one in three transformer layers.
    """

    one_in_four = "one_in_four"
    """
    Checkpoint one in four transformer layers.
    """

    fine_grained = "fine_grained"
    """
    Focus checkpointing on where it is cheap to recompute and saves most memory.
    """


class SaveStrategy(StrEnum):
    epoch = "epoch"
    step = "step"


@dataclass
class CheckpointConfig(BaseConfig):
    save_strategy: SaveStrategy = SaveStrategy.step
    """
    The strategy to use for saving checkpoints.
    """

    save_interval: int = 1000
    """
    How often (in terms of steps) to save sharded training state checkpoints.
    """

    save_interval_unsharded: Optional[int] = None
    """
    How often (if at all) to save unsharded training state checkpoint.
    For large models it can be costly to save these, so it usually makes sense to save
    these less often than regular (sharded) training checkpoints.
    """

    save_interval_ephemeral: Optional[int] = None
    """
    How often (if at all) to save ephemeral sharded checkpoints. These checkpoints are the same
    as those saved every `save_interval` except that at most only the most recent one of these is kept.
    This is useful when you want to checkpoint often for restarts in case of failures, but don't
    want to keep the majority of these checkpoints.

    For example, suppose you want to keep your checkpoints at every 1000 steps, but you also want to save
    a temporary checkpoint every 100 steps in case your job fails. In that case you would
    set `save_interval=1000` and `save_interval_ephemeral=100`.
    """

    save_num_checkpoints_to_keep: int = -1
    """
    How many sharded checkpoints to keep.
    """

    save_num_unsharded_checkpoints_to_keep: int = -1
    """
    How many unsharded checkpoints to keep.
    """

    force_save_unsharded: bool = False
    """
    Save an unsharded checkpoint before training (even during a dry run).
    Use this option with `--load-path={PATH}` and `--dry_run` to convert a sharded
    checkpoint into an unsharded checkpoint.
    """


@dataclass
class TrainConfig(BaseConfig):
    """
    Training configuration.
    """

    run_name: Optional[str] = None
    """
    The name of the run.
    """

    seed: int = 6198
    """
    Used to seed all initial RNG states.
    """

    run_strategy: TrainStrategy = TrainStrategy.step
    """
    The training strategy to use.
    """

    run_duration: Optional[int] = 1
    """
    The duration of the training run.
    """

    dry_run: bool = False
    """
    If ``True``, don't actually train.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """
    Model configuration.
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """
    Optimizer configuration.
    """

    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    """
    DeepSpeed configuration.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """
    Learning rate scheduler configuration.
    """

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """
    Checkpoint configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """
    Training data configuration.
    """

    restore_dataloader: bool = True
    """
    When restarting, restore the data loader to where it left off.
    If you restarting in order to train on a different dataset, set this to ``False``.
    """

    fast_forward_batches: Optional[int] = None
    """
    When restarting, use this to fast-forward the dataloader beyond the last checkpoint.
    This can be useful when restarting due to a loss spike in order to skip the data that
    corresponded to the spike.
    """

    evaluators: Optional[EvaluatorConfig] = None
    """
    Evaluation configurations.
    """

    save_folder: str = "./"
    """
    The directory to save checkpoints to.
    """

    save_overwrite: bool = False
    """
    If ``True``, overwrite any conflicting checkpoint files.
    """

    load_path: Optional[str] = None
    """
    The path to a training checkpoint to restore/resume from.

    Note that you can make use of the "path.last_checkpoint" Omegaconfig YAML resolver here, which takes
    a local or remote directory and resolves to the latest checkpoint (sharded or unsharded) in that directory.
    For example,

    ```bash
    --load_path='${path.last_checkpoint:s3://ai2-llm/checkpoints/7b/v1_5-mix-run-001}'
    ```
    """

    load_path_sharded_checkpointer: Optional[ShardedCheckpointerType] = None
    """
    The sharded checkpointer type to use to load the initial checkpoint from ``load_path``.
    """

    reset_optimizer_state: bool = False
    """
    When this is set, we restore the model from a checkpoint (if given), but we leave the optimizer uninitialized.
    We also set a new learning rate schedule that does a new warmup, such that it intercepts the original learning
    curve (according to the current learning rate schedule settings), and continues from there.
    """

    reset_trainer_state: bool = False
    """
    When this is set we don't restore the trainer state from a checkpoint.
    """

    sharded_checkpointer: ShardedCheckpointerType = ShardedCheckpointerType.torch_legacy
    """
    The name of the sharded checkpointer to use to save (sharded) checkpoints throughout training.
    """

    device_train_batch_size: int = 16
    """
    Don't set this manually. This will be set to ``global_train_batch_size // world_size``.
    """

    device_eval_batch_size: int = 16
    """
    The number of evaluation instances passed to the model in a single forward pass on each device.
    """

    max_grad_norm: Optional[float] = None
    """
    Clip gradient norms to this value if set.
    """

    precision: Optional[str] = None
    """
    Precision to train with (e.g. "amp_bf16", "amp_fp16", or "fp32").
    """

    wandb: Optional[WandbConfig] = None
    """
    Weights & Biases configuration.
    """

    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    """
    Speed monitor configuration.
    """

    console_log_interval: int = 1
    """
    How often to log to the console.
    """

    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    """
    Fully sharded data parallel settings.
    """

    time_limit: Optional[float] = 60 * 60 * 47.5
    """
    The maximum amount of time to train for before saving a checkpoint and ending early.
    On LUMI we have 48 hours max per job, so we default to just under 48 hours to give us time
    to write out a final checkpoint.
    """

    early_stopping_factor: Optional[float] = None

    save_data_indices: bool = True
    """
    Save training data indices from each batch for each worker.
    """

    python_profiling: bool = False
    """
    Whether to run the Python profiler on batches 6, 7, and 8.
    """

    torch_profiling: bool = False
    """
    Whether to run the PyTorch profiler on batches 6, 7, and 8.
    """

    stop_at: Optional[int] = None
    """
    Stop at a specific step.
    """

    activation_checkpointing: Optional[ActivationCheckpointingStrategy] = None
    """
    The activation checkpointing strategy to use.
    """

    gradient_accumulation_steps: int = 1
    """
    Number of steps to accumulate gradients before performing an optimizer step.
    """

    accelerator_type: Optional[str] = "cpu"
    """
    The accelerator to use for training.
    ["cuda", "cpu", "mps"]
    """

    local_rank: Optional[int] = None
    """
    The local rank of the process.
    """

    @property
    def accelerator(self) -> str:
        if self.accelerator_type == "gpu":
            return "gpu"
        elif self.accelerator_type == "cpu":
            return "cpu"
        elif self.accelerator_type == "mps":
            return "mps"
        else:
            raise ValueError(f"Unexpected accelerator type '{self.accelerator_type}'")

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    @property
    def fsdp_precision(self) -> MixedPrecision:
        if self.fsdp.precision == FSDPPrecision.pure:
            return MixedPrecision(
                param_dtype=self.autocast_precision,
                reduce_dtype=self.autocast_precision,
                buffer_dtype=self.autocast_precision,
            )
        elif self.fsdp.precision == FSDPPrecision.mixed:
            return MixedPrecision(
                param_dtype=self.autocast_precision,
                reduce_dtype=torch.float32,
                buffer_dtype=self.autocast_precision,
            )
        else:
            raise NotImplementedError(f"{self.fsdp.precision}")

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "activation_checkpointing"):
                if new_config.activation_checkpointing is False:
                    new_config.activation_checkpointing = None
                if new_config.activation_checkpointing is True:
                    new_config.activation_checkpointing = (
                        ActivationCheckpointingStrategy.whole_layer
                    )

            if hasattr(new_config, "optimizer"):
                new_config.optimizer = OptimizerConfig.update_legacy_settings(
                    new_config.optimizer
                )

        return new_config

    @property
    def deepspeed_init(self):
        ds_config = {}
        if self.deepspeed.enabled:
            if self.optimizer.name == OptimizerType.adamw:
                optimizer = {
                    "type": "AdamW",
                    "params": {
                        "lr": self.optimizer.learning_rate,
                        "eps": 1e-6,
                        "betas": self.optimizer.betas,
                        "weight_decay": self.optimizer.weight_decay,
                    },
                }
                ds_config["optimizer"] = optimizer
            else:
                ds_config["zero_allow_untested_optimizer"] = True
                ds_config["zero_force_ds_cpu_optimizer"] = False

            if self.autocast_precision == torch.float16:
                ds_config["fp16"] = {
                    "enabled": True,
                    "initial_scale_power": 16,
                    "loss_scale_window": 500,
                    "auto_cast": False,
                }
            elif self.autocast_precision == torch.bfloat16:
                ds_config["bf16"] = {"enabled": True, "auto_cast": False}

            zero_optimization = {
                "stage": self.deepspeed.stage,
                "sub_group_size": 1e9,
                "contiguous_gradients": True,
                "overlap_comm": False,
                "stage3_gather_16bit_weights_on_model_save": True,
            }

            if self.deepspeed.offload_optimizer:
                zero_optimization["offload_optimizer"] = {"device": "cpu"}

            if self.deepspeed.offload_param:
                zero_optimization["offload_param"] = {"device": "cpu"}

            ds_config["zero_optimization"] = zero_optimization
            ds_config["gradient_accumulation_steps"] = self.gradient_accumulation_steps
            ds_config["gradient_clipping"] = self.max_grad_norm
            ds_config["train_micro_batch_size_per_gpu"] = self.device_train_batch_size

        return ds_config
