PROJECT_DIR="" # your project directory to ScoreRS
CONFIG_PATH=$PROJECT_DIR/config/sft_finetuning.yaml

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export WANDB_PROJECT="" # your wandb project name

unset WANDB_RUN_ID
unset WANDB_RUN_NAME

FORCE_TORCHRUN=1 llamafactory-cli train \
    $CONFIG_PATH \

FORCE_TORCHRUN=1 llamafactory-cli train \
    $CONFIG_PATH \