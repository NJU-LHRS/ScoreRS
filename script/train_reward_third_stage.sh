PROJECT_DIR="" # your project directory to ScoreRS
SCRIPT_PATH=$PROJECT_DIR/python_script/train_reward.py

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

WANDB_ENABLED=true
WANDB_PROJECT="" # your wandb project name
GROUP="" # your wandb group name
NAME="" # your wandb run name

OUTPUT_DIR="" # your output directory
CONFIG_PATH="" # your config path

IMAGE_ROOT="" # your image root
TARGET_ROOT="" # your target root
MODEL_PATH="" # your model path
REWARD_MODEL_PATH="" # your reward model path
TRAIN_VALUE_HEAD=true
TRAIN_VISION_MODEL=false
TRAIN_TEXT_MODEL=true

unset WANDB_RUN_ID
unset WANDB_RUN_NAME

deepspeed \
    --num_gpus=8 \
    --master_port=29503 \
    $SCRIPT_PATH \
    $CONFIG_PATH \
    save_overwrite=true \
    data.image_root=$IMAGE_ROOT \
    data.target_root=$TARGET_ROOT \
    data.num_workers=8 \
    model.base_model_path=$MODEL_PATH \
    model.reward_model_ckpt_path=$REWARD_MODEL_PATH \
    model.train_value_head=$TRAIN_VALUE_HEAD \
    model.train_vision_model=$TRAIN_VISION_MODEL \
    model.train_text_model=$TRAIN_TEXT_MODEL \
    wandb.enabled=$WANDB_ENABLED \
    wandb.project=$WANDB_PROJECT \
    wandb.group=$GROUP \
    wandb.name=$NAME \
    save_folder=$OUTPUT_DIR \