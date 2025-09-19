PROJECT_DIR="" # your project directory to ScoreRS
SCRIPT_PATH=$PROJECT_DIR/python_script/train_clip.py

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

WANDB_ENABLED=true
WANDB_PROJECT="" # your wandb project name
GROUP="" # your wandb group name
NAME="" # your wandb run name

OUTPUT_DIR="" # your output directory
CONFIG_PATH=$PROJECT_DIR/config/clip_training_remoteclip.yaml

IMAGE_COLUMN=filename
TARGET_COLUMN=title
FILTER_RECORD=true
SAVE_TOP_PERCENT=0.7
FILTER_COLUMN=clip_score

unset WANDB_RUN_ID
unset WANDB_RUN_NAME

deepspeed \
    --num_gpus=4 \
    $SCRIPT_PATH \
    $CONFIG_PATH \
    save_overwrite=true \
    data.val_data_path="" # your eval data path
    data.image_path="" # your image path
    data.csv_path="" # your csv path
    data.image_column=$IMAGE_COLUMN \
    data.target_column=$TARGET_COLUMN \
    data.filter_record=$FILTER_RECORD \
    data.save_top_percent=$SAVE_TOP_PERCENT \
    data.filter_column=$FILTER_COLUMN \
    data.num_workers=8 \
    model.model_name="" # your model name
    wandb.enabled=$WANDB_ENABLED \
    wandb.project=$WANDB_PROJECT \
    wandb.group=$GROUP \
    wandb.name=$NAME \
    save_folder=$OUTPUT_DIR \