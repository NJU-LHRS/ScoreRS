set -x

MODEL_PATH=""  # replace it with your local file path
REWARD_MODEL_PATH="" # your reward model path
REWARD_MODEL_SERVER_URL=http://localhost:8000/v1/rewards
FORMAT_WEIGHT=0.3
TENSOR_PARALLEL_SIZE=1
FORMAT_PROMPT="" # your format prompt path

MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=4096

NAME="" # your experiment name

unset WANDB_RUN_ID
unset WANDB_RUN_NAME

CKPT_PATH="" # your checkpoint path
VAL_FREQ=20
SAVE_FREQ=20

TOTAL_EPISODES=2

python3 -m customVeRL.verl.trainer.main \
    config=customVeRL/examples/config.yaml \
    data.train_files="" # your train files path
    data.val_files="" # your val files path
    data.image_key=image
    data.format_prompt=${FORMAT_PROMPT}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${NAME} \
    trainer.project_name="" # your project name
    trainer.save_checkpoint_path=${CKPT_PATH}/${NAME} \
    trainer.n_gpus_per_node=7 \
    trainer.val_freq=${VAL_FREQ} \
    trainer.save_freq=${SAVE_FREQ} \
    worker.reward.reward_function_kwargs.reward_model_name=${REWARD_MODEL_PATH} \
    worker.reward.reward_function_kwargs.reward_server_url=${REWARD_MODEL_SERVER_URL} \
    worker.reward.reward_function_kwargs.format_weight=${FORMAT_WEIGHT} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.rollout.max_num_batched_tokens=12288 \
    trainer.total_episodes=${TOTAL_EPISODES}