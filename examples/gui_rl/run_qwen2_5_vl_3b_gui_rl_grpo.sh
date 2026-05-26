#!/usr/bin/env bash
set -euo pipefail
set -x

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-WARN}

MODEL_PATH=${MODEL_PATH:-/inspire/hdd/global_user/wanrui-p-wanrui/checkpoint/qwen/Qwen2.5-VL-3B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-/inspire/hdd/global_user/wanrui-p-wanrui/dataset/RealWeb/all_parquet_high_gui_rl_1103/merged_filtered_high.parquet}
VAL_FILES=${VAL_FILES:-/inspire/hdd/global_user/wanrui-p-wanrui/dataset/RealWeb/all_parquet_high_gui_rl_1103/high_test.parquet}

NNODES=${NNODES:-${PET_NNODES:-1}}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TP=${TP:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_AGENT_WORKERS=${ROLLOUT_AGENT_WORKERS:-8}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_PIXELS=${MAX_PIXELS:-3000000}
MIN_PIXELS=${MIN_PIXELS:-262144}

PROJECT_NAME=${PROJECT_NAME:-verl_gui}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_3b_gui_rl_grpo_latest_verl}
TRAINER_LOGGERS=${TRAINER_LOGGERS:-'["console"]'}
CKPT_DIR=${CKPT_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    data.custom_cls.path=pkg://verl.utils.dataset.gui_rl_dataset \
    data.custom_cls.name=GUIRLDataset \
    data.image_key=images \
    +data.max_pixels="$MAX_PIXELS" \
    +data.min_pixels="$MIN_PIXELS" \
    +data.gui_rl_data_source=gui_rl \
    reward.custom_reward_function.path=pkg://verl.utils.reward_score.r1gui \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    reward.reward_manager.name=naive \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.freeze_vision_tower=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.limit_images=1 \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_WORKERS" \
    critic.enable=False \
    trainer.logger="$TRAINER_LOGGERS" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=5 \
    trainer.total_epochs=1 \
    "$@"
