#!/usr/bin/env bash
set -euo pipefail
set -x

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-WARN}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

MODEL_PATH=${MODEL_PATH:-/inspire/hdd/global_user/wanrui-p-wanrui/checkpoint/qwen/Qwen3.5-9B}
TRAIN_FILES=${TRAIN_FILES:-/inspire/hdd/global_user/wanrui-p-wanrui/dataset/RealWeb/all_parquet_high_gui_rl_1103/merged_filtered_high.parquet}
VAL_FILES=${VAL_FILES:-/inspire/hdd/global_user/wanrui-p-wanrui/dataset/RealWeb/all_parquet_high_gui_rl_1103/high_test.parquet}

NNODES=${NNODES:-${PET_NNODES:-2}}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
WORLD_SIZE=$((NNODES * N_GPUS_PER_NODE))

ROLLOUT_TP=${ROLLOUT_TP:-2}
if (( ROLLOUT_TP <= 0 )); then
    echo "ROLLOUT_TP must be positive"
    exit 1
fi
if (( WORLD_SIZE % ROLLOUT_TP != 0 )); then
    echo "WORLD_SIZE (${WORLD_SIZE}) must be divisible by ROLLOUT_TP (${ROLLOUT_TP})"
    exit 1
fi
ROLLOUT_DP=${ROLLOUT_DP:-$((WORLD_SIZE / ROLLOUT_TP))}
if (( ROLLOUT_TP * ROLLOUT_DP != WORLD_SIZE )); then
    echo "ROLLOUT_TP * ROLLOUT_DP must equal WORLD_SIZE (${WORLD_SIZE})"
    exit 1
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_AGENT_WORKERS=${ROLLOUT_AGENT_WORKERS:-8}
DATA_DATALOADER_NUM_WORKERS=${DATA_DATALOADER_NUM_WORKERS:-16}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_PIXELS=${MAX_PIXELS:-3000000}
MIN_PIXELS=${MIN_PIXELS:-262144}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-65536}

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
ROLLOUT_BUCKET_MB=${ROLLOUT_BUCKET_MB:-4096}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-}
ROLLOUT_DISABLE_LOG_STATS=${ROLLOUT_DISABLE_LOG_STATS:-True}
ROLLOUT_CUDAGRAPH_CAPTURE_SIZES=${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-}

ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-False}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}
OPTIMIZER_FOREACH=${OPTIMIZER_FOREACH:-False}

PROJECT_NAME=${PROJECT_NAME:-verl_gui}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_gui_rl_grpo_h200_16gpu}
TRAINER_LOGGERS=${TRAINER_LOGGERS:-'["console"]'}
CKPT_DIR=${CKPT_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TEST_FREQ=${TEST_FREQ:--1}
SAVE_FREQ=${SAVE_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}

if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
    echo "ROLLOUT_MAX_NUM_BATCHED_TOKENS (${ROLLOUT_MAX_NUM_BATCHED_TOKENS}) must be >= MAX_MODEL_LEN (${MAX_MODEL_LEN})"
    exit 1
fi

EXTRA_OVERRIDES=()
if [[ -n "${OPTIMIZER_FOREACH:-}" ]]; then
    EXTRA_OVERRIDES+=(+actor_rollout_ref.actor.optim.override_optimizer_config.foreach="$OPTIMIZER_FOREACH")
fi
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    EXTRA_OVERRIDES+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi
if [[ -n "${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
    EXTRA_OVERRIDES+=(actor_rollout_ref.rollout.cudagraph_capture_sizes="$ROLLOUT_CUDAGRAPH_CAPTURE_SIZES")
fi
if [[ -n "${ROLLOUT_ENABLE_SLEEP_MODE:-}" ]]; then
    EXTRA_OVERRIDES+=(++actor_rollout_ref.rollout.enable_sleep_mode="$ROLLOUT_ENABLE_SLEEP_MODE")
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$TRAIN_BATCH_SIZE" \
    data.dataloader_num_workers="$DATA_DATALOADER_NUM_WORKERS" \
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
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload="$ACTIVATION_OFFLOAD" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_remove_padding=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.freeze_vision_tower=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="$USE_TORCH_COMPILE" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile="$USE_TORCH_COMPILE" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.use_torch_compile="$USE_TORCH_COMPILE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.data_parallel_size="$ROLLOUT_DP" \
    actor_rollout_ref.rollout.n_gpus_per_node="$N_GPUS_PER_NODE" \
    actor_rollout_ref.rollout.nnodes="$NNODES" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
    actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.disable_log_stats="$ROLLOUT_DISABLE_LOG_STATS" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
    +actor_rollout_ref.rollout.limit_images=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$ROLLOUT_BUCKET_MB" \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_WORKERS" \
    critic.enable=False \
    trainer.logger="$TRAINER_LOGGERS" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    "${EXTRA_OVERRIDES[@]}" \
    "$@"
