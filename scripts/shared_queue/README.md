# Shared Queue Runner

This directory provides a tiny file-based queue for machines that share
`/inspire/hdd/global_user/wanrui-p-wanrui` but cannot reach each other by SSH.

Start this once on the target GPU machine:

```bash
cd /inspire/hdd/global_user/wanrui-p-wanrui/dataset/RealWeb/GUI-RL-verl-latest
export SHARED_JOB_QUEUE=/inspire/hdd/global_user/wanrui-p-wanrui/shared_h800_jobs
mkdir -p "$SHARED_JOB_QUEUE"
nohup /inspire/hdd/global_user/wanrui-p-wanrui/miniconda3/envs/gui_rl_qwen35_verl/bin/python \
  scripts/shared_queue/poll_worker.py \
  --queue-root "$SHARED_JOB_QUEUE" \
  --worker-id h800-8gpu \
  --poll-interval 5 \
  > "$SHARED_JOB_QUEUE/worker.out" 2>&1 &
```

Submit jobs from any machine that can write to the same queue root:

```bash
python scripts/shared_queue/submit_job.py \
  --queue-root /inspire/hdd/global_user/wanrui-p-wanrui/shared_h800_jobs \
  --workdir /inspire/hdd/global_user/wanrui-p-wanrui/dataset/RealWeb/GUI-RL-verl-latest \
  --name my_test \
  --cmd 'echo hello from $(hostname)'
```

Queue state is stored under `pending/`, `running/`, `done/`, `failed/`, and
`logs/`. Job files are moved between state directories, and each job writes one
log file in `logs/`.

For the shared 8xH800 machine, submit GUI-RL Qwen3.5-9B jobs with the 8-GPU
script:

```bash
python scripts/shared_queue/submit_job.py \
  --queue-root /inspire/hdd/global_user/wanrui-p-wanrui/shared_h800_jobs \
  --workdir /inspire/hdd/global_user/wanrui-p-wanrui/dataset/RealWeb/GUI-RL-verl-latest \
  --name qwen35_9b_h800_100step \
  --timeout-sec 0 \
  --cmd 'PATH=/inspire/hdd/global_user/wanrui-p-wanrui/miniconda3/envs/gui_rl_qwen35_verl/bin:$PATH TOTAL_TRAINING_STEPS=100 EXPERIMENT_NAME=qwen3_5_gui_rl_h800_8gpu_100step bash examples/gui_rl/run_qwen3_5_gui_rl_grpo_h800_8gpu.sh'
```
