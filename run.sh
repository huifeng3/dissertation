env_llm_path="./models/Mistral-7B-Instruct-v0.3"
judger_llm_path="./models/HarmBench-Mistral-7b-val-cls"
model_path="./models/Qwen2.5-3B-Instruct"

env_llm_port=8001
env_llm_base_url="http://localhost:$env_llm_port/v1"
judger_llm_port=8002
judger_llm_base_url="http://localhost:$judger_llm_port/v1"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting env_llm on port $env_llm_port..."
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model $env_llm_path \
  --port $env_llm_port \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 13312 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 51200 \
  --tensor-parallel-size 1 \
  --dtype half \
  --enforce-eager \
  --trust-remote-code \
  > env_llm.log 2>&1 &

env_llm_pid=$!

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting judger_llm on port $judger_llm_port..."
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
  --model $judger_llm_path \
  --port $judger_llm_port \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2048 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 51200 \
  --tensor-parallel-size 1 \
  --dtype half \
  --enforce-eager \
  --trust-remote-code \
  > judger_llm.log 2>&1 &

judger_llm_pid=$!


echo "[$(date +'%Y-%m-%d %H:%M:%S')] Waiting for vLLM servers to be ready..."
timeout=300
counter=0
while ! nc -z localhost $env_llm_port; do
  sleep 5
  counter=$((counter + 5))
  if [ $counter -ge $timeout ]; then
    echo "Error: env_llm server failed to start within $timeout seconds."
    kill $env_llm_pid 2>/dev/null
    kill $judger_llm_pid 2>/dev/null
    exit 1
  fi
  # Check if process is still running
  if ! kill -0 $env_llm_pid 2>/dev/null; then
    echo "Error: env_llm process died unexpectedly. Check env_llm.log for details."
    kill $judger_llm_pid 2>/dev/null
    exit 1
  fi
done

counter=0
while ! nc -z localhost $judger_llm_port; do
  sleep 5
  counter=$((counter + 5))
  if [ $counter -ge $timeout ]; then
    echo "Error: judger_llm server failed to start within $timeout seconds."
    kill $env_llm_pid 2>/dev/null
    kill $judger_llm_pid 2>/dev/null
    exit 1
  fi
  # Check if process is still running
  if ! kill -0 $judger_llm_pid 2>/dev/null; then
    echo "Error: judger_llm process died unexpectedly. Check judger_llm.log for details."
    kill $env_llm_pid 2>/dev/null
    exit 1
  fi
done
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Both vLLM servers are ready!"


experiment_name_2="train_demo"
TENSORBOARD_DIR_2="./tensorboard_log/${experiment_name_2}"
mkdir -p "$TENSORBOARD_DIR_2"
mkdir -p nohup_logs/run_logs

# Set Ray temp dir to local project directory to avoid permission issues

#ray_temp_path="/home/zkf/dissertation/RAGEN/temp/ray"
ray_temp_path="/data1/TROJail/RAGEN/temp/ray"
export RAY_TMPDIR=$(pwd)/ray_temp
mkdir -p $RAY_TMPDIR

#similarity_model_path="/home/zkf/dissertation/models/all-MiniLM-L6-v2"
similarity_model_path="/data1/TROJail/models/all-MiniLM-L6-v2"


echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting training: ${experiment_name_2}..."
RAY_TEMP_PATH=$ray_temp_path SIMILARITY_MODEL_PATH=$similarity_model_path python train.py --config-name _7_jailbreak.yaml \
  model_path=$model_path env_llm.model_path=$env_llm_path judger_llm.model_path=$judger_llm_path env_llm.base_url=$env_llm_base_url judger_llm.base_url=$judger_llm_base_url \
  algorithm.heuristic_process_adv_lambda=0.1 \
  # experiment_name=${experiment_name_2} trainer.total_training_steps=260 trainer.test_freq=10 \
  experiment_name=${experiment_name_2} trainer.total_training_steps=100 trainer.test_freq=10 \
  env_llm.max_tokens=512 \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.01 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  2>&1 | tee nohup_logs/run_logs/${experiment_name_2}.log

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training ${experiment_name_2} finished."


echo "[$(date +'%Y-%m-%d %H:%M:%S')] Killing vLLM servers..."
kill $env_llm_pid
kill $judger_llm_pid

echo "vLLM servers stopped."
