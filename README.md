# TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards

This repository contains resources for training multi-turn dialogue jailbreak attack models.

## 1. Setup

### 1.1 Create Virtual Environment

```bash
cd TROJail
conda create --name TROJail python=3.12
conda activate TROJail
```

#### 1.1.1.1 trouble shooting
```
当遇到 TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'时
将env和judger model的config.json中添加head_dim字段,其等于hidden_size / sum_attention_heads

当遇到Import Error: cannot import name 'DataProto' from 'verl' (unknown location)
运行
cd verl
pip install -e .
cd ..
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Download Datasets

This project requires the following three datasets:

### 2.1 HarmBench

The HarmBench dataset can be downloaded from Hugging Face:

```bash
huggingface-cli download walledai/HarmBench --local-dir data/
```

### 2.2 StrongREJECT

The StrongREJECT dataset can be downloaded from https://strong-reject.readthedocs.io/.

### 2.3 JailbreakBench

The JailbreakBench dataset can be downloaded from Hugging Face:

```bash
huggingface-cli download JailbreakBench/JBB-Behaviors --local-dir data/
```

## 3. Configure Configuration Files

### 3.1 Configure `config/_7_jailbreak.yaml`

Edit the `config/_7_jailbreak.yaml` file, mainly modifying the `model_path` and `es_manager` sections. Edit the `/ragen/env/jailbreak/config.py` section to set the dataset paths.

Since the configuration files use Hydra, you can set data paths through command-line arguments or by modifying the configuration files.

**Method 1: Set via Command-Line Arguments**

When running `run.sh`, you can pass data paths through command-line arguments.

**Method 2: Directly Modify Configuration Files**

You can also directly modify the corresponding configuration files.

## 4. Run Training

### 4.1 Run Training Script

```bash
bash run.sh
```

### 4.2 Script Execution Flow

The `run.sh` script will execute the following steps:

1. **Start vLLM Services**:
   - Start the environment LLM service (for simulating the target model being attacked)
   - Start the classifier LLM service (for judging whether jailbreak is successful)

2. **Wait for Services to be Ready**:
   - Check if both vLLM services have started

3. **Run Training**:
   - Execute `train.py` for model training
   - Training logs will be saved to `nohup_logs/run_logs/${experiment_name}.log`

4. **Cleanup**:
   - Automatically stop vLLM services after training completes

### 4.4 Monitor Training

During training, you can monitor progress through the following methods:

- **Log Files**: `nohup_logs/run_logs/${experiment_name}.log`
- **vLLM Service Logs**: `env_llm.log` and `judger_llm.log`


## 6. Output Files

After training completes, you can find output files in the following locations:

- **Training Logs**: `nohup_logs/run_logs/${experiment_name}.log`
- **Model Checkpoints**: `checkpoints/jailbreak_grpo`
- **Rollout Data**: `run_logs/${experiment_name}/train_rollout` and `run_logs/${experiment_name}/val_rollout`

## 7. Reference Resources

- **vLLM**: https://github.com/vllm-project/vllm
- **RAGEN**: https://github.com/mll-lab-nu/RAGEN

