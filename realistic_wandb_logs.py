#!/usr/bin/env python3
"""
Generate Realistic W&B Logs Based on Real-World Examples

This script creates W&B log files that closely match the structure and content
of actual W&B logs found in public repositories, including the KoAlpaca project
and other ML projects.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
import random

def generate_realistic_logs():
    """Generate realistic W&B log files based on actual examples found online."""
    
    # Create realistic directory
    os.makedirs("realistic_wandb_logs", exist_ok=True)
    
    # 1. Generate realistic wandb-metadata.json based on KoAlpaca example
    metadata = {
        "os": "Linux-5.4.0-124-generic-x86_64-with-glibc2.27",
        "python": "3.10.8", 
        "heartbeatAt": "2023-03-17T11:47:46.443808",
        "startedAt": "2023-03-17T11:47:45.777816",
        "docker": None,
        "cuda": None,
        "args": [
            "--model_name_or_path",
            "/workspace/llama/llama-7b",
            "--data_path", 
            "./alpaca_data.json",
            "--bf16",
            "True",
            "--output_dir",
            "./KoAlpaca",
            "--num_train_epochs", 
            "3",
            "--per_device_train_batch_size",
            "4",
            "--per_device_eval_batch_size", 
            "--gradient_accumulation_steps",
            "8",
            "--evaluation_strategy",
            "no",
            "--save_strategy", 
            "steps",
            "--save_steps",
            "2000",
            "--save_total_limit",
            "1",
            "--learning_rate",
            "2e-5",
            "--weight_decay", 
            "0.",
            "--warmup_ratio",
            "0.03",
            "--lr_scheduler_type",
            "cosine",
            "--logging_steps",
            "1",
            "--logging_strategy",
            "steps",
            "--report_to",
            "wandb",
            "--wandb_project", 
            "koalpaca",
            "--logging_first_step",
            "False",
            "--run_name",
            "koalpaca-polyglot-5.8b-v1.1b"
        ],
        "state": "finished",
        "program": "/home/jovyan/work/train.py",
        "codePathLocal": "train.py",
        "codePath": "train.py", 
        "git": {
            "remote": "https://github.com/Beomi/KoAlpaca.git",
            "commit": "1a2b3c4d5e6f7890abcdef1234567890abcdef12"
        },
        "email": "beomi@kakao.com",
        "root": "/home/jovyan/work",
        "host": "training-server-01.ml.company.com",
        "username": "beomi",
        "executable": "/opt/conda/bin/python",
        "cpu_count": 32,
        "cpu_count_logical": 64,
        "memory": {
            "total": 270582587392,  # ~250GB
            "available": 265318113280
        },
        "gpuapple": None,
        "gpu": "NVIDIA A100-SXM-80GB",
        "gpu_count": 8,
        "slurm": {
            "job_id": "12345",
            "node_list": "gpu-node-[01-02]",
            "partition": "gpu"
        }
    }
    
    with open("realistic_wandb_logs/wandb-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    # 2. Generate realistic config based on LLM training
    config = {
        "_wandb": {
            "desc": None,
            "value": {
                "python_version": "3.10.8",
                "cli_version": "0.14.0",
                "framework": "huggingface",
                "huggingface_version": "4.28.0.dev0",
                "start_time": 1679056065.777816,
                "t": {
                    "1": [1, 5, 11, 49, 55, 71, 98, 103, 107, 111],
                    "2": [1, 5, 11, 49, 55, 71, 98, 103, 107, 111],
                    "3": [13, 16, 23],
                    "4": "3.10.8",
                    "5": "0.14.0",
                    "6": "4.28.0.dev0",
                    "8": ["huggingface"]
                }
            }
        },
        "model_name_or_path": "/workspace/llama/llama-7b",
        "data_path": "./alpaca_data.json",
        "bf16": True,
        "output_dir": "./KoAlpaca", 
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "evaluation_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 2000,
        "save_total_limit": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "logging_strategy": "steps",
        "report_to": "wandb",
        "wandb_project": "koalpaca",
        "logging_first_step": False,
        "run_name": "koalpaca-polyglot-5.8b-v1.1b",
        "seed": 42,
        "data_seed": None,
        "jit_mode_eval": False,
        "use_ipex": False,
        "bf16_full_eval": False,
        "fp16": False,
        "fp16_opt_level": "O1",
        "half_precision_backend": "auto",
        "bf16_full_eval": False,
        "tf32": None,
        "local_rank": 0,
        "ddp_backend": None,
        "ddp_find_unused_parameters": None,
        "ddp_bucket_cap_mb": None,
        "dataloader_drop_last": False,
        "eval_steps": None,
        "eval_delay": 0,
        "past_index": -1,
        "run_name": "koalpaca-polyglot-5.8b-v1.1b",
        "disable_tqdm": None,
        "remove_unused_columns": True,
        "label_names": None,
        "load_best_model_at_end": False,
        "metric_for_best_model": None,
        "greater_is_better": None,
        "ignore_data_skip": False,
        "sharded_ddp": [],
        "fsdp": [],
        "fsdp_min_num_params": 0,
        "fsdp_config": None,
        "fsdp_transformer_layer_cls_to_wrap": None,
        "deepspeed": None,
        "label_smoothing_factor": 0.0,
        "optim": "adamw_hf",
        "optim_args": None,
        "adafactor": False,
        "group_by_length": False,
        "length_column_name": "length",
        "report_to": ["wandb"],
        "ddp_find_unused_parameters": None,
        "ddp_bucket_cap_mb": None,
        "dataloader_pin_memory": True,
        "skip_memory_metrics": True,
        "use_legacy_prediction_loop": False,
        "push_to_hub": False,
        "resume_from_checkpoint": None,
        "hub_model_id": None,
        "hub_strategy": "every_save",
        "hub_token": "<HUB_TOKEN>",
        "hub_private_repo": False,
        "gradient_checkpointing": False,
        "include_inputs_for_metrics": False,
        "auto_find_batch_size": False,
        "full_determinism": False,
        "torchdynamo": None,
        "ray_scope": "last",
        "ddp_timeout": 1800,
        "torch_compile": False,
        "torch_compile_backend": None,
        "torch_compile_mode": None
    }
    
    with open("realistic_wandb_logs/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 3. Generate realistic wandb-history.jsonl for LLM training
    history_entries = []
    base_time = datetime.now()
    
    # Simulate 3000 training steps (typical for LLM fine-tuning)
    for step in range(3000):
        timestamp = base_time + timedelta(seconds=step * 2.5)  # ~2.5 seconds per step
        
        # Realistic LLM training metrics
        train_loss = 2.8 * (0.9995 ** step) + random.uniform(0, 0.1)  # Gradual decrease
        learning_rate = 2e-5 * max(0.1, (1 - step/3000) * (1 + 0.03 * (step/100)))  # Cosine schedule
        
        entry = {
            "_timestamp": timestamp.timestamp(),
            "_runtime": step * 2.5,
            "_step": step,
            "train/epoch": step / 1000,  # 3 epochs total
            "train/global_step": step,
            "train/learning_rate": learning_rate,
            "train/loss": train_loss,
            "train/train_loss": train_loss,
            "train/train_samples_per_second": random.uniform(1.8, 2.2),
            "train/train_steps_per_second": random.uniform(0.45, 0.55),
            "train/total_flos": step * 1.2345e18,  # Floating point operations
            "train/train_runtime": step * 2.5,
            "system/gpu.0.gpu": random.uniform(85, 100),
            "system/gpu.0.memory": random.uniform(75, 85),  
            "system/gpu.0.memoryAllocated": random.uniform(70, 80),
            "system/gpu.0.temp": random.uniform(75, 85),
            "system/gpu.0.powerWatts": random.uniform(380, 420),
            "system/gpu.0.powerPercent": random.uniform(85, 95),
            "system/proc.memory.availableMB": random.uniform(200000, 220000),
            "system/proc.memory.rssMB": random.uniform(45000, 55000),
            "system/proc.memory.percent": random.uniform(18, 22),
            "system/proc.cpu.threads": 128,
            "system/disk.in": random.uniform(0, 100),
            "system/disk.out": random.uniform(500, 1500),
            "system/disk.percent": random.uniform(45, 55),
            "system/network.sent": random.uniform(1000000, 5000000),
            "system/network.recv": random.uniform(500000, 2000000)
        }
        
        # Add periodic evaluation metrics
        if step % 500 == 0 and step > 0:
            entry.update({
                "eval/loss": train_loss + random.uniform(0.1, 0.3),
                "eval/perplexity": 2 ** (train_loss + random.uniform(0.1, 0.3)),
                "eval/runtime": 120 + random.uniform(-20, 20),
                "eval/samples_per_second": random.uniform(8, 12),
                "eval/steps_per_second": random.uniform(2, 3)
            })
        
        history_entries.append(entry)
    
    with open("realistic_wandb_logs/wandb-history.jsonl", "w") as f:
        for entry in history_entries:
            f.write(json.dumps(entry) + "\n")
    
    # 4. Generate realistic wandb-summary.json
    summary = {
        "train/epoch": 3.0,
        "train/global_step": 3000,
        "train/learning_rate": 0.0,
        "train/loss": 1.234,
        "train/total_flos": 3.7035e21,
        "train/train_loss": 1.234,
        "train/train_runtime": 7500.0,
        "train/train_samples_per_second": 2.048,
        "train/train_steps_per_second": 0.512,
        "eval/loss": 1.456,
        "eval/perplexity": 4.29,
        "eval/runtime": 118.5,
        "eval/samples_per_second": 10.2,
        "eval/steps_per_second": 2.55,
        "_runtime": 7500,
        "_timestamp": datetime.now().timestamp(),
        "_step": 3000,
        "system/gpu.0.gpu": 95.2,
        "system/gpu.0.memory": 78.5,
        "system/gpu.0.temp": 82.1,
        "system/proc.memory.percent": 20.3,
        "_wandb": {
            "runtime": 7500
        }
    }
    
    with open("realistic_wandb_logs/wandb-summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # 5. Generate realistic requirements.txt
    requirements = """torch==2.0.0
transformers==4.28.0.dev0
datasets==2.10.1
accelerate==0.18.0
wandb==0.14.0
numpy==1.24.2
tokenizers==0.13.2
huggingface-hub==0.13.3
tqdm==4.65.0
packaging==23.0
pyyaml==6.0
requests==2.28.2
safetensors==0.3.0
scikit-learn==1.2.2
scipy==1.10.1
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2
tensorboard==2.12.1
protobuf==4.22.1
psutil==5.9.4
GPUtil==1.4.0
"""
    
    with open("realistic_wandb_logs/requirements.txt", "w") as f:
        f.write(requirements.strip())
    
    # 6. Generate realistic output.log
    log_entries = [
        "03/17/2023 11:47:45 - INFO - __main__ - Training/evaluation parameters:",
        "TrainingArguments(",
        "_n_gpu=8,",
        "adafactor=False,", 
        "adam_beta1=0.9,",
        "adam_beta2=0.999,",
        "adam_epsilon=1e-08,",
        "auto_find_batch_size=False,",
        "bf16=True,",
        f"data_path=./alpaca_data.json,",
        "dataloader_drop_last=False,",
        "dataloader_num_workers=0,",
        "dataloader_pin_memory=True,",
        "ddp_backend=None,",
        "ddp_find_unused_parameters=None,",
        "deepspeed=None,",
        "disable_tqdm=None,",
        "do_eval=False,",
        "do_predict=False,",
        "do_train=True,",
        "eval_accumulation_steps=None,",
        "eval_delay=0,",
        "eval_steps=None,",
        "eval_strategy=no,",
        "fp16=False,",
        "fp16_backend=auto,",
        "fp16_full_eval=False,",
        "fp16_opt_level=O1,",
        "fsdp=[],",
        "full_determinism=False,",
        "gradient_accumulation_steps=8,",
        "gradient_checkpointing=False,",
        "greater_is_better=None,",
        "group_by_length=False,",
        "half_precision_backend=auto,",
        "hub_model_id=None,",
        "hub_private_repo=False,",
        "hub_strategy=every_save,",
        "hub_token=<HUB_TOKEN>,",
        "ignore_data_skip=False,",
        "include_inputs_for_metrics=False,",
        "jit_mode_eval=False,",
        "label_names=None,",
        "label_smoothing_factor=0.0,",
        "learning_rate=2e-05,",
        "length_column_name=length,",
        "load_best_model_at_end=False,",
        "local_rank=0,",
        "log_level=passive,",
        "log_level_replica=warning,",
        "log_on_each_node=True,",
        "logging_dir=./KoAlpaca/runs/Mar17_11-47-45_training-server-01,",
        "logging_first_step=False,",
        "logging_nan_inf_filter=True,",
        "logging_steps=1,",
        "logging_strategy=steps,",
        "lr_scheduler_type=cosine,",
        "max_grad_norm=1.0,",
        "max_steps=-1,",
        "metric_for_best_model=None,",
        "mp_parameters=,",
        "num_train_epochs=3.0,",
        "optim=adamw_hf,",
        "optim_args=None,",
        f"output_dir=./KoAlpaca,",
        "overwrite_output_dir=False,",
        "past_index=-1,",
        "per_device_eval_batch_size=4,",
        "per_device_train_batch_size=4,",
        "prediction_loss_only=False,",
        "push_to_hub=False,",
        "push_to_hub_model_id=None,",
        "push_to_hub_organization=None,",
        "push_to_hub_token=<PUSH_TO_HUB_TOKEN>,",
        "ray_scope=last,",
        "remove_unused_columns=True,",
        f"report_to=['wandb'],",
        "resume_from_checkpoint=None,",
        f"run_name=koalpaca-polyglot-5.8b-v1.1b,",
        "save_on_each_node=False,",
        "save_only_model=False,",
        "save_safetensors=False,",
        "save_steps=2000,",
        "save_strategy=steps,",
        "save_total_limit=1,",
        "seed=42,",
        "sharded_ddp=[],",
        "skip_memory_metrics=True,",
        "tf32=None,",
        "torch_compile=False,",
        "torch_compile_backend=None,",
        "torch_compile_mode=None,",
        "torchdynamo=None,",
        "tpu_metrics_debug=False,",
        "tpu_num_cores=None,",
        "use_ipex=False,",
        "use_legacy_prediction_loop=False,",
        "use_mps_device=False,",
        "warmup_ratio=0.03,",
        "warmup_steps=0,",
        "weight_decay=0.0,",
        ")",
        "",
        "03/17/2023 11:47:46 - INFO - __main__ - Loading dataset from ./alpaca_data.json",
        "03/17/2023 11:47:47 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/jovyan/.cache/huggingface/datasets/json/default-abc123/0.0.0/cache-def456.arrow",
        "03/17/2023 11:47:48 - INFO - __main__ - Dataset loaded: 52002 examples",
        "03/17/2023 11:47:48 - INFO - __main__ - Tokenizing dataset...",
        "03/17/2023 11:49:23 - INFO - __main__ - Tokenization complete. Max length: 2048",
        "03/17/2023 11:49:24 - INFO - __main__ - Starting training...",
        "",
        "03/17/2023 11:49:25 - INFO - transformers.trainer - The following columns in the training set don't have a corresponding argument in `LlamaForCausalLM.forward` and have been ignored: instruction, input, output. If instruction, input, output are not expected by `LlamaForCausalLM.forward`,  you can safely ignore this message.",
        "03/17/2023 11:49:25 - INFO - transformers.trainer - ***** Running training *****",
        "03/17/2023 11:49:25 - INFO - transformers.trainer -   Num examples = 52,002",
        "03/17/2023 11:49:25 - INFO - transformers.trainer -   Num Epochs = 3",
        "03/17/2023 11:49:25 - INFO - transformers.trainer -   Instantaneous batch size per device = 4",
        "03/17/2023 11:49:25 - INFO - transformers.trainer -   Total train batch size (w. parallel, distributed & accumulation) = 256",
        "03/17/2023 11:49:25 - INFO - transformers.trainer -   Gradient Accumulation steps = 8",
        "03/17/2023 11:49:25 - INFO - transformers.trainer -   Total optimization steps = 609",
        "",
        f"03/17/2023 11:49:26 - INFO - wandb.wandb_torch - Watching model with wandb.",
        f"03/17/2023 11:49:26 - INFO - wandb.wandb_run - Run initialized: beomi/koalpaca/run-{uuid.uuid4().hex[:8]}",
        f"03/17/2023 11:49:26 - INFO - wandb.wandb_run - Logging to https://wandb.ai/beomi/koalpaca/runs/{uuid.uuid4().hex[:8]}",
        "",
        "03/17/2023 11:49:27 - INFO - transformers.trainer - Training started!",
        "03/17/2023 11:49:30 - INFO - transformers.trainer - {'loss': 2.1234, 'learning_rate': 1.9876e-05, 'epoch': 0.01, 'step': 1}",
        "03/17/2023 11:49:33 - INFO - transformers.trainer - {'loss': 2.1145, 'learning_rate': 1.9852e-05, 'epoch': 0.01, 'step': 2}",
        "03/17/2023 11:49:36 - INFO - transformers.trainer - {'loss': 2.1067, 'learning_rate': 1.9829e-05, 'epoch': 0.01, 'step': 3}",
        "...",
        "",
        f"03/17/2023 14:32:15 - INFO - transformers.trainer - Saving model checkpoint to ./KoAlpaca/checkpoint-2000",
        f"03/17/2023 14:32:15 - INFO - transformers.tokenization_utils_base - tokenizer config file saved in ./KoAlpaca/checkpoint-2000/tokenizer_config.json",
        f"03/17/2023 14:32:15 - INFO - transformers.tokenization_utils_base - Special tokens file saved in ./KoAlpaca/checkpoint-2000/special_tokens_map.json",
        "",
        "03/17/2023 17:14:28 - INFO - transformers.trainer - ***** Training completed *****",
        "03/17/2023 17:14:28 - INFO - transformers.trainer - Total training time: 7500.123 seconds",
        "03/17/2023 17:14:28 - INFO - transformers.trainer - Training loss: 1.234",
        f"03/17/2023 17:14:29 - INFO - __main__ - Model saved to ./KoAlpaca/",
        f"03/17/2023 17:14:30 - INFO - wandb.wandb_run - Run finished successfully. View at https://wandb.ai/beomi/koalpaca/runs/{uuid.uuid4().hex[:8]}"
    ]
    
    with open("realistic_wandb_logs/output.log", "w") as f:
        f.write("\n".join(log_entries))
    
    # 7. Generate conda-environment.yaml
    conda_env = """name: koalpaca-env
channels:
  - pytorch
  - nvidia
  - huggingface
  - conda-forge
  - defaults
dependencies:
  - python=3.10.8
  - pytorch=2.0.0
  - pytorch-cuda=11.7
  - cudatoolkit=11.7
  - numpy=1.24.2
  - pip=23.0.1
  - pip:
    - transformers==4.28.0.dev0
    - datasets==2.10.1
    - accelerate==0.18.0
    - wandb==0.14.0
    - tokenizers==0.13.2
    - huggingface-hub==0.13.3
    - safetensors==0.3.0
    - scikit-learn==1.2.2
    - scipy==1.10.1
    - pandas==1.5.3
    - matplotlib==3.7.1
    - seaborn==0.12.2
    - tensorboard==2.12.1
    - protobuf==4.22.1
    - psutil==5.9.4
    - GPUtil==1.4.0
prefix: /opt/conda/envs/koalpaca-env
"""
    
    with open("realistic_wandb_logs/conda-environment.yaml", "w") as f:
        f.write(conda_env)
    
    print("Realistic W&B log files generated in 'realistic_wandb_logs/' directory:")
    print("- wandb-metadata.json (based on KoAlpaca project)")
    print("- config.json (LLM fine-tuning configuration)")
    print("- wandb-history.jsonl (3000 training steps with GPU metrics)")
    print("- wandb-summary.json (final training summary)")
    print("- requirements.txt (ML/LLM dependencies)")
    print("- output.log (detailed training logs)")
    print("- conda-environment.yaml (conda environment)")
    
    print("\n🔍 This data contains REAL-WORLD sensitive information like:")
    print("- Email: beomi@kakao.com")
    print("- Username: beomi") 
    print("- Host: training-server-01.ml.company.com")
    print("- File paths: /workspace/llama/llama-7b, /home/jovyan/work/")
    print("- Git repo: https://github.com/Beomi/KoAlpaca.git")
    print("- Slurm job info and GPU node details")
    print("- Hub tokens and sensitive configuration")
    
    print("\n✨ Perfect for testing the anonymization script with realistic data!")

if __name__ == "__main__":
    generate_realistic_logs()
