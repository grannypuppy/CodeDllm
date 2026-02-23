### train/ 是训练脚本

#### 现在使用的是：

1. 正常sft： `sft_dream_dataset_m.py` (构造mask) + `sft_dream_train.py` （sft训练）

2. online sft: `sft_dream_online.py`

3. ast online sft: `sft_dream_online_ast.py`

+ train/其他脚本都是以前的draft

#### 训练的模型保存在 projects/下面

1. 正常sft的在 `projects/sft_dream_py/5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs`

2. online的在

`projects/sft_dream_py/online_5e-7lr_1lm_2ga_4gpus_1000traincases_2epochs`

`projects/sft_dream_py/online_ast_5e-7lr_1lm_2ga_4gpus_1000traincases_2epochs`

### eval/ 是generation 脚本

#### generation_utils都定义在models下面

+ dream/ & dream_online/

+ `generation_utils.py`  origin的diffusion loop
+ `generation_utils_block.py`  支持block的
+ `generation_utils_ast.py`  origin的diffusion loop + AST weight
+ `generation_utils_block_ast.py`  支持block的 + AST weight

`generation_utils_block_ast_full.py` 是 draft 

#### benchmark generation 脚本是
+ `gen_eval.py` 
+ `gen_eval_ast.py`
+ `gen_eval_online.py`
+ `gen_eval_online_ast.py`

script样例 `gen_eval.sh`

use_cache = False 就是没有导入block generation

生成的结果在 `results/test_verified`

对于生成结果 应该是 `generation_results_rank*.jsonl`

先用 `CodeDllm/utils/postprocess_pl.py`合成一个`generation_results_processed.jsonl` （`eval/posprocess.sh`）

再用 `eval/batch_eval.py` 运行evaluate output

最后用 `analysis.py` (analysis.sh) 分析结果

--------------

1. CodeDllm/results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt100

这个结果是用： CodeDllm/projects/sft_dream_py/online_v1_1e-6lr_1lm_2ga_4gpus_1000traincases_4epochs/checkpoint-100

生成结果的运行命令:

```bash
torchrun --nproc_per_node=4 --master_port=29501 eval/gen_eval_online.py \
    --model_path projects/sft_dream_py/online_v1_1e-6lr_1lm_2ga_4gpus_1000traincases_4epochs/checkpoint-100 \
    --output_dir results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt100 \
    --data_path data/python_splits/test_verified.jsonl \
    --max_new_tokens 512 \
    --diffusion_steps 512 \
    --block_size 512 \
    --use_cache False \
    --top_p 0.9 \
    --temperature 0.1 \
    --num_generations 4 \
    2>&1 | tee results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt100/output.log
```

这个模型的训练命令和参数配置:

```bash
accelerate launch --config_file accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml     train/sft_dream_online_v1.py config=configs/sft_dream_py_online.yaml
``` (其中 sft_dream_online_v1.py 中 USE_GROUPED_LOSS = True )

2. CodeDllm/results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt250

这个结果是用： CodeDllm/results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt250

生成结果的运行命令：

```bash
torchrun --nproc_per_node=4 --master_port=29501 eval/gen_eval_online.py \
    --model_path projects/sft_dream_py/online_v1_1e-6lr_1lm_2ga_4gpus_1000traincases/checkpoint-250 \
    --output_dir results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt250 \
    --data_path data/python_splits/test_verified.jsonl \
    --max_new_tokens 512 \
    --diffusion_steps 512 \
    --block_size 512 \
    --use_cache False \
    --top_p 0.9 \
    --temperature 0.1 \
    --num_generations 4 \
    2>&1 | tee results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt250/output.log
```

```bash
accelerate launch --config_file accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml     train/sft_dream_online_v1.py config=configs/sft_dream_py_online_1.yaml
``` (其中 sft_dream_online_v1.py 中 USE_GROUPED_LOSS = False )

3. CodeDllm/results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt350

这个结果是用： CodeDllm/results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt350

生成结果的运行命令：

```bash
torchrun --nproc_per_node=4 --master_port=29501 eval/gen_eval_online.py \
    --model_path projects/sft_dream_py/online_v1_1e-6lr_1lm_2ga_4gpus_1000traincases/checkpoint-350 \
    --output_dir results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt350 \
    --data_path data/python_splits/test_verified.jsonl \
    --max_new_tokens 512 \
    --diffusion_steps 512 \
    --block_size 512 \
    --use_cache False \
    --top_p 0.9 \
    --temperature 0.1 \
    --num_generations 4 \
    2>&1 | tee results/test_verified/sft_dream_online_v1_len512_steps512_block512_ckpt350/output.log
```

```bash
accelerate launch --config_file accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml     train/sft_dream_online_v1.py config=configs/sft_dream_py_online_2.yaml
``` (其中 sft_dream_online_v1.py 中 USE_GROUPED_LOSS = True )