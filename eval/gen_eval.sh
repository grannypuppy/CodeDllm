# # export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node=4 --master_port=29501 eval/gen_eval.py \
#     --model_path ../CodeDllm/projects/sft_dream_py/5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs/checkpoint-600 \
#     --output_dir results/test_verified/sft_dream_py_len512_steps512_block32_5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs_ckpt600 \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 32 \
#     --use_cache True \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/sft_dream_py_len512_steps512_block32_5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs_ckpt600/output.log

# torchrun --nproc_per_node=4 --master_port=29502 eval/gen_eval.py \
#     --model_path ../dLLM-RL/local_models/dream-7b-instruct \
#     --output_dir results/test_verified/dream-7b-instruct_len512_steps512_block32 \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 32 \
#     --use_cache True \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/dream-7b-instruct_len512_steps512_block32/output.log

# torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval.py \
#     --model_path ../dLLM-RL/local_models/dream-7b-base \
#     --output_dir results/test_verified/dream-7b-base_len512_steps512_block32 \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 32 \
#     --use_cache True \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/dream-7b-base_len512_steps512_block32/output.log


# ##########

# torchrun --nproc_per_node=4 --master_port=29504 eval/gen_eval_ast.py \
#     --model_path ../CodeDllm/projects/sft_dream_py/5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs/checkpoint-600 \
#     --output_dir results/test_verified/sft_dream_py_len512_steps512_block32_ast_5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs_ckpt600 \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 32 \
#     --use_cache True \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/sft_dream_py_len512_steps512_block32_ast_5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs_ckpt600/output.log

# torchrun --nproc_per_node=4 --master_port=29502 eval/gen_eval_ast.py \
#     --model_path ../dLLM-RL/local_models/dream-7b-instruct \
#     --output_dir results/test_verified/dream-7b-instruct_len512_steps512_block32_ast \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 32 \
#     --use_cache True \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/dream-7b-instruct_len512_steps512_block32_ast/output.log

# torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval_ast.py \
#     --model_path ../dLLM-RL/local_models/dream-7b-base \
#     --output_dir results/test_verified/dream-7b-base_len512_steps512_block32_ast \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 32 \
#     --use_cache True \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/dream-7b-base_len512_steps512_block32_ast/output.log

################################

torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval.py \
    --model_path ../dLLM-RL/local_models/dream-7b-instruct \
    --output_dir results/test_verified/dream-7b-instruct_len512_steps512_block512 \
    --data_path data/python_splits/test_verified.jsonl \
    --max_new_tokens 512 \
    --diffusion_steps 512 \
    --block_size 512 \
    --use_cache False \
    --top_p 0.9 \
    --temperature 0.1 \
    --num_generations 4 \
    2>&1 | tee results/test_verified/dream-7b-instruct_len512_steps512_block512/output.log

# torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval.py \
#     --model_path ../dLLM-RL/local_models/dream-7b-base \
#     --output_dir results/test_verified/dream-7b-base_len512_steps512_block512 \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --use_cache False \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/dream-7b-base_len512_steps512_block512/output.log

# torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval.py \
#     --model_path projects/sft_dream_py/online_5e-7lr_1lm_2ga_4gpus_1000traincases_2epochs/checkpoint-250 \
#     --output_dir results/test_verified/sft_dream_online_len512_steps512_block512_ckpt250_gen_eval \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --use_cache False \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/sft_dream_online_len512_steps512_block512_ckpt250_gen_eval/output.log

# torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval.py \
#     --model_path projects/sft_dream_py/5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs/checkpoint-600\
#     --output_dir results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600 \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --use_cache False \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600/output.log

##############################

torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval_ast.py \
    --model_path ../dLLM-RL/local_models/dream-7b-instruct \
    --output_dir results/test_verified/dream-7b-instruct_len512_steps512_block512_ast \
    --data_path data/python_splits/test_verified.jsonl \
    --max_new_tokens 512 \
    --diffusion_steps 512 \
    --block_size 512 \
    --use_cache False \
    --top_p 0.9 \
    --temperature 0.1 \
    --num_generations 4 \
    2>&1 | tee results/test_verified/dream-7b-instruct_len512_steps512_block512_ast/output.log

# torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval_ast.py \
#     --model_path ../dLLM-RL/local_models/dream-7b-base \
#     --output_dir results/test_verified/dream-7b-base_len512_steps512_block512_ast \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --use_cache False \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/dream-7b-base_len512_steps512_block512_ast/output.log

# torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval_ast.py \
#     --model_path projects/sft_dream_py/5e-7lr_2lm_8ga_4gpus_m7_1000traincases_2epochs/checkpoint-600\
#     --output_dir results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600_ast \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --use_cache False \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600_ast/output.log

###############################

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

# torchrun --nproc_per_node=4 --master_port=29503 eval/gen_eval_online_ast.py \
#     --model_path projects/sft_dream_py/online_ast_5e-7lr_1lm_2ga_4gpus_1000traincases_2epochs/checkpoint-250 \
#     --output_dir results/test_verified/sft_dream_online_ast_len512_steps512_block512_ckpt250 \
#     --data_path data/python_splits/test_verified.jsonl \
#     --max_new_tokens 512 \
#     --diffusion_steps 512 \
#     --block_size 512 \
#     --use_cache False \
#     --top_p 0.9 \
#     --temperature 0.1 \
#     --num_generations 4 \
#     2>&1 | tee results/test_verified/sft_dream_online_ast_len512_steps512_block512_ckpt250/output.log