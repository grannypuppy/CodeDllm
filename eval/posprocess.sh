python utils/postprocess_pl.py \
    --results_dir results/test_verified/dream-7b-base_len512_steps512_block512   \
    --output_file results/test_verified/dream-7b-base_len512_steps512_block512/generation_results_processed.jsonl

python utils/postprocess_pl.py \
    --results_dir results/test_verified/dream-7b-base_len512_steps512_block512_ast   \
    --output_file results/test_verified/dream-7b-base_len512_steps512_block512_ast/generation_results_processed.jsonl


python utils/postprocess_pl.py \
    --results_dir results/test_verified/dream-7b-instruct_len512_steps512_block512   \
    --output_file results/test_verified/dream-7b-instruct_len512_steps512_block512/generation_results_processed.jsonl

python utils/postprocess_pl.py \
    --results_dir results/test_verified/dream-7b-instruct_len512_steps512_block512_ast   \
    --output_file results/test_verified/dream-7b-instruct_len512_steps512_block512_ast/generation_results_processed.jsonl

# python utils/postprocess_pl.py \
#     --results_dir results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600   \
#     --output_file results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600/generation_results_processed.jsonl

# python utils/postprocess_pl.py \
#     --results_dir results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600_ast   \
#     --output_file results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600_ast/generation_results_processed.jsonl