python eval/batch_eval.py \
    --input_file results/test_verified/dream-7b-instruct_len512_steps512_block512/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/dream-7b-instruct_len512_steps512_block512

python eval/batch_eval.py \
    --input_file results/test_verified/dream-7b-base_len512_steps512_block512/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/dream-7b-base_len512_steps512_block512

python eval/batch_eval.py \
    --input_file results/test_verified/dream-7b-instruct_len512_steps512_block512_ast/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/dream-7b-instruct_len512_steps512_block512_ast 

python eval/batch_eval.py \
    --input_file results/test_verified/dream-7b-base_len512_steps512_block512_ast/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/dream-7b-base_len512_steps512_block512_ast 

python eval/batch_eval.py \
    --input_file results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/sft_dream_py_len512_steps512_block512_ckpt600 

python eval/batch_eval.py \
    --input_file results/test_verified/sft_dream_py_len512_steps512_block512_ckpt600_ast/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/sft_dream_py_len512_steps512_block512_ckpt600_ast 