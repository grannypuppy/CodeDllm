# RUN_NAME=dream-7b-base_len512_steps512_block512
# uv run python eval/batch_eval.py --use_evalperf \
#     --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
#     --output_dir results \
#     --run_name test_verified/${RUN_NAME}

RUN_NAME=dream-7b-base_len512_steps512_block512_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=dream-7b-instruct_len512_steps512_block512
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=dream-7b-instruct_len512_steps512_block512_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_multitask_len512_steps512_block512_ckpt1500
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}


RUN_NAME=sft_dream_multitask_ast_len512_steps512_block512_ckpt1500_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_py_ast_len512_steps512_block512_ckpt1500_bkp
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}


RUN_NAME=sft_dream_py_ast_len512_steps512_block512_ckpt1500_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}


RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt1500
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}


RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt1500_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}


RUN_NAME=sft_dream_py_ast_len512_steps512_block512_ckpt600
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}


RUN_NAME=sft_dream_py_ast_len512_steps512_block512_ckpt600_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}


RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt600
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}


RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt600_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}