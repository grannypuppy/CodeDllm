uv run python eval/batch_eval.py     --input_file results/baseline/baseline.jsonl     --output_dir results     --run_name baseline/baseline --use_evalperf

RUN_NAME=dream-7b-base_len512_steps512_block512
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

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

RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt800_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt800
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

RUN_NAME=sft_dream_multitask_len512_steps512_block512_ckpt600
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_multitask_ast_len512_steps512_block512_ckpt600_ast
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


RUN_NAME=sft_dream_py_ast_len512_steps512_block512_ckpt600_ast_noeos
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt600_ast_noeos
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_multitask_len512_steps512_block512_ckpt1000_totalmean_0-5_codeonly
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt1000
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt1200_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt1200
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_multitask_len512_steps512_block512_ckpt1000_ast_ex_totalmean_0-5_codeonly
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_py_len512_steps512_block512_ckpt1000_ast
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_multitask_len512_steps512_block512_ckpt1000_random
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}

RUN_NAME=sft_dream_multitask_len512_steps512_block512_ckpt1000_totalmean_0-5_codeonly_astorigin
uv run python eval/batch_eval.py --use_evalperf \
    --input_file results/test_verified/${RUN_NAME}/generation_results_processed.jsonl \
    --output_dir results \
    --run_name test_verified/${RUN_NAME}