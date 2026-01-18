export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_OFFLINE=1

task=humaneval
length=256
block_length=32
steps=$((length / block_length))

# Baseline (decode one block in one try, very low performance)
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,threshold=0.9 \
--output_path evals_results/baseline/humaneval-ns0-${length} --log_samples

# Dual-cache (bug, threshold mechanism doesn't work)
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

# Prefix-cache + parallel (bug, no mask would be decoded if no logits >= threshold, which causes dead loop.)
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples
