# math500 -> --include_path tasks_pro
# DreamBase -> remove apply_chat_template
# humaneval[0-shot] mbpp[3-shot] gsm8k[5-shot] math500[0-shot] ifeval[0-shot]
# Llada/Dream

# Initialize GPUS
nvidia-smi
accelerate
export CUDA_VISIBLE_DEVICES=0

# Define Parameters
MODELS_A=("Llada" "Dream")
MODELS_B=("FastDLLMv2")
TASKS=("gsm8k" "math500" "mbpp" "humaneval" "ifeval")
GENLEN=512

# Loop
for TASK in "${TASKS[@]}"; do
for MODEL in "${MODELS_A[@]}"; do

echo "ATTENTION! RUN TASK ${TASK} MODEL ${MODEL}!"

################
# FIXED-LENGTH WINDOW, NO CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},lookahead=1,window_type=static,max_window_length=${GENLEN},generate_length=${GENLEN},block_size=32,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=none,refresh_count=32,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_static_none_${GENLEN}/" --output_path "evals_results_${MODEL}/${TASK}_static_none_${GENLEN}/" --log_samples --batch_size 1 --apply_chat_template

# FIXED-LENGTH WINDOW, PREFIX CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},lookahead=1,window_type=static,max_window_length=${GENLEN},generate_length=${GENLEN},block_size=32,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=prefix,refresh_count=32,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_static_prefix_${GENLEN}/" --output_path "evals_results_${MODEL}/${TASK}_static_prefix_${GENLEN}/" --log_samples --batch_size 1 --apply_chat_template

# FIXED-LENGTH WINDOW, DUAL CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},lookahead=1,window_type=static,max_window_length=${GENLEN},generate_length=${GENLEN},block_size=32,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=dual,refresh_count=32,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_static_dual_${GENLEN}/" --output_path "evals_results_${MODEL}/${TASK}_static_dual_${GENLEN}/" --log_samples --batch_size 1 --apply_chat_template

################
# SLIDING WINDOW WITH MAX LENGTH 128, NO CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},lookahead=1,window_type=sliding,max_window_length=128,generate_length=${GENLEN},initial_window_length=16,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=none,refresh_count=32,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_sliding_lim128_none_${GENLEN}/" --output_path "evals_results_${MODEL}/${TASK}_sliding_lim128_none_${GENLEN}/" --log_samples --batch_size 1 --apply_chat_template

# SLIDING WINDOW WITH MAX LENGTH 128, PREFIX CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks $TASK --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},lookahead=1,window_type=sliding,max_window_length=128,generate_length=${GENLEN},initial_window_length=16,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=prefix-delay2,refresh_count=32,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_sliding_lim128_prefixdelay2_${GENLEN}/" --output_path "evals_results_${MODEL}/${TASK}_sliding_lim128_prefixdelay2_${GENLEN}/" --log_samples --batch_size 1 --apply_chat_template

# SLIDING WINDOW WITH MAX LENGTH 128, DUAL CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},lookahead=1,window_type=sliding,max_window_length=128,generate_length=${GENLEN},initial_window_length=16,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=dual-delay2,refresh_count=32,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_sliding_lim128_dualdelay2_${GENLEN}/" --output_path "evals_results_${MODEL}/${TASK}_sliding_lim128_dualdelay2_${GENLEN}/" --log_samples --batch_size 1 --apply_chat_template

# Process
python llada/postprocess_all.py .

done

for MODEL in "${MODELS_B[@]}"; do

echo "ATTENTION! RUN TASK ${TASK} MODEL ${MODEL}!"

# SLIDING WINDOW, DUAL CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},window_type=sliding,generate_length=1024,initial_window_length=8,block_size=32,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=dual-delay2,refresh_count=8,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_sliding_dualdelay2/" --output_path "evals_results_${MODEL}/${TASK}_sliding_dualdelay2/" --log_samples --batch_size 1 --apply_chat_template

# SLIDING WINDOW, NO CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},window_type=sliding,generate_length=1024,initial_window_length=8,block_size=32,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=none,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_sliding_none/" --output_path "evals_results_${MODEL}/${TASK}_sliding_none/" --log_samples --batch_size 1 --apply_chat_template

# FIXED-LENGTH WINDOW, DUAL CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},window_type=static,generate_length=1024,initial_window_length=8,block_size=32,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=dual,refresh_count=1024,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_static_dual/" --output_path "evals_results_${MODEL}/${TASK}_static_dual/" --log_samples --batch_size 1 --apply_chat_template

# FIXED-LENGTH WINDOW, NO CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args model_type=${MODEL},window_type=static,generate_length=1024,initial_window_length=8,block_size=32,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=none,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_static_none/ --output_path evals_results_${MODEL}/${TASK}_static_none/ --log_samples --batch_size 1 --apply_chat_template

# BLOCK CACHE
accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --num_processes 1 eval.py --tasks ${TASK} --confirm_run_unsafe_code --model model --model_args "model_type=${MODEL},window_type=static,generate_length=1024,initial_window_length=1024,block_size=32,temperature=0.0,decode_algo=threshold,decode_param=0.9,cache_type=none,debug_type=full,debug_dir=evals_results_${MODEL}/${TASK}_block/" --output_path "evals_results_${MODEL}/${TASK}_block/" --log_samples --batch_size 1 --apply_chat_template

python llada/postprocess_all.py .

done
done