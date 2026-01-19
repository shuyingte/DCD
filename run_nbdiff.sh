#!/bin/bash
export WORLD_SIZE=8
export SEED=0
for i in {0..7}; do
	nohup bash -c "CUDA_VISIBLE_DEVICES=$i RANK=$i python eval2.py --cuda-i $i \
    --model_type NBdiff \
    --window_type sliding \
    --batch_size 1 \
    --generate_length 28672 \
    --initial_window_length 8 \
    --block_size 32 \
    --temperature 1.0 \
    --top_p 0.9 \
    --decode_algo entropy-fixed \
    --decode_param 1 \
    --debug_type stat \
    --debug_dir ./debug_dir \
    --expansion_max 16 \
    --expansion_step 4 \
    --expansion_threshold -0.5 | tee log_device_$i.out" &
    pids[$i]=$!
done
for pid in ${pids[*]}; do
    wait $pid
done
sleep 30
wait