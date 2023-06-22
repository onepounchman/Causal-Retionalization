#!/bin/bash
model_option=$1


if [[ "$model_option" == "causalVIB" ]]; then
    python3 -m cr.run_beer --run-name hotel_causalVIB \
                                 --scale normal\
                                 --dataset-name hotel \
                                 --model-type causal_hotel_token \
                                 --aspect Location\
                                 --lr 5e-5 \
                                 --batch_size 16 \
                                 --num_epoch 1 \
                                 --max_length 150\
                                 --grad_accumulation_steps 16 \
                                 --pi 0.1 \
                                 --beta 0.1 \
                                 --tau 1\
                                 --k 5\
                                 --eval-interval 400\
                                 --print-every 80\
                                 --mu 0.1\
                                 --device_id 0\
                                 --wandb
                                                               

else
    echo "model should be causal-rationale"
fi