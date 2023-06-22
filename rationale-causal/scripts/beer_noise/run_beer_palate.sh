#!/bin/bash
model_option=$1


if [[ "$model_option" == "causal-rationale" ]]; then
    python3 -m cr.run_sentiment --run-name beer_causal \
                                 --scale noise\
                                 --dataset-name beer \
                                 --model-type causal_beer_token \
                                 --aspect Palate\
                                 --lr 5e-5 \
                                 --batch_size 32 \
                                 --num_epoch 10 \
                                 --max_length 120\
                                 --grad_accumulation_steps 8 \
                                 --pi 0.1 \
                                 --beta 0.1 \
                                 --tau 0.5\
                                 --k 5\
                                 --eval-interval 200\
                                 --print-every 40\
                                 --mu 0.1\
                                 --device_id 0\
                                 --wandb
                   

else
    echo "model should be causal-rationale"
fi