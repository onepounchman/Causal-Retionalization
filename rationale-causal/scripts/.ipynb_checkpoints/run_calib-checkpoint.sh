#!/bin/bash



                          
python3 -m rrtl.calibration --eval-mode stats\
                          --method causalVIB\
                          --dataset-split test\
                          --dataset-name beer\
                          --pi 0.1\
                          --load-path /home/ec2-user/SageMaker/Causal-Rationale/rationale-causal/experiments/beer/Look/causalVIB/small/mu=1.0-k=5-lr=5e-06-beta=1.0-epoch=10-tau=0.5/model-seed=48-step=4199-acc=91.86.pt
                          
python3 -m rrtl.calibration --eval-mode stats\
                           --method vib\
                           --dataset-split test\
                           --dataset-name beer\
                           --pi 0.1\
                           --load-path /home/ec2-user/SageMaker/Causal-Rationale/rationale-causal/experiments/beer/Look/vib/small/lr=5e-05-beta=1.0-epoch=10-tau=0.1/model-seed=48-step=4799-acc=95.02.pt
                 
# python3 -m rrtl.calibration --eval-mode stats\
#                           --method rnp\
#                           --dataset-split test\
#                           --dataset-name beer\
#                           --load-path /home/ec2-user/SageMaker/Causal-Rationale/rationale-causal/experiments/beer/Look/rnp/small/lr=1e-05-beta=0.1-epoch=10-tau=0.5/model-seed=48-step=4799-acc=93.53.pt
                          
                          
# python3 -m rrtl.calibration --eval-mode stats\
#                           --method SLM\
#                           --dataset-split test\
#                           --dataset-name beer\
#                           --load-path /home/ec2-user/SageMaker/Causal-Rationale/rationale-causal/experiments/beer/Look/SLM/small/lr=5e-05-beta=1.0-epoch=10-tau=0.5/model-seed=48-step=4199-acc=97.02.pt