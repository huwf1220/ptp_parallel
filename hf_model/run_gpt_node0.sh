#!/bin/bash
export OMP_NUM_THREADS=1
NODE_RANK=0
NNODES=1
GPUS_PER_NODE=8
# MASTER_ADDR="192.168.0.172"
MASTER_ADDR=localhost
MASTER_PORT=6006
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
                    --node_rank $NODE_RANK \
                    --master_addr $MASTER_ADDR \
                    --master_port $MASTER_PORT"



for i in 6
do
  torchrun $DISTRIBUTED_ARGS gpt_dp.py  \
        --batch $i \
        --iter 80 \
        --config GPT_config/gpt2_6.7B_2layers.json \
        --logdir log  >> gpt
done

