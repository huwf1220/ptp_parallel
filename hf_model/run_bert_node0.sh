#!/bin/bash
export OMP_NUM_THREADS=1
NODE_RANK=0
NNODES=2
GPUS_PER_NODE=8
MASTER_ADDR="192.168.0.172"
MASTER_PORT=6006
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
                    --node_rank $NODE_RANK \
                    --master_addr $MASTER_ADDR \
                    --master_port $MASTER_PORT"



for i in 4 8 16
do
  torchrun $DISTRIBUTED_ARGS bert_dp.py  \
        --batch $i \
        --iter 80 \
        --config Bert_config/Bert_1.3B_2layers.json \
        --logdir log
done
