#!/bin/bash
> llama
export OMP_NUM_THREADS=1
NODE_RANK=0
NNODES=1
GPUS_PER_NODE=4
# MASTER_ADDR="192.168.0.172"
MASTER_ADDR=localhost
MASTER_PORT=6006
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
                    --node_rank $NODE_RANK \
                    --master_addr $MASTER_ADDR \
                    --master_port $MASTER_PORT"

config="Llama2_config/zero_config.json"
for i in 16 20 24 28 32 36 40
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun $DISTRIBUTED_ARGS llama_deepspeed.py  \
            --deepspeed --deepspeed_config ${config} \
            --batch $i \
            --iter 50 \
            --config Llama2_config/Llama2_7B_6layers.json \
            --logdir log >> llama
done


# > llama
# export OMP_NUM_THREADS=1
# NODE_RANK=0
# NNODES=1
# GPUS_PER_NODE=4
# # MASTER_ADDR="192.168.0.172"
# MASTER_ADDR=localhost
# MASTER_PORT=6006
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
#                     --node_rank $NODE_RANK \
#                     --master_addr $MASTER_ADDR \
#                     --master_port $MASTER_PORT"

# zero_config="Llama2_config/zero_config.json"
# for LAYERS in 2 3 4 5 6 7 8
# do
#   config="Llama2_config/Llama2_7B.json"
#   template_json="Llama2_config/Llama2_7B_xlayers.json"
#   sed "s/LAYERS/${LAYERS}/" ${template_json} > ${config}
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun $DISTRIBUTED_ARGS llama_deepspeed.py  \
#             --deepspeed --deepspeed_config ${zero_config} \
#             --batch 32 \
#             --iter 50 \
#             --config ${config} \
#             --logdir log >> llama
# done
