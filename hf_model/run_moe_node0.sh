
# for batch in 8 16 32 128
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 \
#       python moe_layer_torch_dp.py \
#             --batch ${batch} \
#             --iter 50 \
#             --config Moe_config/moe_7.1B_2layers.json \
#             --logdir log 
# done

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



for i in 2 4 8 12 16
do
  torchrun $DISTRIBUTED_ARGS moe_dp.py  \
        --batch $i \
        --iter 80 \
        --config Moe_config/moe_10B_2layers.json \
        --logdir log
done

# for i in 2 4 8 16
# do
#   torchrun $DISTRIBUTED_ARGS moe_dp.py  \
#         --batch $i \
#         --iter 80 \
#         --config Moe_config/moe_7.1B_2layers.json \
#         --logdir log  >> moe
# done
# for i in 2 4 8
# do
#   torchrun $DISTRIBUTED_ARGS moe_dp.py  \
#         --batch $i \
#         --iter 80 \
#         --config Moe_config/moe_10B_2layers.json \
#         --logdir log  >> moe
# done