#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DEVICE_NUM=4
DATA_TYPE="float16"

TEST_CONFIG=(
    "F, 4, 4"
    "F, 8, 4"
    "F, 16, 2"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPT_DIR="$(cd "$SCRIPT_DIR/../gpt" && pwd)" 

for CONFIG_LINE in "${TEST_CONFIG[@]}"; do
    IFS=',' read -r CONFIG BATCH_SIZE NUM_LAYER <<< "$CONFIG_LINE"
    $GPT_DIR/gpt_test.sh \
        --config "$CONFIG" \
        --batch_size "$BATCH_SIZE" \
        --num_layer "$NUM_LAYER" \
        --device_num "$DEVICE_NUM" \
        --data_type "$DATA_TYPE" \

    echo "================================================================="
done
unset CUDA_VISIBLE_DEVICES