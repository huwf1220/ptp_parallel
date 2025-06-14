#!/bin/bash
DEVICE_NUM=8
DATA_TYPE="float32"

TEST_CONFIG=(
    "F, 16, 4"
    "F, 24, 2"
    "F, 32, 2"
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
        --parallel_block_opt 1 

    echo "================================================================="
done
