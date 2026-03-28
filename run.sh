#!/usr/bin/env sh

if [ -z "$1" ]; then
    echo "Usage: sh run.sh <config-file>"
    exit 1
fi

config_file="$1"
if [ ! -f "$config_file" ]; then
    echo "Config file not found: $config_file"
    exit 1
fi

# POSIX-compatible config loading (works with both sh and bash).
. "$config_file"

require_var() {
    eval "var_value=\${$1-}"
    if [ -z "$var_value" ]; then
        echo "Missing required config variable: $1"
        exit 1
    fi
}

require_var base_output_dir
require_var path_length
require_var hidden_size
require_var embedding_size
require_var batch_size
require_var beta
require_var Lambda
require_var use_entity_embeddings
require_var train_entity_embeddings
require_var train_relation_embeddings
require_var data_input_dir
require_var vocab_dir
require_var model_load_dir
require_var load_model
require_var total_iterations
require_var nell_evaluation

export PYTHONPATH="."
gpu_id="${gpu_id:-0}"

echo "Executing python code/model/trainer.py with config: $config_file"

CUDA_VISIBLE_DEVICES="$gpu_id" python code/model/trainer.py \
    --base_output_dir "$base_output_dir" \
    --path_length "$path_length" \
    --hidden_size "$hidden_size" \
    --embedding_size "$embedding_size" \
    --batch_size "$batch_size" \
    --beta "$beta" \
    --Lambda "$Lambda" \
    --use_entity_embeddings "$use_entity_embeddings" \
    --train_entity_embeddings "$train_entity_embeddings" \
    --train_relation_embeddings "$train_relation_embeddings" \
    --data_input_dir "$data_input_dir" \
    --vocab_dir "$vocab_dir" \
    --model_load_dir "$model_load_dir" \
    --load_model "$load_model" \
    --total_iterations "$total_iterations" \
    --nell_evaluation "$nell_evaluation"
