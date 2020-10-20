#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set +e

output_dir=data/train_data/vertical_tabert
mkdir -p ${output_dir}

python -m utils.generate_vertical_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus data/preprocessed_data/tables.jsonl \
    --base_model_name bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate 15 \
    --max_context_len 128 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'column|type|value' \
    --column_delimiter "[SEP]" \
    --sample_row_num 3 \
    --predict_cell_tokens
