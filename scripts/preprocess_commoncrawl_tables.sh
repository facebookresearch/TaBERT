#!/bin/bash

CW_DATA_PATH=data/datasets/commoncrawl.sample.jsonl
OUTPUT_FILE=data/preprocessed_data/common_crawl.preprocessed.jsonl
FILTER="[0-1][0-9].tar.gz"

python \
    -m preprocess.common_crawl \
    --worker_num 12 \
    --input_file ${CW_DATA_PATH} \
    --filter ${FILTER} \
    --output_file ${OUTPUT_FILE}
