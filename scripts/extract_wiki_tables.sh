#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# path to the jdk
JAVA_PATH=/System/Library/Frameworks/JavaVM.framework/Versions/Current/Commands/
CLASSPATH=contrib/wiki_extractor/tableBERT-1.0-SNAPSHOT-jar-with-dependencies.jar

# The following dump is a sample downloaded from
# https://dumps.wikimedia.org/enwiki/20200901/enwiki-20200901-pages-articles-multistream1.xml-p1p30303.bz2
# You may need to use the full Wikipedia dump for data extraction, for example,
# https://dumps.wikimedia.org/enwiki/20201001/enwiki-20201001-pages-articles-multistream.xml.bz2 .
# In our paper we used the dump `enwiki-20190520-pages-articles-multistream.xml.bz2`.
wget -nc https://dumps.wikimedia.org/enwiki/20200901/enwiki-20200901-pages-articles-multistream1.xml-p1p30303.bz2 -P data/datasets/
WIKI_DUMP=data/datasets/enwiki-20200901-pages-articles-multistream1.xml-p1p30303.bz2

OUTPUT_FILE=data/preprocessed_data/wiki_tables.jsonl

CLASSPATH=${CLASSPATH} JAVA_PATH=${JAVA_PATH} python \
    -m preprocess.extract_wiki_data \
    --wiki_dump ${WIKI_DUMP} \
    --output_file ${OUTPUT_FILE}
