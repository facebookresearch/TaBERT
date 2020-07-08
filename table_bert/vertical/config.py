#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from argparse import ArgumentParser
from pathlib import Path

from typing import Union

from table_bert.utils import BertForMaskedLM
from table_bert.config import TableBertConfig, BERT_CONFIGS


class VerticalAttentionTableBertConfig(TableBertConfig):
    def __init__(
        self,
        num_vertical_attention_heads=6,
        num_vertical_layers=3,
        sample_row_num=3,
        table_mask_strategy='column',
        predict_cell_tokens=False,
        # vertical_layer_use_intermediate_transform=True,
        initialize_from=None,
        **kwargs,
    ):
        TableBertConfig.__init__(self, **kwargs)

        self.num_vertical_attention_heads = num_vertical_attention_heads
        self.num_vertical_layers = num_vertical_layers
        self.sample_row_num = sample_row_num
        self.table_mask_strategy = table_mask_strategy
        self.predict_cell_tokens = predict_cell_tokens
        # self.vertical_layer_use_intermediate_transform = vertical_layer_use_intermediate_transform
        self.initialize_from = initialize_from

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        TableBertConfig.add_args(parser)

        parser.add_argument("--num_vertical_attention_heads", type=int, default=6)
        parser.add_argument("--num_vertical_layers", type=int, default=3)
        parser.add_argument("--sample_row_num", type=int, default=3)
        parser.add_argument("--predict_cell_tokens", action='store_true', dest='predict_cell_tokens')
        parser.add_argument("--no_predict_cell_tokens", action='store_false', dest='predict_cell_tokens')
        parser.set_defaults(predict_cell_tokens=False)

        parser.add_argument("--initialize_from", type=Path, default=None)

    @staticmethod
    def is_valid_config_file(file_path: Union[str, Path]):
        return 'num_vertical_layers' in json.load(open(file_path))
