#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import inspect
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, Union

from table_bert.utils import BertTokenizer, BertConfig


BERT_CONFIGS = {
    'bert-base-uncased': BertConfig(
        vocab_size_or_config_json_file=30522,
        attention_probs_dropout_prob=0.1,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=2,
    )
    # Model config {
    #   "attention_probs_dropout_prob": 0.1,
    #   "hidden_act": "gelu",
    #   "hidden_dropout_prob": 0.1,
    #   "hidden_size": 768,
    #   "initializer_range": 0.02,
    #   "intermediate_size": 3072,
    #   "layer_norm_eps": 1e-12,
    #   "max_position_embeddings": 512,
    #   "num_attention_heads": 12,
    #   "num_hidden_layers": 12,
    #   "type_vocab_size": 2,
    #   "vocab_size": 30522
    # }
    ,
    'bert-large-uncased': BertConfig(
        vocab_size_or_config_json_file=30522,
        attention_probs_dropout_prob=0.1,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        num_attention_heads=16,
        num_hidden_layers=24,
        type_vocab_size=2,
    )
}


class TableBertConfig(SimpleNamespace):
    def __init__(
        self,
        base_model_name: str = 'bert-base-uncased',
        column_delimiter: str = '[SEP]',
        context_first: bool = True,
        cell_input_template: str = 'column | type | value',
        column_representation: str = 'mean_pool',
        max_cell_len: int = 5,
        max_sequence_len: int = 512,
        max_context_len: int = 256,
        masked_context_prob: float = 0.15,
        masked_column_prob: float = 0.2,
        max_predictions_per_seq: int = 100,
        context_sample_strategy: str = 'nearest',
        table_mask_strategy: str = 'column',
        do_lower_case: bool = True,
        **kwargs
    ):
        super(TableBertConfig, self).__init__()

        self.base_model_name = base_model_name
        self.column_delimiter = column_delimiter
        self.context_first = context_first
        self.column_representation = column_representation

        self.max_cell_len = max_cell_len
        self.max_sequence_len = max_sequence_len
        self.max_context_len = max_context_len

        self.do_lower_case = do_lower_case

        # tokenizer = BertTokenizer.from_pretrained(self.base_model_name)
        if isinstance(cell_input_template, str):
            if ' ' in cell_input_template:
                cell_input_template = cell_input_template.split(' ')
            else:
                print(f'WARNING: cell_input_template is outdated: {cell_input_template}', file=sys.stderr)
                cell_input_template = BertTokenizer.from_pretrained(self.base_model_name).tokenize(cell_input_template)

        self.cell_input_template = cell_input_template

        self.masked_context_prob = masked_context_prob
        self.masked_column_prob = masked_column_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.context_sample_strategy = context_sample_strategy
        self.table_mask_strategy = table_mask_strategy

        if not hasattr(self, 'vocab_size_or_config_json_file'):
            bert_config = BERT_CONFIGS[self.base_model_name]
            for k, v in vars(bert_config).items():
                setattr(self, k, v)

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument('--base_model_name', type=str, default='bert-base-uncased')

        parser.add_argument('--context_first', dest='context_first', action='store_true')
        parser.add_argument('--table_first', dest='context_first', action='store_false')
        parser.set_defaults(context_first=True)

        parser.add_argument("--column_delimiter", type=str, default='[SEP]', help='Column delimiter')
        parser.add_argument("--cell_input_template", type=str, default='column | type | value', help='Cell representation')
        parser.add_argument("--column_representation", type=str, default='mean_pool', help='Column representation')

        # training specifications
        parser.add_argument("--max_sequence_len", type=int, default=512)
        parser.add_argument("--max_context_len", type=int, default=256)
        parser.add_argument("--max_cell_len", type=int, default=5)

        parser.add_argument("--masked_context_prob", type=float, default=0.15,
                            help="Probability of masking each token for the LM task")
        parser.add_argument("--masked_column_prob", type=float, default=0.20,
                            help="Probability of masking each token for the LM task")
        parser.add_argument("--max_predictions_per_seq", type=int, default=200,
                            help="Maximum number of tokens to mask in each sequence")

        parser.add_argument('--context_sample_strategy', type=str, default='nearest',
                            choices=['nearest', 'concate_and_enumerate'])
        parser.add_argument('--table_mask_strategy', type=str, default='column',
                            choices=['column', 'column_token'])

        parser.add_argument("--do_lower_case", action="store_true")
        parser.set_defaults(do_lower_case=True)

        return parser

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **override_args):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        args = json.load(file_path.open())
        override_args = override_args or dict()
        args.update(override_args)
        default_config = cls()
        config_dict = {}
        for key, default_val in vars(default_config).items():
            val = args.get(key, default_val)
            config_dict[key] = val

        # backward compatibility
        if 'column_item_delimiter' in args:
            column_item_delimiter = args['column_item_delimiter']
            cell_input_template = 'column'
            use_value = args.get('use_sample_value', True)
            use_type = args.get('use_type_text', True)

            if use_type:
                cell_input_template += column_item_delimiter + 'type'
            if use_value:
                cell_input_template += column_item_delimiter + 'value'

            config_dict['cell_input_template'] = cell_input_template

        config = cls(**config_dict)

        return config

    @classmethod
    def from_dict(cls, args: Dict):
        return cls(**args)

    def with_new_args(self, **updated_args):
        new_config = self.__class__(**vars(self))
        for key, val in updated_args.items():
            setattr(new_config, key, val)

        return new_config

    def save(self, file_path: Path):
        json.dump(vars(self), file_path.open('w'), indent=2, sort_keys=True, default=str)

    def to_log_string(self):
        return json.dumps(vars(self), indent=2, sort_keys=True, default=str)

    def to_dict(self):
        return vars(self)

    def get_default_values_for_parameters(self):
        signature = inspect.signature(self.__init__)

        default_args = OrderedDict(
            (k, v.default)
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        )

        return default_args

    def extract_args(self, kwargs, pop=True):
        arg_dict = {}

        for key, default_val in self.get_default_values_for_parameters().items():
            if key in kwargs:
                val = kwargs.get(key)
                if pop:
                    kwargs.pop(key)

                arg_dict[key] = val

        return arg_dict

    @staticmethod
    def infer_model_class_from_config_dict(config_dict):
        if 'num_vertical_layers' in config_dict:
            from .vertical.vertical_attention_table_bert import VerticalAttentionTableBert
            return VerticalAttentionTableBert

        from .vanilla_table_bert import VanillaTableBert
        return VanillaTableBert

    @staticmethod
    def infer_model_class_from_config_file(config_file):
        config_dict = json.load(open(config_file))
        return TableBertConfig.infer_model_class_from_config_dict(config_dict)
