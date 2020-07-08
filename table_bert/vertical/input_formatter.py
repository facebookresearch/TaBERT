#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import numpy as np
from itertools import chain
from typing import List, Callable, Any, Dict

from table_bert.utils import BertTokenizer
from table_bert.config import TableBertConfig
from table_bert.dataset import Example
from table_bert.input_formatter import TableBertBertInputFormatter, VanillaTableBertInputFormatter
from table_bert.table import Column, Table
from table_bert.table_bert import MAX_BERT_INPUT_LENGTH


# noinspection PyMethodOverriding
from table_bert.vertical.config import VerticalAttentionTableBertConfig


class VerticalAttentionTableBertInputFormatter(VanillaTableBertInputFormatter):
    def __init__(self, config: VerticalAttentionTableBertConfig, tokenizer: BertTokenizer):
        super(VerticalAttentionTableBertInputFormatter, self).__init__(config, tokenizer)

        column_span_method = 'whole_span'
        if 'column_name' in self.config.column_representation:
            column_span_method = 'column_name'
        elif 'first_token' in self.config.column_representation:
            column_span_method = 'first_token'

        self.column_span_method = column_span_method

    def get_input(self, context: List[str], table: Table):
        row_instances = []

        table_data = table.data[:self.config.sample_row_num]

        for row_data in table_data:
            if isinstance(row_data, dict):
                row_data = [row_data[col.name] for col in table.header]
            row_instance = self.get_row_input(context, table.header, row_data)
            row_instances.append(row_instance)

        return {
            'rows': row_instances
        }

    def get_row_input(self, context: List[str], header: List[Column], row_data: List[Any]):
        row_instance = super(VerticalAttentionTableBertInputFormatter, self).get_row_input(
            context, header, row_data=row_data)

        input_sequence_len = len(row_instance['tokens'])
        column_token_position_to_column_ids = [np.iinfo(np.uint16).max] * input_sequence_len

        for col_id, column in enumerate(header):
            if col_id < len(row_instance['column_spans']):
                col_start, col_end = row_instance['column_spans'][col_id][self.column_span_method]

                column_token_position_to_column_ids[col_start: col_end] = [col_id] * (col_end - col_start)

        row_instance['column_token_position_to_column_ids'] = column_token_position_to_column_ids

        return row_instance

    def get_pretraining_instances_from_example(
        self, example: Example,
        context_sampler: Callable
    ):
        instances = []
        context_iter = context_sampler(
            example, self.config.max_context_len, context_sample_strategy=self.config.context_sample_strategy)

        for context in context_iter:
            row_num = len(example.column_data[0])
            row_ids = list(range(row_num))

            if self.config.sample_row_num < row_num:
                sampled_row_ids = sorted(random.sample(row_ids, k=self.config.sample_row_num))
            else:
                sampled_row_ids = row_ids

            rows = []
            for row_id in sampled_row_ids:
                row = [column[row_id] for column in example.column_data]

                # if all cells are empty
                if all(val is None or len(val) == 0 for val in row):
                    continue

                row = [[] if val is None else self.tokenizer.tokenize(val) for val in row]
                rows.append(row)

            if rows:
                table = Table(example.uuid, example.header, data=rows)

                instance = self.create_pretraining_instance(context, table, example)
                if instance is not None:
                    instances.append(instance)

        return instances

    def create_pretraining_instance(self, context: List[str], table: Table, example: Example):
        assert self.config.table_mask_strategy == 'column'

        row_instances = self.get_input(context, table)['rows']

        num_maskable_columns = min(len(row_inst['column_spans']) for row_inst in row_instances)
        num_column_to_mask = max(1, math.ceil(num_maskable_columns * self.config.masked_column_prob))
        columns_to_mask = sorted(random.sample(list(range(num_maskable_columns)), num_column_to_mask))

        masked_column_token_indices_list = []
        masked_column_token_column_ids = []

        masked_cell_token_indices_list = []
        masked_cell_token_column_ids_list = []
        masked_cell_token_labels_list = []
        for row_id, row_instance in enumerate(row_instances):
            maskable_column_token_indices = [
                (
                        list(range(*span['column_name'])) +
                        list(range(*span['type']))
                )
                for col_id, span
                in enumerate(row_instance['column_spans'])
            ]

            masked_column_token_indices = [
                token_idx
                for col_id in columns_to_mask
                for token_idx in maskable_column_token_indices[col_id]
            ]

            masked_cell_token_indices = [
                range(*row_instance['column_spans'][col_id]['value'])
                for col_id in columns_to_mask
            ]

            # masked_cell_token_column_ids = [
            #     col_id
            #     for col_id in columns_to_mask
            #     for token_idx in masked_cell_token_indices[col_id]
            # ]

            masked_cell_token_indices = list(chain(*masked_cell_token_indices))
            masked_cell_token_labels = [row_instance['tokens'][pos] for pos in masked_cell_token_indices]

            masked_cell_token_indices_list.append(masked_cell_token_indices)
            # masked_cell_token_column_ids_list.append(masked_cell_token_column_ids)
            masked_cell_token_labels_list.append(masked_cell_token_labels)

            if row_id == 0:
                masked_column_token_column_ids = [
                    col_id
                    for col_id in columns_to_mask
                    for token_idx in maskable_column_token_indices[col_id]
                ]

            masked_column_token_indices_list.append(masked_column_token_indices)

        num_masked_column_tokens = len(masked_column_token_indices_list[0])
        assert all(len(mask_list) == num_masked_column_tokens for mask_list in masked_column_token_indices_list)

        max_context_token_to_mask = self.config.max_predictions_per_seq - num_masked_column_tokens

        context_token_indices = (
            list(range(*row_instances[0]['context_span']))[1:]
            if self.config.context_first else
            list(range(*row_instances[0]['context_span']))[:-1]
        )

        num_context_tokens_to_mask = min(
            max_context_token_to_mask,
            max(
                1,
                int(len(context_token_indices) * self.config.masked_context_prob)
            )
        )
        if num_context_tokens_to_mask > 0:
            masked_context_token_indices = sorted(random.sample(context_token_indices, num_context_tokens_to_mask))
        else:
            masked_context_token_indices = []

        masked_token_indices_list = []
        for row_id, row_instance in enumerate(row_instances):
            masked_token_indices_list.append(
                masked_context_token_indices + masked_column_token_indices_list[row_id]
            )

        first_row_tokens = row_instances[0]['tokens']
        masked_context_token_labels = [first_row_tokens[idx] for idx in masked_context_token_indices]
        masked_column_token_labels = [first_row_tokens[idx] for idx in masked_column_token_indices_list[0]]
        masked_token_labels = [first_row_tokens[idx] for idx in masked_token_indices_list[0]]

        for token_relative_idx, token in enumerate(masked_token_labels):
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = token
                # 10% of the time, replace with random word
                else:
                    masked_token = random.choice(self.vocab_list)

            # Once we've saved the true label for that token, we can overwrite it with the masked version
            for row_id, row_instance in enumerate(row_instances):
                token_idx = masked_token_indices_list[row_id][token_relative_idx]
                row_instance['tokens'][token_idx] = masked_token

        if (
            self.config.predict_cell_tokens and
            all(
                len(masked_cell_token_indices_list[i]) == 0
                for i in range(len(row_instances))
            )
        ):
            return None

        info = {}
        pretrain_instance = {
            "rows": [
                {
                    'tokens': row_instance['tokens'],
                    'token_ids': self.tokenizer.convert_tokens_to_ids(row_instance['tokens']),
                    'segment_a_length': row_instance['segment_a_length'],
                    'context_span': row_instance['context_span'],
                    'column_token_position_to_column_ids': row_instance['column_token_position_to_column_ids'],
                    'masked_cell_token_positions': masked_cell_token_indices_list[row_id] if self.config.predict_cell_tokens else None,
                    'masked_cell_token_label_ids': self.tokenizer.convert_tokens_to_ids(masked_cell_token_labels_list[row_id]) if self.config.predict_cell_tokens else None
                }
                for row_id, row_instance
                in enumerate(row_instances)
            ],
            'table_size': (len(row_instances), num_maskable_columns),
            'masked_context_token_positions': masked_context_token_indices,
            'masked_context_token_labels': masked_context_token_labels,
            'masked_context_token_label_ids': self.tokenizer.convert_tokens_to_ids(masked_context_token_labels),
            'masked_column_token_column_ids': masked_column_token_column_ids,
            'masked_column_token_labels': masked_column_token_labels,
            'masked_column_token_label_ids': self.tokenizer.convert_tokens_to_ids(masked_column_token_labels),
            "info": info
        }

        assert all(x < num_maskable_columns for x in masked_column_token_column_ids)

        return pretrain_instance

    def remove_unecessary_instance_entries(self, instance: Dict):
        for row in instance['rows']:
            del row['tokens']
