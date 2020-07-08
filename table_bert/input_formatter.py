#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
from random import choice, shuffle, sample, random
from typing import List, Callable, Dict, Any

from table_bert.utils import BertTokenizer
from table_bert.table_bert import MAX_BERT_INPUT_LENGTH
from table_bert.config import TableBertConfig
from table_bert.dataset import Example
from table_bert.table import Column, Table


class TableBertBertInputFormatter(object):
    def __init__(self, config: TableBertConfig, tokenizer: BertTokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.vocab_list = list(self.tokenizer.vocab.keys())


class TableTooLongError(ValueError):
    pass


class VanillaTableBertInputFormatter(TableBertBertInputFormatter):
    def get_cell_input(
        self,
        column: Column,
        cell_value: List[str],
        token_offset: int
    ):
        input = []
        span_map = {
            'first_token': (token_offset, token_offset + 1)
        }

        for token in self.config.cell_input_template:
            start_token_abs_position = len(input) + token_offset
            if token == 'column':
                span_map['column_name'] = (start_token_abs_position,
                                           start_token_abs_position + len(column.name_tokens))
                input.extend(column.name_tokens)
            elif token == 'value':
                span_map['value'] = (start_token_abs_position,
                                     start_token_abs_position + len(cell_value))
                input.extend(cell_value)
            elif token == 'type':
                span_map['type'] = (start_token_abs_position,
                                    start_token_abs_position + 1)
                input.append(column.type)
            else:
                span_map.setdefault('other_tokens', []).append(start_token_abs_position)
                input.append(token)

        span_map['whole_span'] = (token_offset, token_offset + len(input))

        return input, span_map

    def get_input(self, context: List[str], table: Table, trim_long_table=False):
        row_data = [
            column.sample_value_tokens
            for column in table.header
        ]

        return self.get_row_input(context, table.header, row_data, trim_long_table=trim_long_table)

    def get_row_input(self, context: List[str], header: List[Column], row_data: List[Any], trim_long_table=False):
        if self.config.context_first:
            table_tokens_start_idx = len(context) + 2  # account for [CLS] and [SEP]
            # account for [CLS] and [SEP], and the ending [SEP]
            max_table_token_length = MAX_BERT_INPUT_LENGTH - len(context) - 2 - 1
        else:
            table_tokens_start_idx = 1  # account for starting [CLS]
            # account for [CLS] and [SEP], and the ending [SEP]
            max_table_token_length = MAX_BERT_INPUT_LENGTH - len(context) - 2 - 1

        # generate table tokens
        row_input_tokens = []
        column_token_span_maps = []
        column_start_idx = table_tokens_start_idx

        for col_id, column in enumerate(header):
            value_tokens = row_data[col_id]
            truncated_value_tokens = value_tokens[:self.config.max_cell_len]

            column_input_tokens, token_span_map = self.get_cell_input(
                column,
                truncated_value_tokens,
                token_offset=column_start_idx
            )
            column_input_tokens.append(self.config.column_delimiter)

            early_stop = False
            if trim_long_table:
                if len(row_input_tokens) + len(column_input_tokens) > max_table_token_length:
                    valid_column_input_token_len = max_table_token_length - len(row_input_tokens)
                    column_input_tokens = column_input_tokens[:valid_column_input_token_len]
                    end_index = column_start_idx + len(column_input_tokens)
                    keys_to_delete = []
                    for key in token_span_map:
                        if key in {'column_name', 'type', 'value', 'whole_span'}:
                            span_start_idx, span_end_idx = token_span_map[key]
                            if span_start_idx < end_index < span_end_idx:
                                token_span_map[key] = (span_start_idx, end_index)
                            elif end_index < span_start_idx:
                                keys_to_delete.append(key)
                        elif key == 'other_tokens':
                            old_positions = token_span_map[key]
                            new_positions = [idx for idx in old_positions if idx < end_index]
                            if not new_positions:
                                keys_to_delete.append(key)

                    for key in keys_to_delete:
                        del token_span_map[key]

                    # nothing left, we just skip this cell and break
                    if len(token_span_map) == 0:
                        break

                    early_stop = True
                elif len(row_input_tokens) + len(column_input_tokens) == max_table_token_length:
                    early_stop = True
            elif len(row_input_tokens) + len(column_input_tokens) > max_table_token_length:
                break

            row_input_tokens.extend(column_input_tokens)
            column_start_idx = column_start_idx + len(column_input_tokens)
            column_token_span_maps.append(token_span_map)

            if early_stop: break

        # it is possible that the first cell to too long and cannot fit into `max_table_token_length`
        # we need to discard this sample
        if len(row_input_tokens) == 0:
            raise TableTooLongError()

        if row_input_tokens[-1] == self.config.column_delimiter:
            del row_input_tokens[-1]

        if self.config.context_first:
            sequence = ['[CLS]'] + context + ['[SEP]'] + row_input_tokens + ['[SEP]']
            # segment_ids = [0] * (len(context) + 2) + [1] * (len(row_input_tokens) + 1)
            segment_a_length = len(context) + 2
            context_span = (0, 1 + len(context))
            # context_token_indices = list(range(0, 1 + len(context)))
        else:
            sequence = ['[CLS]'] + row_input_tokens + ['[SEP]'] + context + ['[SEP]']
            # segment_ids = [0] * (len(row_input_tokens) + 2) + [1] * (len(context) + 1)
            segment_a_length = len(row_input_tokens) + 2
            context_span = (len(row_input_tokens) + 1, len(row_input_tokens) + 1 + 1 + len(context) + 1)
            # context_token_indices = list(range(len(row_input_tokens) + 1, len(row_input_tokens) + 1 + 1 + len(context) + 1))

        instance = {
            'tokens': sequence,
            #'token_ids': self.tokenizer.convert_tokens_to_ids(sequence),
            'segment_a_length': segment_a_length,
            # 'segment_ids': segment_ids,
            'column_spans': column_token_span_maps,
            'context_length': 1 + len(context),  # beginning [CLS]/[SEP] + input question
            'context_span': context_span,
            # 'context_token_indices': context_token_indices
        }

        return instance

    def get_pretraining_instances_from_example(
        self, example: Example,
        context_sampler: Callable
    ):
        instances = []
        context_iter = context_sampler(
            example, self.config.max_context_len, context_sample_strategy=self.config.context_sample_strategy)

        for context in context_iter:
            # row_num = len(example.column_data)
            # sampled_row_id = choice(list(range(row_num)))

            for col_idx, column in enumerate(example.header):
                col_values = example.column_data[col_idx]
                col_values = [val for val in col_values if val is not None and len(val) > 0]

                sampled_value = choice(col_values)

                # print('chosen value', sampled_value)
                sampled_value_tokens = self.tokenizer.tokenize(sampled_value)
                column.sample_value_tokens = sampled_value_tokens

            instance = self.create_pretraining_instance(context, example.header)
            instance['source'] = example.source

            instances.append(instance)

        return instances

    def create_pretraining_instance(self, context, header):
        table = Table('fake_table', header)
        input_instance = self.get_input(context, table, trim_long_table=True)
        column_spans = input_instance['column_spans']

        column_candidate_indices = [
            (
                list(range(*span['column_name']) if 'column_name' in span else []) +
                list(range(*span['type']) if 'type' in span else []) +
                (
                    span['other_tokens']
                    if random() < 0.01 and 'other_tokens' in span
                    else []
                )
            )
            for col_id, span
            in enumerate(column_spans)
        ]

        context_candidate_indices = (
            list(range(*input_instance['context_span']))[1:]
            if self.config.context_first else
            list(range(*input_instance['context_span']))[:-1]
        )

        masked_sequence, masked_lm_positions, masked_lm_labels, info = self.create_masked_lm_predictions(
            input_instance['tokens'], context_candidate_indices, column_candidate_indices
        )

        info['num_columns'] = len(header)

        instance = {
            "tokens": masked_sequence,
            "token_ids": self.tokenizer.convert_tokens_to_ids(masked_sequence),
            "segment_a_length": input_instance['segment_a_length'],
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "masked_lm_label_ids": self.tokenizer.convert_tokens_to_ids(masked_lm_labels),
            "info": info
        }

        return instance

    def create_masked_lm_predictions(
        self,
        tokens, context_indices, column_indices
    ):
        table_mask_strategy = self.config.table_mask_strategy

        info = dict()
        info['num_maskable_column_tokens'] = sum(len(token_ids) for token_ids in column_indices)

        if table_mask_strategy == 'column_token':
            column_indices = [i for l in column_indices for i in l]
            num_column_tokens_to_mask = min(self.config.max_predictions_per_seq,
                                            max(2, int(len(column_indices) * self.config.masked_column_prob)))
            shuffle(column_indices)
            masked_column_token_indices = sorted(sample(column_indices, num_column_tokens_to_mask))
        elif table_mask_strategy == 'column':
            num_maskable_columns = len(column_indices)
            num_column_to_mask = max(1, ceil(num_maskable_columns * self.config.masked_column_prob))
            columns_to_mask = sorted(sample(list(range(num_maskable_columns)), num_column_to_mask))
            shuffle(columns_to_mask)
            num_column_tokens_to_mask = sum(len(column_indices[i]) for i in columns_to_mask)
            masked_column_token_indices = [idx for col in columns_to_mask for idx in column_indices[col]]

            info['num_masked_columns'] = num_column_to_mask
        else:
            raise RuntimeError('unknown mode!')

        max_context_token_to_mask = self.config.max_predictions_per_seq - num_column_tokens_to_mask
        num_context_tokens_to_mask = min(max_context_token_to_mask,
                                         max(1, int(len(context_indices) * self.config.masked_context_prob)))

        if num_context_tokens_to_mask > 0:
            # if num_context_tokens_to_mask < 0 or num_context_tokens_to_mask > len(context_indices):
            #     for col_id in columns_to_mask:
            #         print([tokens[i] for i in column_indices[col_id]])
            #     print(num_context_tokens_to_mask, num_column_tokens_to_mask)
            shuffle(context_indices)
            masked_context_token_indices = sorted(sample(context_indices, num_context_tokens_to_mask))
            masked_indices = sorted(masked_context_token_indices + masked_column_token_indices)
        else:
            masked_indices = masked_column_token_indices

        masked_token_labels = []

        for index in masked_indices:
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(self.vocab_list)
            masked_token_labels.append(tokens[index])
            # Once we've saved the true label for that token, we can overwrite it with the masked version
            tokens[index] = masked_token

        info.update({
            'num_column_tokens_to_mask': num_column_tokens_to_mask,
            'num_context_tokens_to_mask': num_context_tokens_to_mask,
        })

        return tokens, masked_indices, masked_token_labels, info

    def remove_unecessary_instance_entries(self, instance: Dict):
        del instance['tokens']
        del instance['masked_lm_labels']
        del instance['info']


if __name__ == '__main__':
    config = TableBertConfig()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_formatter = VanillaTableBertInputFormatter(config, tokenizer)

    header = []
    for i in range(1000):
        header.append(
            Column(
                name='test',
                type='text',
                name_tokens=['test'] * 3,
                sample_value='ha ha ha yay',
                sample_value_tokens=['ha', 'ha', 'ha', 'yay']
            )
        )

    print(
        input_formatter.get_row_input(
            context='12 213 5 345 23 234'.split(),
            header=header,
            row_data=[col.sample_value_tokens for col in header],
            trim_long_table=True
        )
    )
