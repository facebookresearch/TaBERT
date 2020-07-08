#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import sys
from typing import List, Any, Tuple, Dict
import numpy as np
from fairseq import distributed_utils
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_max, scatter_mean

from table_bert.utils import BertForPreTraining, BertForMaskedLM, TRANSFORMER_VERSION, TransformerVersion
from table_bert.table_bert import TableBertModel
from table_bert.config import TableBertConfig, BERT_CONFIGS
from table_bert.table import Table
from table_bert.input_formatter import VanillaTableBertInputFormatter


class VanillaTableBert(TableBertModel):
    CONFIG_CLASS = TableBertConfig

    def __init__(
        self,
        config: TableBertConfig,
        **kwargs
    ):
        super(VanillaTableBert, self).__init__(config, **kwargs)

        self._bert_model = BertForMaskedLM.from_pretrained(config.base_model_name)
        self.input_formatter = VanillaTableBertInputFormatter(self.config, self.tokenizer)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        sequence_output, _ = self._bert_model.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self._bert_model.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.bert_config.vocab_size), masked_lm_labels.view(-1))

            sample_size = masked_lm_labels.ne(-1).sum().item()
            logging_output = {
                'sample_size': sample_size,
                'loss': masked_lm_loss.item()
            }

            return masked_lm_loss, logging_output
        else:
            return prediction_scores

    def encode_context_and_table(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_token_indices: torch.Tensor,
        context_token_mask: torch.Tensor,
        column_token_mask: torch.Tensor,
        column_token_to_column_id: torch.Tensor,
        column_mask: torch.Tensor,
        return_bert_encoding: bool = False,
        **kwargs
    ):

        # print('input_ids', input_ids.size(), file=sys.stderr)
        # print('segment_ids', segment_ids.size(), file=sys.stderr)
        # print('attention_mask', attention_mask.size(), file=sys.stderr)
        # print('column_token_mask', column_token_mask.size(), file=sys.stderr)
        # print('column_token_mask', column_token_mask.sum(dim=-1), file=sys.stderr)
        # print('column_token_to_column_id', column_token_to_column_id.size(), file=sys.stderr)
        # print('column_token_to_column_id', column_token_to_column_id.sum(dim=-1), file=sys.stderr)
        # print('column_mask', column_mask.size(), file=sys.stderr)

        kwargs = (
            {}
            if TRANSFORMER_VERSION == TransformerVersion.TRANSFORMERS
            else {'output_all_encoded_layers': False}
        )
        sequence_output, _ = self.bert(
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask,
            **kwargs
        )
        # except:
        #     print('!!!!!Exception!!!!!')
        #     datum = (input_ids, segment_ids, attention_mask, question_token_mask,
        #              column_token_mask, column_token_to_column_id, column_mask)
        #     torch.save(datum, 'debug.tensors.bin')
        #     raise

        # gather column representations
        # (batch_size, max_seq_len, encoding_size)
        flattened_column_encoding = sequence_output
        # (batch_size, max_column_size, encoding_size)
        column_encoding = self.get_column_representation(
            flattened_column_encoding,
            column_token_to_column_id,
            column_token_mask,
            column_mask,
            aggregator=self.config.column_representation
        )

        # (batch_size, context_len, encoding_size)
        context_encoding = torch.gather(
            sequence_output,
            dim=1,
            index=context_token_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1)),
        )
        context_encoding = context_encoding * context_token_mask.unsqueeze(-1)

        encoding_info = {}
        if return_bert_encoding:
            encoding_info['bert_encoding'] = sequence_output

        return context_encoding, column_encoding, encoding_info

    @staticmethod
    def get_column_representation(
        flattened_column_encoding: torch.Tensor,
        column_token_to_column_id: torch.Tensor,
        column_token_mask: torch.Tensor,
        column_mask: torch.Tensor,
        aggregator: str = 'mean_pool'
    ) -> torch.Tensor:
        """
        Args:
            flattened_column_encoding: (batch_size, total_column_token_num, encoding_size)
            column_token_to_column_id: (batch_size, total_column_token_num + 1)
            column_mask: (batch_size, max_column_num)
            aggregator: ['mean_pool', 'max_pool', 'first_token']
        Returns:
            column_encoding: (batch_size, max_column_num, encoding_size)
        """

        if aggregator.startswith('max_pool'):
            agg_func = scatter_max
            flattened_column_encoding[column_token_mask == 0] = float('-inf')
        elif aggregator.startswith('mean_pool') or aggregator.startswith('first_token'):
            agg_func = scatter_mean
        else:
            raise ValueError(f'Unknown column representation method {aggregator}')

        max_column_num = column_mask.size(-1)
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size + 1, encoding_size)
        result = agg_func(flattened_column_encoding,
                          column_token_to_column_id.unsqueeze(-1).expand(-1, -1, flattened_column_encoding.size(-1)),
                          dim=1,
                          dim_size=max_column_num + 1)

        # remove the last "garbage collection" entry, mask out padding columns
        result = result[:, :-1] * column_mask.unsqueeze(-1)

        if aggregator == 'max_pool':
            column_encoding = result[0]
        else:
            column_encoding = result

        return column_encoding

    def to_tensor_dict(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        table_specific_tensors=True
    ):
        instances = []
        for e_id, (context, table) in enumerate(zip(contexts, tables)):
            instance = self.input_formatter.get_input(context, table)
            instances.append(instance)

        batch_size = len(contexts)
        max_sequence_len = max(len(x['tokens']) for x in instances)

        # basic tensors
        input_array = np.zeros((batch_size, max_sequence_len), dtype=np.int)
        mask_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)
        segment_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

        # table specific tensors
        if table_specific_tensors:
            max_column_num = max(len(x['column_spans']) for x in instances)
            max_context_len = max(x['context_length'] for x in instances)

            context_token_indices = np.zeros((batch_size, max_context_len), dtype=np.int)
            context_mask = np.zeros((batch_size, max_context_len), dtype=np.bool)
            column_token_mask = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

            # we initialize the mapping with the id of last column as the "garbage collection" entry for reduce ops
            column_token_to_column_id = np.zeros((batch_size, max_sequence_len), dtype=np.int)
            column_token_to_column_id.fill(max_column_num)

            column_mask = np.zeros((batch_size, max_column_num), dtype=np.bool)

            column_span = 'whole_span'
            if 'column_name' in self.config.column_representation:
                column_span = 'column_name'
            elif 'first_token' in self.config.column_representation:
                column_span = 'first_token'

        for i, instance in enumerate(instances):
            token_ids = self.tokenizer.convert_tokens_to_ids(instance['tokens'])

            input_array[i, :len(token_ids)] = token_ids
            segment_array[i, instance['segment_a_length']: len(token_ids)] = 1
            mask_array[i, :len(token_ids)] = 1.

            if table_specific_tensors:
                context_token_indices[i, :instance['context_length']] = list(range(*instance['context_span'])) #instance['context_token_indices']
                context_mask[i, :instance['context_length']] = 1.

                header = tables[i].header
                for col_id, column in enumerate(header):
                    if col_id < len(instance['column_spans']):
                        col_start, col_end = instance['column_spans'][col_id][column_span]

                        column_token_to_column_id[i, col_start: col_end] = col_id
                        column_token_mask[i, col_start: col_end] = 1.
                        column_mask[i, col_id] = 1.

        tensor_dict = {
            'input_ids': torch.tensor(input_array.astype(np.int64)),
            'segment_ids': torch.tensor(segment_array.astype(np.int64)),
            'attention_mask': torch.tensor(mask_array, dtype=torch.float32),
        }

        if table_specific_tensors:
            tensor_dict.update({
                'context_token_indices': torch.tensor(context_token_indices.astype(np.int64)),
                'context_token_mask': torch.tensor(context_mask, dtype=torch.float32),
                'column_token_to_column_id': torch.tensor(column_token_to_column_id.astype(np.int64)),
                'column_token_mask': torch.tensor(column_token_mask, dtype=torch.float32),
                'column_mask': torch.tensor(column_mask, dtype=torch.float32)
            })

        # for instance in instances:
        #     print(instance)

        return tensor_dict, instances

    def encode(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        return_bert_encoding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        tensor_dict, instances = self.to_tensor_dict(contexts, tables)
        device = next(self.parameters()).device

        for key in tensor_dict.keys():
            tensor_dict[key] = tensor_dict[key].to(device)

        context_encoding, column_encoding, encoding_info = self.encode_context_and_table(
            **tensor_dict,
            return_bert_encoding=return_bert_encoding
        )

        info = {
            'tensor_dict': tensor_dict,
            'instances': instances,
            **encoding_info
        }

        return context_encoding, column_encoding, info

    def validate(self, data_loader, args):
        was_training = self.training
        self.eval()

        keys = ['loss', 'sample_size']

        logging_info_list = []
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc=f"Evaluation", file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    loss_sum, logging_info = self(**batch)
                    logging_info = {k: logging_info[k] for k in keys}
                    logging_info_list.append(logging_info)

                    pbar.update(1)

        if was_training:
            self.train()

        stats = {
            k: sum(x[k] for x in logging_info_list)
            for k in keys
        }

        # handel distributed evaluation
        if args.multi_gpu:
            stats = distributed_utils.all_gather_list(stats)
            stats = {
                k: sum(x[k] for x in stats)
                for k in keys
            }

        valid_result = {
            'ppl': math.exp(stats['loss'] / stats['sample_size'])
        }

        return valid_result

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if not any(key.startswith('_bert_model') for key in state_dict):
            logging.warning('warning: loading model from an old version')
            self._bert_model.load_state_dict(state_dict, strict)
        else:
            super(VanillaTableBert, self).load_state_dict(state_dict, strict)
