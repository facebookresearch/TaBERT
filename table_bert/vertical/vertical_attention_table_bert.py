#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import gc
import math
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torch_scatter import scatter_mean
from fairseq import distributed_utils

from table_bert.table import Column
from table_bert.utils import (
    BertConfig, BertForPreTraining, BertForMaskedLM,
    BertSelfOutput, BertIntermediate, BertOutput,
    BertLMPredictionHead, BertLayerNorm,
    gelu,
    TransformerVersion, TRANSFORMER_VERSION
)
from table_bert.vanilla_table_bert import VanillaTableBert, VanillaTableBertInputFormatter, TableBertConfig
from table_bert.table import *
from table_bert.vertical.config import VerticalAttentionTableBertConfig
from table_bert.vertical.input_formatter import VerticalAttentionTableBertInputFormatter
from table_bert.vertical.dataset import collate


class VerticalEmbeddingLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states


class BertVerticalAttention(nn.Module):
    def __init__(self, config: TableBertConfig):
        nn.Module.__init__(self)

        self.self_attention = VerticalSelfAttention(config)
        self.self_output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        self_attention_output = self.self_attention(hidden_states, attention_mask)
        output = self.self_output(self_attention_output, hidden_states)

        return output


class VerticalSelfAttention(nn.Module):
    def __init__(self, config: TableBertConfig):
        super(VerticalSelfAttention, self).__init__()

        if config.hidden_size % config.num_vertical_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_vertical_attention_heads))

        self.num_attention_heads = config.num_vertical_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_vertical_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_linear = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # (batch_size, max_row_num, max_sequence_len, num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        # (batch_size, max_sequence_len, num_attention_heads, max_row_num, attention_head_size)
        x = x.permute(0, 2, 3, 1, 4)

        return x

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        mixed_query_layer = self.query_linear(hidden_states)
        mixed_key_layer = self.key_linear(hidden_states)
        mixed_value_layer = self.value_linear(hidden_states)

        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, attention_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, max_row_num)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # TODO: consider remove this cell dropout?

        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, attention_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch_size, max_row_num, max_sequence_len, num_attention_heads, attention_head_size)
        context_layer = context_layer.permute(0, 3, 1, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertVerticalLayer(nn.Module):
    def __init__(self, config: VerticalAttentionTableBertConfig):
        nn.Module.__init__(self)

        self.attention = BertVerticalAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SpanBasedPrediction(nn.Module):
    def __init__(self, config: TableBertConfig, prediction_layer: BertLMPredictionHead):
        super(SpanBasedPrediction, self).__init__()
        
        self.dense1 = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layer_norm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer_norm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.prediction = prediction_layer

    def forward(self, span_representation, position_embedding) -> torch.Tensor:
        h = self.layer_norm1(
            gelu(
                self.dense1(
                    torch.cat(
                        [span_representation, position_embedding],
                        dim=-1
                    )
                )
            )
        )

        token_representation = self.layer_norm2(
            gelu(
                self.dense2(h)
            )
        )

        scores = self.prediction(token_representation)

        return scores


class VerticalAttentionTableBert(VanillaTableBert):
    CONFIG_CLASS = VerticalAttentionTableBertConfig

    def __init__(
        self,
        config: VerticalAttentionTableBertConfig,
        **kwargs
    ):
        super(VanillaTableBert, self).__init__(config, **kwargs)

        self._bert_model = BertForMaskedLM.from_pretrained(config.base_model_name)

        self.input_formatter = VerticalAttentionTableBertInputFormatter(self.config, self.tokenizer)

        if config.predict_cell_tokens:
            self.span_based_prediction = SpanBasedPrediction(config, self._bert_model.cls.predictions)

        self.vertical_embedding_layer = VerticalEmbeddingLayer()
        self.vertical_transformer_layers = nn.ModuleList([
            BertVerticalLayer(self.config)
            for _ in range(self.config.num_vertical_layers)
        ])

        if config.initialize_from:
            print(f'Loading initial parameters from {config.initialize_from}', file=sys.stderr)
            initial_state_dict = torch.load(config.initialize_from, map_location='cpu')
            if not any(key.startswith('_bert_model') for key in initial_state_dict):
                print('warning: loading model from an old version', file=sys.stderr)
                bert_model = BertForMaskedLM.from_pretrained(
                    config.base_model_name,
                    state_dict=initial_state_dict
                )
                self._bert_model = bert_model
            else:
                load_result = self.load_state_dict(initial_state_dict, strict=False)
                if load_result.missing_keys:
                    print(f'warning: missing keys: {load_result.missing_keys}', file=sys.stderr)
                if load_result.unexpected_keys:
                    print(f'warning: unexpected keys: {load_result.unexpected_keys}', file=sys.stderr)

        added_modules = [self.vertical_embedding_layer, self.vertical_transformer_layers]
        if config.predict_cell_tokens:
            added_modules.extend([
                self.span_based_prediction.dense1, self.span_based_prediction.dense2,
                self.span_based_prediction.layer_norm1, self.span_based_prediction.layer_norm2
            ])

        for module in added_modules:
            if TRANSFORMER_VERSION == TransformerVersion.TRANSFORMERS:
                module.apply(self._bert_model._init_weights)
            else:
                module.apply(self._bert_model.init_bert_weights)

    @property
    def parameter_type(self):
        return next(self.parameters()).dtype

    # noinspection PyMethodOverriding
    def forward(
        self,
        input_ids: torch.Tensor, segment_ids: torch.Tensor,
        context_token_positions: torch.Tensor, column_token_position_to_column_ids: torch.Tensor,
        sequence_mask: torch.Tensor, context_token_mask: torch.Tensor, table_mask: torch.Tensor,
        # masked_lm_labels: torch.Tensor = None
        masked_context_token_labels: torch.Tensor = None,
        masked_column_token_column_ids: torch.Tensor = None,
        masked_column_token_positions: torch.Tensor = None,
        masked_column_token_labels: torch.Tensor = None,
        masked_cell_token_positions: torch.Tensor = None,
        masked_cell_token_column_ids: torch.Tensor = None,
        masked_cell_token_labels: torch.Tensor = None,
        **kwargs
    ):
        """

        Args:
            input_ids: (batch_size, max_row_num, sequence_len)
            segment_ids: (batch_size, max_row_num, sequence_len)
            context_token_positions: (batch_size, max_row_num, max_context_len)
            column_token_position_to_column_ids: (batch_size, max_row_num, sequence_len)
            sequence_mask: (batch_size, max_row_num, sequence_len)
            context_token_mask: (batch_size, max_context_len)
            table_mask: (batch_size, max_row_num, max_column_num)
            masked_context_token_labels: (batch_size, max_context_len)
            masked_column_token_column_ids: (batch_size, max_masked_column_token_num)
            # masked_column_token_positions: (batch_size, max_masked_column_token_num)
            masked_column_token_labels: (batch_size, max_masked_column_token_num)
        """

        batch_size, max_row_num, sequence_len = input_ids.size()

        if self.parameter_type == torch.float16:
            sequence_mask = sequence_mask.to(dtype=torch.float16)
            context_token_mask = context_token_mask.to(dtype=torch.float16)
            table_mask = table_mask.to(dtype=torch.float16)

        flattened_input_ids = input_ids.view(batch_size * max_row_num, -1)
        flattened_segment_ids = segment_ids.view(batch_size * max_row_num, -1)
        flattened_sequence_mask = sequence_mask.view(batch_size * max_row_num, -1)

        # (batch_size * max_row_num, sequence_len, hidden_size)
        # (sequence_output, pooler_output)
        if TRANSFORMER_VERSION == TransformerVersion.PYTORCH_PRETRAINED_BERT:
            kwargs = {'output_all_encoded_layers': False}
        else:
            kwargs = {}

        bert_output, _ = self.bert(
            input_ids=flattened_input_ids,
            token_type_ids=flattened_segment_ids,
            attention_mask=flattened_sequence_mask,
            **kwargs
        )

        # torch.save(
        #     {
        #         'input_ids': flattened_input_ids,
        #         'token_type_ids': flattened_segment_ids,
        #         'attention_mask': flattened_sequence_mask
        #     },
        #     f'data/test_data/bert_output.{hf_flag}_hf.bin'
        # )

        # (batch_size, max_row_num, sequence_len, hidden_size)
        bert_output = bert_output.view(batch_size, max_row_num, sequence_len, -1)

        # expand to the same size as `bert_output`
        column_token_to_column_id_expanded = column_token_position_to_column_ids.unsqueeze(-1).expand(
            -1, -1, -1, bert_output.size(-1)  # (batch_size, max_row_num, sequence_len, hidden_size)
        )

        # (batch_size, max_row_num, max_column_num, hidden_size)
        max_column_num = table_mask.size(-1)
        table_encoding = scatter_mean(
            src=bert_output,
            index=column_token_to_column_id_expanded,
            dim=-2,  # over `sequence_len`
            dim_size=max_column_num + 1   # last dimension is the used for collecting unused entries
        )
        table_encoding = table_encoding[:, :, :-1, :] * table_mask.unsqueeze(-1)

        context_encoding = torch.gather(
            bert_output,
            dim=-2,
            index=context_token_positions.unsqueeze(-1).expand(-1, -1, -1, bert_output.size(-1)),
        )

        # expand to (batch_size, max_row_num, max_context_len)
        # context_token_mask = context_token_mask.unsqueeze(1).expand(-1, max_row_num, -1)

        context_encoding = context_encoding * context_token_mask.unsqueeze(-1)

        # perform vertical attention
        context_encoding, schema_encoding, final_table_encoding = self.vertical_transform(
            context_encoding, context_token_mask, table_encoding, table_mask)

        if masked_column_token_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

            # context MLM loss
            context_token_scores = self._bert_model.cls.predictions(context_encoding)

            # table cell span prediction loss
            if self.config.predict_cell_tokens:
                # (batch_size, max_row_num, max_masked_cell_token_num)
                masked_cell_token_position_embedding = self.bert.embeddings.position_embeddings(masked_cell_token_positions)
                # (batch_size, max_row_num, max_masked_cell_token_num)
                masked_cell_representation = torch.gather(
                    final_table_encoding,
                    dim=2,
                    index=masked_cell_token_column_ids.unsqueeze(-1).expand(-1, -1, -1, bert_output.size(-1))
                )
                # (batch_size, max_row_num, max_masked_cell_token_num, vocab_size)
                cell_token_scores = self.span_based_prediction(masked_cell_representation, masked_cell_token_position_embedding)
                # scalar
                masked_cell_token_loss = loss_fct(cell_token_scores.view(-1, self.config.vocab_size), masked_cell_token_labels.view(-1))
                masked_cell_token_num = masked_cell_token_labels.ne(-1).sum().item()

            # table schema MLM loss
            # (batch_size, masked_column_token_num, hidden_size)
            column_token_span_representation = torch.gather(
                schema_encoding,
                dim=1,
                index=masked_column_token_column_ids.unsqueeze(-1).expand(-1, -1, bert_output.size(-1))
            )
            # column_token_position_embedding = self.bert.embedding.position_embeddings(masked_column_token_positions)
            # column_token_scores = self.column_token_prediction(column_token_span_representation, column_token_position_embedding)
            column_token_scores = self._bert_model.cls.predictions(column_token_span_representation)

            masked_context_token_loss = loss_fct(context_token_scores.view(-1, self.config.vocab_size), masked_context_token_labels.view(-1))
            masked_context_token_num = masked_context_token_labels.ne(-1).sum().item()

            masked_column_token_loss = loss_fct(column_token_scores.view(-1, self.config.vocab_size), masked_column_token_labels.view(-1))
            masked_column_token_num = masked_column_token_labels.ne(-1).sum().item()

            loss = masked_context_token_loss + masked_column_token_loss

            masked_context_token_loss = masked_context_token_loss.item()
            masked_context_token_ppl = math.exp(masked_context_token_loss / masked_context_token_num)
            masked_column_token_loss = masked_column_token_loss.item()
            masked_column_token_ppl = math.exp(masked_column_token_loss / masked_column_token_num)

            logging_info = {
                'sample_size': masked_context_token_num + masked_column_token_num,
                'masked_context_token_loss': masked_context_token_loss,
                'masked_context_token_num': masked_context_token_num,
                'masked_context_token_ppl': masked_context_token_ppl,
                'masked_column_token_loss': masked_column_token_loss,
                'masked_column_token_num': masked_column_token_num,
                'masked_column_token_ppl': masked_column_token_ppl,
            }

            if self.config.predict_cell_tokens:
                loss = loss + masked_cell_token_loss

                masked_cell_token_loss = masked_cell_token_loss.item()
                masked_cell_token_ppl = math.exp(masked_cell_token_loss / masked_cell_token_num)

                logging_info['masked_cell_token_loss'] = masked_cell_token_loss
                logging_info['masked_cell_token_num'] = masked_cell_token_num
                logging_info['masked_cell_token_ppl'] = masked_cell_token_ppl

                logging_info['sample_size'] += masked_cell_token_num

            logging_info['ppl'] = math.exp(loss.item() / logging_info['sample_size'])

            return loss, logging_info
        else:
            return context_encoding, schema_encoding

    def vertical_transform(self, context_encoding, context_token_mask, table_encoding, table_mask):
        # (batch_size, max_row_num, sequence_len)
        sequence_mask = torch.cat(
            [context_token_mask, table_mask],
            dim=-1
        )

        # (batch_size, sequence_len, 1, max_row_num, 1)
        attention_mask = sequence_mask.permute(0, 2, 1)[:, :, None, :, None]
        attention_mask = (1.0 - attention_mask) * -10000.0

        # (batch_size, max_row_num, sequence_len, encoding_size)
        bert_output = torch.cat([context_encoding, table_encoding], dim=2)

        hidden_states = bert_output
        vertical_layer_outputs = []
        for vertical_layer in self.vertical_transformer_layers:
            hidden_states = vertical_layer(hidden_states, attention_mask=attention_mask)
            vertical_layer_outputs.append(hidden_states)

        last_hidden_states = vertical_layer_outputs[-1] * sequence_mask.unsqueeze(-1)

        last_context_encoding = last_hidden_states[:, :, :context_encoding.size(2), :]
        last_table_encoding = last_hidden_states[:, :, context_encoding.size(2):, :]

        # mean-pool last encoding

        # (batch_size, 1, 1)
        table_row_nums = table_mask[:, :, 0].sum(dim=-1)[:, None, None]
        # (batch_size, context_len, hidden_size)
        mean_pooled_context_encoding = last_context_encoding.sum(dim=1) / table_row_nums
        # (batch_size, max_column_num, hidden_size)
        mean_pooled_schema_encoding = last_table_encoding.sum(dim=1) / table_row_nums

        return mean_pooled_context_encoding, mean_pooled_schema_encoding, last_table_encoding

    # noinspection PyUnboundLocalVariable
    def to_tensor_dict(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        table_specific_tensors=True
    ):
        examples = []
        for e_id, (context, table) in enumerate(zip(contexts, tables)):
            instance = self.input_formatter.get_input(context, table)

            for row_inst in instance['rows']:
                row_inst['token_ids'] = self.tokenizer.convert_tokens_to_ids(row_inst['tokens'])


            examples.append(instance)

        batch_size = len(contexts)

        tensor_dict = collate(examples, config=self.config, train=False)

        return tensor_dict, examples

    def validate(self, data_loader, args):
        gc.collect()

        keys = [
            'masked_context_token_loss',
            'masked_context_token_num',
            'masked_column_token_loss',
            'masked_column_token_num'
        ]

        if self.config.predict_cell_tokens:
            keys += [
                'masked_cell_token_loss',
                'masked_cell_token_num'
            ]

        was_training = self.training
        self.eval()

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
            'masked_context_token_ppl': math.exp(stats['masked_context_token_loss'] / stats['masked_context_token_num']),
            'masked_column_token_ppl': math.exp(stats['masked_column_token_loss'] / stats['masked_column_token_num'])
        }

        if self.config.predict_cell_tokens:
            valid_result['masked_cell_token_ppl'] = math.exp(stats['masked_cell_token_loss'] / stats['masked_cell_token_num'])

        return valid_result

    def encode(
            self,
            contexts: List[List[str]],
            tables: List[Table],
            return_bert_encoding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        assert return_bert_encoding is False, 'VerticalTableBert does not support `return_bert_encoding=True`'

        tensor_dict, instances = self.to_tensor_dict(contexts, tables)
        tensor_dict = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in tensor_dict.items()
        }

        context_encoding, schema_encoding = self.forward(**tensor_dict)

        tensor_dict['context_token_mask'] = tensor_dict['context_token_mask'][:, 0, :]
        tensor_dict['column_mask'] = tensor_dict['table_mask'][:, 0, :]

        info = {
            'tensor_dict': tensor_dict,
            'instances': instances
        }

        return context_encoding, schema_encoding, info
