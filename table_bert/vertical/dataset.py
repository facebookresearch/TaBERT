#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import torch
from typing import Dict, List, Set
import numpy as np
import h5py
from table_bert.vertical.config import VerticalAttentionTableBertConfig
from tqdm import tqdm


from table_bert.dataset import TableDataset


def check_row_example(table_size, row_example):
    row_num, column_num = table_size
    column_token_position_to_column_ids = row_example['column_token_position_to_column_ids']

    mask_val = np.iinfo(np.uint16).max
    col_ids = [x for x in column_token_position_to_column_ids if x != mask_val]
    assert max(col_ids) < column_num


def collate(examples: List[Dict], config: VerticalAttentionTableBertConfig, train=True) -> Dict[str, torch.Tensor]:
    batch_size = len(examples)
    max_sequence_len = max(
        len(row['token_ids'])
        for e in examples
        for row in e['rows']
    )
    max_context_len = max(
        row['context_span'][1] - row['context_span'][0]
        for e in examples
        for row in e['rows']
    )
    max_row_num = max(len(inst['rows']) for inst in examples)
    # max_row_num = max(inst['table_size'][0] for inst in examples)
    # max_column_num = max(inst['table_size'][1] for inst in examples)

    input_ids = np.zeros((batch_size, max_row_num, max_sequence_len), dtype=np.int64)
    mask_array = np.zeros((batch_size, max_row_num, max_sequence_len), dtype=np.float32)
    segment_ids = np.zeros((batch_size, max_row_num, max_sequence_len), dtype=np.int64)

    # table specific tensors
    context_token_positions = np.zeros((batch_size, max_row_num, max_context_len), dtype=np.int)
    context_token_mask = np.zeros((batch_size, max_row_num, max_context_len), dtype=np.bool)

    row_column_nums = []

    # we initialize the mapping with the id of last column as the "garbage collection" entry for reduce ops
    column_token_to_column_id_fill_val = np.iinfo(np.uint16).max
    column_token_position_to_column_ids = np.full((batch_size, max_row_num, max_sequence_len), dtype=np.int, fill_value=column_token_to_column_id_fill_val)
    # column_token_mask = np.zeros((batch_size, max_row_num, max_sequence_len), dtype=np.bool)

    if train:
        # MLM objectives
        masked_context_token_label_ids = np.full((batch_size, max_context_len), dtype=np.int64, fill_value=-1)

        max_column_pred_token_num = max(len(e['masked_column_token_column_ids']) for e in examples)
        masked_column_token_column_ids = np.zeros((batch_size, max_column_pred_token_num), dtype=np.int64)
        masked_column_token_label_ids = np.full((batch_size, max_column_pred_token_num), dtype=np.int64, fill_value=-1)

        # cell token prediction
        predict_cell_tokens = config.predict_cell_tokens
        if predict_cell_tokens:
            max_masked_cell_token_num = max(
                len(row['masked_cell_token_positions'])
                for e in examples
                for row in e['rows']
            )

            masked_cell_token_positions = np.zeros((batch_size, max_row_num, max_masked_cell_token_num), dtype=np.int64)
            masked_cell_token_column_ids = np.zeros((batch_size, max_row_num, max_masked_cell_token_num), dtype=np.int64)
            masked_cell_token_label_ids = np.full((batch_size, max_row_num, max_masked_cell_token_num), dtype=np.int64,
                                                  fill_value=-1)

    for e_id, example in enumerate(examples):
        for row_id, row_inst in enumerate(example['rows']):
            bert_input_seq_length = len(row_inst['token_ids'])

            input_ids[e_id, row_id, :bert_input_seq_length] = row_inst['token_ids']
            mask_array[e_id, row_id, :bert_input_seq_length] = 1
            segment_ids[e_id, row_id, row_inst['segment_a_length']:] = 1

            row_context_token_positions = list(range(
                row_inst['context_span'][0],
                row_inst['context_span'][1]
            ))
            context_token_positions[e_id, row_id, :len(row_context_token_positions)] = row_context_token_positions
            context_token_mask[e_id, row_id, row_context_token_positions] = 1

            row_column_token_position_to_column_ids = row_inst['column_token_position_to_column_ids']
            if not train:
                row_column_token_position_to_column_ids = np.array(row_column_token_position_to_column_ids)

            cur_column_num = row_column_token_position_to_column_ids[row_column_token_position_to_column_ids != column_token_to_column_id_fill_val].max() + 1
            row_column_nums.append(cur_column_num)
            column_token_position_to_column_ids[e_id, row_id, :len(row_column_token_position_to_column_ids)] = row_column_token_position_to_column_ids

            if train and predict_cell_tokens:
                row_masked_cell_token_positions = row_inst['masked_cell_token_positions']
                masked_cell_token_positions[e_id, row_id, :len(row_masked_cell_token_positions)] = row_masked_cell_token_positions
                masked_cell_token_column_ids[e_id, row_id, :len(row_masked_cell_token_positions)] = [
                    row_column_token_position_to_column_ids[pos] for pos in row_masked_cell_token_positions]
                masked_cell_token_label_ids[e_id, row_id, :len(row_masked_cell_token_positions)] = row_inst['masked_cell_token_label_ids']

        # row_num, column_num = example['table_size']
        # table_mask[e_id, :row_num, :column_num] = 1.

        if train:
            masked_context_token_label_ids[e_id, example['masked_context_token_positions']] = example['masked_context_token_label_ids']

            masked_column_token_num = len(example['masked_column_token_column_ids'])
            masked_column_token_column_ids[e_id, :masked_column_token_num] = example['masked_column_token_column_ids']
            masked_column_token_label_ids[e_id, :masked_column_token_num] = example['masked_column_token_label_ids']

    max_column_num = max(row_column_nums)
    table_mask = np.zeros((batch_size, max_row_num, max_column_num), dtype=np.bool)
    global_col_id = 0
    for e_id, example in enumerate(examples):
        for row_id, row_inst in enumerate(example['rows']):
            row_column_num = row_column_nums[global_col_id]
            table_mask[e_id, row_id, :row_column_num] = 1
            global_col_id += 1

    column_token_position_to_column_ids[
        column_token_position_to_column_ids == column_token_to_column_id_fill_val] = max_column_num

    # for table_id in range(len(examples)):
    #     row_num, column_num = examples[table_id]['table_size']
    #     for row_id in range(row_num):
    #         for column_id in range(column_num):
    #             assert column_id in column_token_position_to_column_ids[table_id, row_id]
    #     for masked_col_id in masked_column_token_column_ids[table_id]:
    #         assert masked_col_id < column_num
    #
    # assert column_token_position_to_column_ids.flatten().max() == table_mask.sum(axis=-1).max()

    tensor_dict = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
        'context_token_positions': torch.tensor(context_token_positions, dtype=torch.long),
        'column_token_position_to_column_ids': torch.tensor(column_token_position_to_column_ids, dtype=torch.long),
        'sequence_mask': torch.tensor(mask_array, dtype=torch.float32),
        'context_token_mask': torch.tensor(context_token_mask, dtype=torch.float32),
        'table_mask': torch.tensor(table_mask, dtype=torch.float32),
    }

    if train:
        sample_size = int((masked_context_token_label_ids != -1).sum() + (masked_column_token_label_ids != -1).sum())
        if predict_cell_tokens:
            sample_size += int((masked_cell_token_label_ids != -1).sum())

        tensor_dict.update({
            'masked_context_token_labels': torch.tensor(masked_context_token_label_ids, dtype=torch.long),
            'masked_column_token_column_ids': torch.tensor(masked_column_token_column_ids, dtype=torch.long),
            'masked_column_token_labels': torch.tensor(masked_column_token_label_ids, dtype=torch.long),
            'sample_size': sample_size
        })

        if predict_cell_tokens:
            tensor_dict.update({
                'masked_cell_token_positions': torch.tensor(masked_cell_token_positions, dtype=torch.long),
                'masked_cell_token_column_ids': torch.tensor(masked_cell_token_column_ids, dtype=torch.long),
                'masked_cell_token_labels': torch.tensor(masked_cell_token_label_ids, dtype=torch.long),
            })

    return tensor_dict


def deserialize_row_data(row_data, config: VerticalAttentionTableBertConfig):
    row_data_chunk_len = row_data[0]
    assert row_data_chunk_len == len(row_data)

    sequence_len = row_data[1]
    token_ids = row_data[2: 2 + sequence_len]
    ptr = 2 + sequence_len
    segment_a_length = row_data[ptr]
    ptr += 1
    context_span = row_data[ptr], row_data[ptr + 1]
    ptr += 2
    column_token_position_to_column_ids = row_data[ptr: ptr + sequence_len]
    ptr += sequence_len

    row_inst = {
        'token_ids': token_ids,
        'segment_a_length': segment_a_length,
        'context_span': context_span,
        'column_token_position_to_column_ids': column_token_position_to_column_ids
    }

    if config.predict_cell_tokens:
        remain_len = row_data_chunk_len - ptr
        row_inst['masked_cell_token_positions'] = row_data[ptr: ptr + remain_len // 2]
        row_inst['masked_cell_token_label_ids'] = row_data[ptr + remain_len // 2:]

    return row_inst


def serialize_row_data(example, config: VerticalAttentionTableBertConfig):
    payload = (
        [len(example['token_ids'])] +
        example['token_ids'] +
        [example['segment_a_length']] +
        list(example['context_span']) +
        example['column_token_position_to_column_ids']
    )

    if config.predict_cell_tokens:
        payload += example['masked_cell_token_positions'] + example['masked_cell_token_label_ids']
        assert len(example['masked_cell_token_positions']) == len(example['masked_cell_token_label_ids'])

    assert len(example['token_ids']) == len(example['column_token_position_to_column_ids'])
    data_size = len(payload) + 1
    row_data = [data_size] + payload

    return row_data


class VerticalAttentionTableBertDataset(TableDataset):
    DEFAULT_CONFIG_CLS = VerticalAttentionTableBertConfig

    @classmethod
    def get_shard_size(cls, shard_file: Path):
        with h5py.File(str(shard_file), 'r', rdcc_nbytes=1024 * 1024 * 2048) as data:
            shard_size = data['mlm_data_offsets'].shape[0]

        return shard_size

    def collate(self, examples):
        return collate(examples, self.config)

    def load_epoch(self, file_prefix: Path, shard_num: int, valid_indices: Set = None):
        examples = []
        idx = -1
        finished = False
        for shard_id in range(shard_num):
            # if shard_id > 0:
            #     break

            file_name = file_prefix.with_suffix(f'.shard{shard_id}.bin')
            if file_name.exists():
                data = torch.load(str(file_name))
                use_hdf5 = False
            else:
                file_name = file_prefix.with_suffix(f'.shard{shard_id}.h5')
                data = h5py.File(str(file_name), 'r', rdcc_nbytes=1024 * 1024 * 2048)
                use_hdf5 = True

            # [
            #   row_data_length,
            #   token_id_length,
            #   token_ids (List),
            #   segment_a_length (1),
            #   context_span (2),
            #   column_token_position_to_column_ids (List),
            # ]
            row_data_sequences = data['row_data_sequences']
            # [row_num, column_num, start_index, end_index]
            row_data_offsets = data['row_data_offsets']

            # [
            #   masked_context_token_positions,
            #   masked_context_token_labels,
            #   masked_column_token_column_ids,
            #   masked_column_token_labels
            # ]
            mlm_data_sequences = data['mlm_data_sequences']
            # [start_index, start_index, start_index, start_index, end_index]
            mlm_data_offsets = data['mlm_data_offsets']

            shard_size = mlm_data_offsets.shape[0]

            for chunk_id in tqdm(range(shard_size), desc=f'Loading shard {shard_id}', file=sys.stdout, miniters=10000):
                idx += 1

                # if idx >= 1000:
                #     finished = True
                #     break

                if valid_indices and idx not in valid_indices:
                    continue

                example = {
                    'rows': []
                }

                row_num, column_num, start_index, end_index = row_data_offsets[chunk_id]
                table_data = np.asarray(row_data_sequences[start_index: end_index])

                rows = []
                row_start_idx = 0
                for _ in range(row_num):
                    row_data_length = table_data[row_start_idx]
                    row_end_idx = row_start_idx + row_data_length
                    row_data = table_data[row_start_idx: row_end_idx]

                    row_instance = deserialize_row_data(row_data, self.config)
                    rows.append(row_instance)

                    # try:
                    #     check_row_example((row_num, column_num), row_instance)
                    # except AssertionError:
                    #     stat = {
                    #         'table_size': (row_num, column_num),
                    #         'rows': rows
                    #     }
                    #     torch.save(stat, 'debug.bin')
                    #     print('!!!!Data Error!!!!', file=sys.stderr)
                    #     exit(-1)

                    row_start_idx = row_end_idx

                assert row_end_idx == len(table_data)
                example['rows'] = rows
                example['table_size'] = (row_num, column_num)

                s1, s2, s3, s4, s5 = mlm_data_offsets[chunk_id]
                example['masked_context_token_positions'] = np.asarray(mlm_data_sequences[s1: s2])
                example['masked_context_token_label_ids'] = np.asarray(mlm_data_sequences[s2: s3])
                example['masked_column_token_column_ids'] = np.asarray(mlm_data_sequences[s3: s4])
                example['masked_column_token_label_ids'] = np.asarray(mlm_data_sequences[s4: s5])

                examples.append(example)

            if use_hdf5:
                data.close()

            del data

            if finished: break

        return examples


def main():
    dev_data = VerticalAttentionTableBertDataset(Path('data/sampled_data/train_data/dev'))


if __name__ == '__main__':
    main()
