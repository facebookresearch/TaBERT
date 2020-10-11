#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import re

import ujson
import msgpack
import logging
import math
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Iterator, Set
import redis

import numpy as np
import torch
import zmq
from table_bert.utils import BertTokenizer
from table_bert.config import TableBertConfig
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
from tqdm import tqdm
from table_bert.table import Column, Table
import h5py


class DistributedSampler(Sampler):
    """Sampler that restricts preprocess loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class TableDataset(Dataset):
    DEFAULT_CONFIG_CLS = TableBertConfig

    def __init__(self, training_path, epoch=0, config=None, tokenizer=None, reduce_memory=False, multi_gpu=False, indices=None, debug=False):
        # self.vocab = tokenizer.vocab
        # self.tokenizer = tokenizer
        self.data_epoch = self.epoch = epoch
        # self.data_epoch = epoch % num_data_epochs
        data_file_prefix = training_path / f"epoch_{self.data_epoch}"
        # metrics_file = training_path / f"epoch_{self.data_epoch}.metrics.json"
        # assert metrics_file.is_file()
        # metrics = json.loads(metrics_file.read_text())
        # dataset_size = metrics['num_training_examples']

        epoch_info = self.get_epoch_shards_info(data_file_prefix)
        epoch_dataset_size = epoch_info['total_size']

        assert reduce_memory is False, 'reduce_memory is not implemented'

        self.config = config or self.DEFAULT_CONFIG_CLS()

        if not indices:
            if multi_gpu:
                num_shards = torch.distributed.get_world_size()
                local_shard_id = torch.distributed.get_rank()

                shard_size = epoch_dataset_size // num_shards

                logging.info(f'dataset_size={epoch_dataset_size}, shard_size={shard_size}')

                g = torch.Generator()
                g.manual_seed(self.epoch)
                indices = torch.randperm(epoch_dataset_size, generator=g).tolist()

                # make it evenly divisible
                indices = indices[:shard_size * num_shards]
                assert len(indices) == shard_size * num_shards

                # subsample
                indices = indices[local_shard_id:len(indices):num_shards]
                assert len(indices) == shard_size

                indices = set(indices)
            else:
                indices = set(range(epoch_dataset_size))

        indices = set(indices)
        if debug:
            indices = set(list(indices)[:1000])

        logging.info(f"Loading examples from {training_path} for epoch {epoch}")
        if indices:
            logging.info(f'Load a sub-sample of the whole dataset')

        self.examples = self.load_epoch(data_file_prefix, epoch_info['shard_num'], indices)

    @classmethod
    def get_dataset_info(cls, data_path: Path, max_epoch=-1):
        if max_epoch == -1:
            epoch_files = list(data_path.glob('epoch_*'))
            epoch_ids = [
                int(re.search(r'epoch_(\d+)', str(f)).group(1))
                for f in epoch_files
            ]
            max_epoch = max(epoch_ids) + 1

        data_size = 0
        for epoch_id in range(max_epoch):
            epoch_file = data_path / f'epoch_{epoch_id}'
            epoch_info = cls.get_epoch_shards_info(epoch_file)
            data_size += epoch_info['total_size']

        return {
            'total_size': data_size,
            'max_epoch': max_epoch
        }

    @classmethod
    def get_shard_size(cls, shard_file: Path):
        with h5py.File(str(shard_file), 'r', rdcc_nbytes=1024 * 1024 * 2048) as data:
            shard_size = data['masked_lm_offsets'].shape[0]

        return shard_size

    @classmethod
    def get_epoch_shards_info(cls, shard_file_prefix: Path):
        shard_files = list(shard_file_prefix.parent.glob(shard_file_prefix.name + '.shard*.h5'))
        shard_ids = [int(re.search(r'shard(\d+)', str(f)).group(1)) for f in shard_files]
        shard_num = max(shard_ids) + 1

        cum_size = 0
        for shard_file in shard_files:
            print(shard_file)
            shard_size = cls.get_shard_size(shard_file)
            cum_size += shard_size

        return {
            'shard_num': shard_num,
            'total_size': cum_size
        }

    def load_epoch(self, file_prefix: Path, shard_num: int, valid_indices: Set = None):
        examples = []
        idx = -1
        for shard_id in range(shard_num):
            file_name = file_prefix.with_suffix(f'.shard{shard_id}.h5')
            if file_name.exists():
                data = h5py.File(str(file_name), 'r', rdcc_nbytes=1024 * 1024 * 2048)
            else:
                file_name = file_name.with_suffix('.bin')
                data = torch.load(str(file_name))

            sequences = data['sequences']
            segment_a_lengths = data['segment_a_lengths']
            sequence_offsets = data['sequence_offsets']
            masked_lm_positions = data['masked_lm_positions']
            masked_lm_label_ids = data['masked_lm_label_ids']
            masked_lm_offsets = data['masked_lm_offsets']

            shard_size = len(segment_a_lengths)

            for i in range(shard_size):
                idx += 1

                if valid_indices and idx not in valid_indices:
                    continue

                example = {}

                seq_begin, seq_end = sequence_offsets[i]
                example['token_ids'] = sequences[seq_begin: seq_end]

                seq_a_length = segment_a_lengths[i]
                example['sequence_a_length'] = seq_a_length

                tgt_begin, tgt_end = masked_lm_offsets[i]
                example['masked_lm_positions'] = masked_lm_positions[tgt_begin: tgt_end]
                example['masked_lm_label_ids'] = masked_lm_label_ids[tgt_begin: tgt_end]

                examples.append(example)

            if isinstance(data, h5py.File):
                data.close()

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def collate(examples):
        batch_size = len(examples)
        max_len = max(len(e['token_ids']) for e in examples)

        input_array = np.zeros((batch_size, max_len), dtype=np.int)
        mask_array = np.zeros((batch_size, max_len), dtype=np.bool)
        segment_array = np.zeros((batch_size, max_len), dtype=np.bool)
        lm_label_array = np.full((batch_size, max_len), dtype=np.int, fill_value=-1)

        for e_id, example in enumerate(examples):
            token_ids = example['token_ids']
            # print(tokenizer.convert_ids_to_tokens(token_ids))
            # assert tokenizer.convert_ids_to_tokens([token_ids[0]]) == ['[CLS]'] and \
            #        tokenizer.convert_ids_to_tokens([token_ids[-1]]) == ['[SEP]']

            masked_label_ids = example['masked_lm_label_ids']
            masked_lm_positions = example['masked_lm_positions']

            input_array[e_id, :len(token_ids)] = token_ids
            mask_array[e_id, :len(token_ids)] = 1
            segment_array[e_id, example['sequence_a_length']:] = 1
            lm_label_array[e_id, masked_lm_positions] = masked_label_ids

        # input_ids, input_mask, segment_ids, lm_label_ids
        return {
            'input_ids': torch.tensor(input_array.astype(np.int64)),
            'attention_mask': torch.tensor(mask_array.astype(np.int64)),
            'token_type_ids': torch.tensor(segment_array.astype(np.int64)),
            'masked_lm_labels': torch.tensor(lm_label_array.astype(np.int64)),
            'sample_size': (lm_label_array != -1).sum()
        }


class Example(object):
    def __init__(self, uuid, header, context, column_data=None, **kwargs):
        self.uuid = uuid
        self.header = header
        self.context = context
        self.column_data = column_data

        for key, val in kwargs.items():
            setattr(self, key, val)

    def serialize(self):
        example = {
            'uuid': self.uuid,
            'source': self.source,
            'context': self.context,
            'column_data': self.column_data,
            'header': [x.to_dict() for x in self.header]
        }

        return example

    def get_table(self):
        num_columns = len(self.header)
        num_rows = len(self.column_data[0])
        row_data = []
        for row_id in range(num_rows):
            row = [self.column_data[i][row_id] for i in range(num_columns)]
            row_data.append(row)

        table = Table(self.uuid, header=self.header, data=row_data)

        return table

    @classmethod
    def from_serialized(cls, data) -> 'Example':
        header = [Column(**x) for x in data['header']]
        data['header'] = header
        return Example(**data)

    @classmethod
    def from_dict(cls, entry: Dict, tokenizer: Optional[BertTokenizer], suffix) -> 'Example':
        def _get_data_source():
            return 'wiki' if 'wiki' in entry['uuid'] else 'common_crawl'

        source = _get_data_source()

        header_entry = entry['header'] if source == 'wiki' else entry['table']['header']
        header = []
        column_data = []
        for col in header_entry:
            sample_value = col['sample_value']['value']
            if tokenizer:
                name_tokens = tokenizer.tokenize(col['name'])
            else: name_tokens = None
            column = Column(col['name'],
                            col['type'],
                            sample_value,
                            name_tokens=name_tokens)
            header.append(column)

        if source == 'wiki':
            for row in entry['data'][1:]:
                for col_id, (tag, cell_val) in enumerate(row):
                    if col_id >= len(column_data):
                        column_data.append([])

                    column_data[col_id].append(cell_val)
        else:
            for row in entry['table']['data']:
                for col_id, (cell_val) in enumerate(row):
                    if col_id >= len(column_data):
                        column_data.append([])

                    column_data[col_id].append(cell_val)

        context_before = []
        context_after = []

        if source == 'wiki':
            for para in entry['context_before']:
                for sent in para:
                    if tokenizer:
                        sent = tokenizer.tokenize(sent)

                    context_before.append(sent)

            caption = entry['caption']
            if caption:
                if tokenizer:
                    caption = tokenizer.tokenize(entry['caption'])

                context_before.append(caption)
        else:
            for sent in entry['context_before']:
                if tokenizer:
                    sent = tokenizer.tokenize(sent)
                context_before.append(sent)

            for sent in entry['context_after']:
                if tokenizer:
                    sent = tokenizer.tokenize(sent)
                context_after.append(sent)

        uuid = entry['uuid']

        return cls(uuid, header,
                   [context_before, context_after],
                   column_data=column_data,
                   source=source)


class TableDatabase:
    def __init__(self, tokenizer, backend='redis', num_workers=None):
        self.tokenizer = tokenizer
        self.backend = backend

        if self.backend == 'redis':
            self.restore_redis_client()
            self.client.flushall(asynchronous=False)

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
            if num_workers > 20:
                num_workers -= 2

        self.num_workers = num_workers
        self._cur_index = multiprocessing.Value('i', 0)

    def restore_redis_client(self):
        self.client = redis.Redis(host='localhost', port=6379, db=0)

    @staticmethod
    def __load_process_zmq(file, num_workers):
        context = zmq.Context()
        job_sender = context.socket(zmq.PUSH)
        # job_sender.setsockopt(zmq.LINGER, -1)
        job_sender.bind("tcp://127.0.0.1:5557")

        controller = context.socket(zmq.PUB)
        controller.bind('tcp://127.0.0.1:5558')

        # wait for sometime to let all workers to connect
        time.sleep(5)

        cnt = 0
        with file.open() as f:
            for line in f:
                cnt += 1
                job_sender.send_string(line)
                # if cnt % 10000 == 0:
                #     print(f'read {cnt} examples')
                #     sys.stdout.flush()

        controller.send_string('kill')
        # print('Reader count:', cnt)

        job_sender.close()
        controller.close()
        context.destroy()
        # while True:
        #     job_sender.send_string('')
        #     time.sleep(0.1)

        # while True:
        #     time.sleep(1)

    @staticmethod
    def __example_worker_process_zmq(tokenizer, db):
        context = zmq.Context()
        job_receiver = context.socket(zmq.PULL)
        # job_receiver.setsockopt(zmq.LINGER, -1)
        job_receiver.connect("tcp://127.0.0.1:5557")

        controller = context.socket(zmq.SUB)
        controller.connect("tcp://127.0.0.1:5558")
        controller.setsockopt(zmq.SUBSCRIBE, b"")

        poller = zmq.Poller()
        poller.register(job_receiver, zmq.POLLIN)
        poller.register(controller, zmq.POLLIN)

        cache_client = redis.Redis(host='localhost', port=6379, db=0)
        buffer_size = 20000

        def _add_to_cache():
            if buffer:
                with db._cur_index.get_lock():
                    index_end = db._cur_index.value + len(buffer)
                    db._cur_index.value = index_end
                index_start = index_end - len(buffer)
                values = {str(i): val for i, val in zip(range(index_start, index_end), buffer)}
                cache_client.mset(values)
                del buffer[:]

        cnt = 0
        buffer = []
        can_exit = False
        while True:
            triggered = False
            socks = dict(poller.poll(timeout=2000))

            if socks.get(job_receiver) == zmq.POLLIN:
                triggered = True
                job = job_receiver.recv_string()
                if job:
                    cnt += 1
                    # print(cnt)
                    example = Example.from_dict(ujson.loads(job), tokenizer, suffix=None)

                    if TableDatabase.is_valid_example(example):
                        data = example.serialize()
                        buffer.append(msgpack.packb(data, use_bin_type=True))

                    if len(buffer) >= buffer_size:
                        _add_to_cache()

            # else:
            #     job_receiver.close()
            #     _add_to_cache()
            #     break

            if socks.get(controller) == zmq.POLLIN:
                triggered = True
                print(controller.recv_string())
                can_exit = True

            # timeout
            # print(socks)
            if not socks and can_exit:
                print('Processor exit...')
                break

            if socks and not triggered:
                print(socks)

        _add_to_cache()
        job_receiver.close()
        controller.close()
        context.destroy()

    @classmethod
    def from_jsonl(
        cls,
        file_path: Path,
        tokenizer: Optional[BertTokenizer] = None,
        backend='redis',
        num_workers=None,
        indices=None
    ) -> 'TableDatabase':
        file_path = Path(file_path)

        db = cls(backend=backend, num_workers=num_workers, tokenizer=tokenizer)

        if backend == 'redis':
            assert indices is None
            db.load_data_to_redis(file_path)
        elif backend == 'memory':
            example_store = dict()
            if indices: indices = set(indices)

            with file_path.open() as f:
                for idx, json_line in enumerate(tqdm(f, desc=f'Loading Tables from {str(file_path)}', unit='entries', file=sys.stdout)):
                    if indices and idx not in indices:
                        continue

                    example = Example.from_dict(
                        ujson.loads(json_line),
                        tokenizer,
                        suffix=None
                    )

                    if TableDatabase.is_valid_example(example):
                        example_store[idx] = example
                        
            db.__example_store = example_store

        return db

    def load_data_to_redis(self, file_path: Path):
        reader = multiprocessing.Process(target=self.__load_process_zmq, args=(file_path, self.num_workers),
                                         daemon=True)

        workers = []
        for _ in range(self.num_workers):
            worker = multiprocessing.Process(target=self.__example_worker_process_zmq,
                                             args=(self.tokenizer, self),
                                             daemon=True)
            worker.start()
            workers.append(worker)

        while any(not worker.is_alive() for worker in workers):
            time.sleep(0.1)

        reader.start()

        stop_count = 0
        db_size = 0
        with tqdm(desc=f"Loading Tables from {str(file_path)}", unit=" entries", file=sys.stdout) as pbar:
            while True:
                cur_db_size = len(self)
                pbar.update(cur_db_size - db_size)
                db_size = cur_db_size

                all_worker_finished = all(not w.is_alive() for w in workers)
                if all_worker_finished:
                    print(f'all workers stoped!')
                    break

                time.sleep(1)

        for worker in workers:
            worker.join()
        reader.terminate()

    def __len__(self):
        if self.backend == 'redis':
            return self._cur_index.value
        elif self.backend == 'memory':
            return len(self.__example_store)
        else:
            raise RuntimeError()

    def __contains__(self, item):
        assert self.backend == 'memory'
        return item in self.__example_store

    def __getitem__(self, item) -> Example:
        if self.backend == 'redis':
            result = self.client.get(str(item))
            if result is None:
                raise IndexError(item)
    
            example = Example.from_serialized(msgpack.unpackb(result, raw=False))
        elif self.backend == 'memory':
            example = self.__example_store[item]
        else:
            raise RuntimeError()

        return example

    def __iter__(self) -> Iterator[Example]:
        if self.backend == 'redis':
            for i in range(len(self)):
                yield self[i]
        else:
            for example in self.__example_store.values():
                yield example

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.backend == 'redis':
            print('Flushing all entries in cache')
            self.client.flushall()

    @classmethod
    def is_valid_example(cls, example):
        # TODO: move this to preprocess pre-processing
        if any(len(col.name.split(' ')) > 10 for col in example.header):
            return False

        if any(len(col.name_tokens) == 0 for col in example.header):
            return False

        return True
