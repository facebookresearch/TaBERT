#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import sys
import math
from argparse import Namespace
from itertools import chain
from pathlib import Path
from typing import Dict
import logging

import fairseq
import numpy as np

import torch
from fairseq.checkpoint_utils import convert_state_dict_type
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, SequentialSampler

# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn

from fairseq import optim, distributed_utils, checkpoint_utils, utils as fairseq_utils
from fairseq.optim import lr_scheduler


class DummyCriterion(_Loss):
    pass


class Trainer(object):
    def __init__(self, model: nn.Module, args: Namespace):
        self.model = model
        self.args = args
        self._num_updates = 0
        self._epoch = 0
        self._in_epoch_step = 0
        self.cuda = not self.args.cpu and torch.cuda.is_available()
        self.logger = logging.getLogger()

        self.build_optimizer()

    @property
    def epoch(self):
        return self._epoch

    @property
    def in_epoch_step(self):
        return self._in_epoch_step

    def next_epoch(self):
        self._in_epoch_step = 0
        self._epoch += 1

    @property
    def device(self):
        return next(self.unwrapped_model.parameters()).device

    @property
    def unwrapped_model(self):
        return self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

    def build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                self.model.parameters(),
            )
        )

        if self.args.fp16:
            self.args.fp16_scale_window = 2 ** 14 / self.args.world_size / self.args.gradient_accumulation_steps
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            if self.args.memory_efficient_fp16:
                self.optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
            else:
                self.optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self.optimizer = optim.build_optimizer(self.args, params)

        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self.lr_scheduler.step_update(0)

    def prepare_sample(self, sample: Dict):
        def _apply_func(x):
            if torch.is_tensor(x):
                if self.cuda:
                    x = x.cuda()
                if self.args.fp16 and x.dtype is torch.float32:
                    x = x.half()

            return x

        return {
            key: _apply_func(val)
            for key, val
            in sample.items()
        }

    def train_step(self, samples):
        self.optimizer.zero_grad()
        logging_outputs = []

        for i, sample in enumerate(samples):
            sample = self.prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.args.world_size > 1
                    and hasattr(self.model, 'no_sync')
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            with maybe_no_sync():
                # forward and backward
                sample_size = int(sample['sample_size'])
                total_loss, logging_output = self.model(**sample)

                avg_loss = total_loss / sample_size
                self.optimizer.backward(avg_loss)

                # logging_output = {
                #     'sample_size': sample['sample_size'],
                #     'total_loss': total_loss.item()
                # }
                logging_outputs.append(logging_output)

        # gather logging outputs from all replicas
        # if self.args.world_size > 1:
        #     logging_outputs = distributed_utils.all_gather_list(logging_outputs)
        #     logging_outputs = list(chain.from_iterable(logging_outputs))

        # sample_size = sum(x['sample_size'] for x in logging_outputs)
        # logging_output = {
        #     'sample_size': sum(x['sample_size'] for x in logging_outputs),
        #     'total_loss': sum(x['total_loss'] for x in logging_outputs),
        # }
        # avg_loss = logging_output['total_loss'] / logging_output['sample_size']
        # avg_ppl = math.exp(avg_loss)
        # logging_output.update({
        #     'avg_loss': avg_loss,
        #     'avg_ppl': avg_ppl
        # })

        avg_logging_output = {
            k: np.average([x[k] for x in logging_outputs])
            for k in logging_outputs[0]
        }

        if (
            0 < self.args.empty_cache_freq <= self._num_updates and
            self._num_updates % self.args.empty_cache_freq == 0 and
            not self.args.cpu
        ):
            torch.cuda.empty_cache()

        try:
            # self.optimizer.multiply_grads(self.args.world_size / float(sample_size))

            # clip grads
            # if self.args.clip_norm > 0.:
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
            # logging.info(f'Iter {self._num_updates} Gradient Norm: {grad_norm}')

            # take an optimization step
            self.optimizer.step()
            self.take_one_step()
        except OverflowError as e:
            logging.error('| WARNING: overflow detected, ' + str(e))
            self.optimizer.zero_grad()

        return avg_logging_output

    def take_one_step(self):
        self._num_updates += 1
        self._in_epoch_step += 1
        self.lr_scheduler.step_update(self._num_updates)
        if self._num_updates >= self.lr_scheduler.total_num_update:
            logging.warning('Reached max num of updates')
            # exit(0)

    def validate(self, dataset):
        def collate_fn(x):
            return self.prepare_sample(dataset.collate(x))

        data_loader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size * 2, sampler=SequentialSampler(dataset),
            collate_fn=collate_fn
        )

        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        valid_result = model.validate(data_loader, self.args)

        return valid_result

    def save_checkpoint(self, ckpt_file: Path):
        if self.args.is_master:
            state_dict = {
                'args': self.args,
                'model': self.unwrapped_model.state_dict(),
                'last_optimizer_state': convert_state_dict_type(self.optimizer.state_dict()),
                'last_lr_scheduler_state': self.lr_scheduler.state_dict(),
                'trainer_state': {
                    'num_updates': self.num_updates,
                    'epoch': self.epoch,
                    'in_epoch_step': self._in_epoch_step
                },
            }

            torch.save(state_dict, ckpt_file)

    def load_checkpoint(self, ckpt_file: Path):
        if ckpt_file.exists():
            state = torch.load(ckpt_file, map_location=self.device)

            # load model parameters
            self.unwrapped_model.load_state_dict(state['model'], strict=True)

            self.build_optimizer()

            self.lr_scheduler.load_state_dict(state.get('last_lr_scheduler_state', None))
            self.optimizer.load_state_dict(state.get('last_optimizer_state', None))

            trainer_state = state['trainer_state']
            self.load_state_dict(trainer_state)

            saved_args = state['args']
            for key in [
                'world_size',
                'train_batch_size',
                'gradient_accumulation_steps'
            ]:
                assert getattr(saved_args, key) == getattr(self.args, key), \
                    f'{key} value must match between saved model checkpoint and current config, got ' \
                    f'{getattr(saved_args, key)} and {getattr(self.args, key)}'
        else:
            raise FileNotFoundError(ckpt_file)

    @property
    def num_updates(self):
        return self._num_updates

    def load_state_dict(self, state_dict):
        self._num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        self._epoch = state_dict['epoch']
        self._in_epoch_step = state_dict['in_epoch_step']

    def resume_batch_loader(self, samples_iter):
        """Resume batch loader to in_epoch_step if we resume training"""
        if self.in_epoch_step > 0:
            for _ in range(self.in_epoch_step):
                x = next(samples_iter)
