#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from functools import partial

import gc
import torch
import torch.distributed as dist
import torch.nn as nn
import logging
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from tqdm import tqdm
from typing import Dict
import math


class Evaluator(object):
    def __init__(self, batch_size, args):
        self.args = args
        self.batch_size = batch_size
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    def evaluate(self, model: nn.Module, dataset: Dataset) -> Dict:
        gc.collect()

        was_training = model.training
        model.eval()

        cum_loss = num_slots = 0.
        with torch.no_grad():
            data_loader = DataLoader(dataset,
                                     batch_size=self.batch_size, sampler=SequentialSampler(dataset),
                                     collate_fn=dataset.collate)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

            with tqdm(total=len(data_loader), desc=f"Evaluation", file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                    # (batch_size, max_seq_len, vocab_size)
                    predictions = model(batch['input_ids'], batch['token_type_ids'], batch['attention_mask'])

                    loss_sum = loss_fct(predictions.view(-1, predictions.size(-1)), batch['masked_lm_labels'].view(-1))
                    cum_loss += loss_sum.item()
                    num_slots += (batch['masked_lm_labels'] != -1).sum().item()

                    pbar.update(1)

        if was_training:
            model.train()

        # handel distributed evaluation
        if self.args.multi_gpu:
            result_tensor = torch.tensor([cum_loss, num_slots]).to(self.device)
            gather_list = [torch.zeros(2).to(self.device) for _ in range(dist.get_world_size())]
            torch.distributed.all_gather(gather_list, result_tensor)

            # if self.args.is_master:
            #     print([t for t in gather_list])

            cum_loss = sum(x[0].item() for x in gather_list)
            num_slots = sum(x[1].item() for x in gather_list)

        ppl = math.exp(cum_loss / num_slots)

        return {'ppl': ppl, 'cum_loss': cum_loss, 'num_slots': num_slots}
