#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import random

import torch.nn as nn
import torch.distributed
from fairseq.data import GroupedIterator
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule
from fairseq.options import eval_str_list
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import json
import numpy as np

from table_bert.vanilla_table_bert import VanillaTableBert
from table_bert.vertical.config import VerticalAttentionTableBertConfig
from table_bert.vertical.dataset import VerticalAttentionTableBertDataset
from table_bert.vertical.vertical_attention_table_bert import VerticalAttentionTableBert
from utils.comm import init_distributed_mode, init_signal_handler
from table_bert.config import TableBertConfig
from table_bert.dataset import TableDataset
from utils.evaluator import Evaluator
from utils.trainer import Trainer
from utils.util import init_logger


task_dict = {
    'vanilla': {
        'dataset': TableDataset,
        'config': TableBertConfig,
        'model': VanillaTableBert
    },
    'vertical_attention': {
        'dataset': VerticalAttentionTableBertDataset,
        'config': VerticalAttentionTableBertConfig,
        'model': VerticalAttentionTableBert
    }
}


def parse_train_arg():
    parser = ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='vanilla',
                        choices=['vanilla', 'vertical_attention'])
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--cpu",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)

    parser.add_argument("--base-model-name", type=str, required=False,
                        help="Bert pre-trained table_bert selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
                        default='bert-base-uncased')
    parser.add_argument("--table-bert-extra-config", type=json.loads, default='{}')
    parser.add_argument('--no-init', action='store_true', default=False)
    # parser.add_argument('--config-file', type=Path, help='table_bert config file if do not use pre-trained BERT table_bert.')

    # distributed training
    parser.add_argument("--ddp-backend", type=str, default='pytorch', choices=['pytorch', 'apex'])
    parser.add_argument("--local_rank", "--local-rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--master-port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--debug-slurm", action='store_true',
                        help="Debug multi-GPU / multi-node within a SLURM job")

    # training details
    parser.add_argument("--train-batch-size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--max-epoch", default=-1, type=int)
    # parser.add_argument("--total-num-update", type=int, default=1000000, help="Number of steps to train for")
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr-scheduler", type=str, default='polynomial_decay', help='Learning rate scheduler')
    parser.add_argument("--optimizer", type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--lr', '--learning-rate', default='0.00005', type=eval_str_list,
                        metavar='LR_1,LR_2,...,LR_N',
                        help='learning rate for the first N epochs; all epochs >N using LR_N'
                             ' (note: this may be interpreted differently depending on --lr-scheduler)')
    parser.add_argument('--clip-norm', default=0., type=float, help='clip gradient')
    parser.add_argument('--empty-cache-freq', default=0, type=int,
                        help='how often to clear the PyTorch CUDA cache (0 to disable)')
    parser.add_argument('--save-checkpoint-every-niter', default=10000, type=int)

    FairseqAdam.add_args(parser)
    PolynomialDecaySchedule.add_args(parser)

    # FP16 training
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--memory-efficient-fp16',
                        action='store_true',
                        help='Use memory efficient fp16')
    parser.add_argument('--threshold-loss-scale', type=float, default=None)
    parser.add_argument('--fp16-init-scale', type=float, default=128)
    # parser.add_argument('--fp16-scale-window', type=int, default=0)
    parser.add_argument('--fp16-scale-tolerance', type=float, default=0.0)
    parser.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
                        help='minimum FP16 loss scale, after which training is stopped')

    parser.add_argument('--debug-dataset', default=False, action='store_true')

    args = parser.parse_args()

    model_cls = task_dict[args.task]['model']
    if hasattr(model_cls, 'add_args'):
        model_cls.add_args(parser)
        args = parser.parse_args()

    return args


def main():
    args = parse_train_arg()
    task = task_dict[args.task]

    init_distributed_mode(args)
    logger = init_logger(args)

    if hasattr(args, 'base_model_name'):
        logger.warning('Argument base_model_name is deprecated! Use `--table-bert-extra-config` instead!')

    init_signal_handler()

    train_data_dir = args.data_dir / 'train'
    dev_data_dir = args.data_dir / 'dev'
    table_bert_config = task['config'].from_file(
        args.data_dir / 'config.json', **args.table_bert_extra_config)

    if args.is_master:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        with (args.output_dir / 'train_config.json').open('w') as f:
            json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

        logger.info(f'Table Bert Config: {table_bert_config.to_log_string()}')

        # copy the table bert config file to the working directory
        # shutil.copy(args.data_dir / 'config.json', args.output_dir / 'tb_config.json')
        # save table BERT config
        table_bert_config.save(args.output_dir / 'tb_config.json')

    assert args.data_dir.is_dir(), \
        "--data_dir should point to the folder of files made by pregenerate_training_data.py!"

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{torch.cuda.current_device()}')

    logger.info("device: {} gpu_id: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.local_rank, bool(args.multi_gpu), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    real_batch_size = args.train_batch_size  # // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.cpu:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logger.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model
    if args.multi_gpu and args.global_rank != 0:
        torch.distributed.barrier()

    if args.no_init:
        raise NotImplementedError
    else:
        model = task['model'](table_bert_config)

    if args.multi_gpu and args.global_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model = model.half()

    model = model.to(device)
    if args.multi_gpu:
        if args.ddp_backend == 'pytorch':
            model = nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
                device_ids=[args.local_rank], output_device=args.local_rank,
                broadcast_buffers=False
            )
        else:
            import apex
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

        model_ptr = model.module
    else:
        model_ptr = model

    # set up update parameters for LR scheduler
    dataset_cls = task['dataset']

    train_set_info = dataset_cls.get_dataset_info(train_data_dir, args.max_epoch)
    total_num_updates = train_set_info['total_size'] // args.train_batch_size // args.world_size // args.gradient_accumulation_steps
    args.max_epoch = train_set_info['max_epoch']
    logger.info(f'Train data size: {train_set_info["total_size"]} for {args.max_epoch} epochs, total num. updates: {total_num_updates}')

    args.total_num_update = total_num_updates
    args.warmup_updates = int(total_num_updates * 0.1)

    trainer = Trainer(model, args)

    checkpoint_file = args.output_dir / 'model.ckpt.bin'
    is_resumed = False
    # trainer.save_checkpoint(checkpoint_file)
    if checkpoint_file.exists():
        logger.info(f'Logging checkpoint file {checkpoint_file}')
        is_resumed = True
        trainer.load_checkpoint(checkpoint_file)

    model.train()

    # we also partitation the dev set for every local process
    logger.info('Loading dev set...')
    sys.stdout.flush()
    dev_set = dataset_cls(epoch=0, training_path=dev_data_dir, tokenizer=model_ptr.tokenizer, config=table_bert_config,
                          multi_gpu=args.multi_gpu, debug=args.debug_dataset)

    logger.info("***** Running training *****")
    logger.info(f"  Current config: {args}")

    if trainer.num_updates > 0:
        logger.info(f'Resume training at epoch {trainer.epoch}, '
                    f'epoch step {trainer.in_epoch_step}, '
                    f'global step {trainer.num_updates}')

    start_epoch = trainer.epoch
    for epoch in range(start_epoch, args.max_epoch):  # inclusive
        model.train()

        with torch.random.fork_rng(devices=None if args.cpu else [device.index]):
            torch.random.manual_seed(131 + epoch)

            epoch_dataset = dataset_cls(epoch=trainer.epoch, training_path=train_data_dir, config=table_bert_config,
                                        tokenizer=model_ptr.tokenizer, multi_gpu=args.multi_gpu, debug=args.debug_dataset)
            train_sampler = RandomSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=real_batch_size,
                                          num_workers=0,
                                          collate_fn=epoch_dataset.collate)

        samples_iter = GroupedIterator(iter(train_dataloader), args.gradient_accumulation_steps)
        trainer.resume_batch_loader(samples_iter)

        with tqdm(total=len(samples_iter), initial=trainer.in_epoch_step,
                  desc=f"Epoch {epoch}", file=sys.stdout, disable=not args.is_master, miniters=100) as pbar:

            for samples in samples_iter:
                logging_output = trainer.train_step(samples)

                pbar.update(1)
                pbar.set_postfix_str(', '.join(f"{k}: {v:.4f}" for k, v in logging_output.items()))

                if (
                    0 < trainer.num_updates and
                    trainer.num_updates % args.save_checkpoint_every_niter == 0 and
                    args.is_master
                ):
                    # Save model checkpoint
                    logger.info("** ** * Saving checkpoint file ** ** * ")
                    trainer.save_checkpoint(checkpoint_file)

            logger.info(f'Epoch {epoch} finished.')

            if args.is_master:
                # Save a trained table_bert
                logger.info("** ** * Saving fine-tuned table_bert ** ** * ")
                model_to_save = model_ptr  # Only save the table_bert it-self
                output_model_file = args.output_dir / f"pytorch_model_epoch{epoch:02d}.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))

            # perform validation
            logger.info("** ** * Perform validation ** ** * ")
            dev_results = trainer.validate(dev_set)

            if args.is_master:
                logger.info('** ** * Validation Results ** ** * ')
                logger.info(f'Epoch {epoch} Validation Results: {dev_results}')

            # flush logging information to disk
            sys.stderr.flush()

        trainer.next_epoch()


if __name__ == '__main__':
    main()
