#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import ArgumentParser
from pathlib import Path
import socket

import torch

from fairseq.options import eval_str_list
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule


def init_logger(args):
    # setup logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[{socket.gethostname()} | Node {args.node_id} | Rank {args.global_rank} | %(asctime)s] %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
