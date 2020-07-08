#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from pathlib import Path
import re
import json
import torch
import numpy as np
import h5py
import shutil


def process_data_folder(data_dir: Path, output_dir: Path):
    epoch_stat_files = list(data_dir.glob('epoch_*.metrics.json'))
    epoch_ids = [
        int(re.search(r'epoch_(\d+).metrics.json', str(f)).group(1))
        for f in epoch_stat_files
    ]

    for epoch_id in epoch_ids:
        metric_file = data_dir / f'epoch_{epoch_id}.metrics.json'
        metric = json.load(metric_file.open())
        shard_num = metric['shard_num']
        shutil.copy(str(metric_file), str(output_dir / metric_file.name))

        for shard_id in range(shard_num):
            shard_data = torch.load(str(data_dir / f'epoch_{epoch_id}.shard{shard_id}.bin'))
            with h5py.File(str(output_dir / f'epoch_{epoch_id}.shard{shard_id}.h5'), 'w') as f:
                for key, val in shard_data.items():
                    f.create_dataset(key, data=val)


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)

    dev_dir = args.data_dir / 'dev'
    (args.output_dir / 'dev').mkdir(parents=True)
    process_data_folder(dev_dir, args.output_dir / 'dev')

    train_dir = args.data_dir / 'train'
    (args.output_dir / 'train').mkdir(parents=True)
    process_data_folder(train_dir, args.output_dir / 'train')

    shutil.copy(str(args.data_dir / 'config.json'), str(args.output_dir / 'config.json'))


if __name__ == '__main__':
    main()
