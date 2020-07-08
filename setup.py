#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='table_bert',
    version='0.1',  # Required
    description='Learning contextual representations on textual and structured tabular data',
    author='Pengcheng Yin',
    author_email='pcyin@cs.cmu.edu',

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(exclude=['exp_runs', 'scripts', 'preprocess', 'utils', 'examples']),
    python_requires='>=3.6'
)
