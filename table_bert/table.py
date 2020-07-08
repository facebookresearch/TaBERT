#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Any, Union
import pandas as pd

from table_bert.utils import BertTokenizer


class Column(object):
    def __init__(
        self,
        name: str,
        type: str,
        sample_value: Any = None,
        is_primary_key: bool = False,
        foreign_key: 'Column' = None,
        name_tokens: List[str] = None,
        sample_value_tokens: List[str] = None,
        **kwargs
    ):
        self.name = name
        self.name_tokens = name_tokens
        self.type = type
        self.sample_value = sample_value
        self.sample_value_tokens = sample_value_tokens
        self.foreign_key: Column = foreign_key
        self.is_primary_key = is_primary_key

        self.fields = []
        for key, val in kwargs.items():
            self.fields.append(key)
            setattr(self, key, val)

    def copy(self):
        return Column(
            **self.to_dict()
        )

    def to_dict(self):
        data = {
            'name': self.name,
            'name_tokens': self.name_tokens,
            'type': self.type,
            'sample_value': self.sample_value,
            'sample_value_tokens': self.sample_value_tokens,
            'is_primary_key': self.is_primary_key,
            'foreign_key': self.foreign_key
        }

        for key in self.fields:
            data[key] = getattr(self, key)

        return data

    def __setattr__(self, key, value):
        # if key == 'table':
        #     assert getattr(self, key, None) is None, f'The column has already been bind to a table `{self.table}`. ' \
        #                                              f'Please remove the reference to the existing table first'

        # if key != 'fields' and getattr(self, key, None) is None:
        #     self.fields.append(key)

        super(Column, self).__setattr__(key, value)

    def __hash__(self):
        # table = (None, ) if self.table is None else (self.table.id, self.table.name)
        return hash((self.name, self.type))

    def __eq__(self, other):
        if not isinstance(other, Column):
            return False

        # if self.table is not other.table:
        #     return False

        # if self.table and (other.table.id != self.table.id or other.table.name != self.table.name):
        #     return False

        return self.name == other.name and self.type == other.type

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f'Column[name={self.name}, type={self.type}]'

    __str__ = __repr__


class Table(object):
    def __init__(
        self,
        id,
        header: List[Column],
        data: Union[List[Dict], List[List]] = None,
        name: str = None,
        **kwargs
    ):
        self.id = id
        self.name = name
        self.header = header
        self.header_index = {column.name: column for column in header}
        self.data: List[Any] = data
        self.fields = []

        # for column in self.header:
        #     setattr(column, 'table', self)

        for key, val in kwargs.items():
            self.fields.append(key)
            setattr(self, key, val)

    def tokenize(self, tokenizer: BertTokenizer):
        for column in self.header:
            column.name_tokens = tokenizer.tokenize(column.name)
            if column.sample_value is not None:
                column.sample_value_tokens = tokenizer.tokenize(
                    str(column.sample_value))

        tokenized_rows = [
            {k: tokenizer.tokenize(str(v)) for k, v in row.items()}
            if isinstance(row, dict)
            else [tokenizer.tokenize(str(v)) for v in row]

            for row in self.data
        ]

        self.data = tokenized_rows

        setattr(self, 'tokenized', True)

        return self

    def with_rows(self, rows):
        extra_fields = {f: getattr(self, f) for f in self.fields}

        header_copy = [column.copy() for column in self.header]

        return Table(self.id, header_copy, data=rows, **extra_fields)

    def get_column(self, column_name):
        return self.header_index[column_name]

    def __len__(self):
        return len(self.data)

    @property
    def as_row_list(self):
        if len(self) > 0 and isinstance(self.data[0], dict):
            return [
                [
                    row[column.name]
                    for column in self.header
                ]
                for row in self.data
            ]

        return self.data

    def to_data_frame(self, tokenizer=None, detokenize=False):
        row_data = self.as_row_list
        columns = [column.name for column in self.header]

        if tokenizer:
            row_data = [
                [
                    ' '.join(tokenizer.tokenize(str(cell)))
                    for cell in row
                ]
                for row in row_data
            ]

            columns = [' '.join(tokenizer.tokenize(str(column))) for column in columns]
        elif detokenize:
            row_data = [
                [
                    ' '.join(cell).replace(' ##', '')
                    for cell in row
                ]
                for row in row_data
            ]

        df = pd.DataFrame(row_data, columns=columns)

        return df

    def __repr__(self):
        column_names =  ', '.join(col.name for col in self.header)
        return f'Table {self.id} [{column_names} | {len(self)} rows]'

    __str__ = __repr__
