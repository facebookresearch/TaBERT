# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from preprocess import data_utils
from table_bert.table import Column

TABLE_MIN_ROW_NUM = 3
TABLE_MIN_COL_NUM = 2
TABLE_MAX_COL_NUM = 15
TABLE_MAX_ROW_NUM = 30


TABLE_HEADER_MAX_TOKEN_NUM = 10
TABLE_HEADER_MIN_ALPHA_WORD_NUM = 1
TABLE_HEADER_MAX_NON_ALPHA_WORD_NUM = 5


TABLE_CELL_MAX_TOKEN_NUM = 20
TABLE_CELL_MAX_NON_ALPHA_TOKEN_NUM = 5


class Table(object):
    def __init__(self, header: List[Column], rows: List, caption=None):
        self.header = header
        self.rows = rows
        self.caption = caption

    def to_dict(self):
        return {
            'caption': self.caption,
            'header': [
                column.to_dict()
                for column
                in self.header
            ],
            'data': self.rows
        }

    @classmethod
    def parse(cls, header, content, nlp_model, caption=None) -> 'Table':
        columns = []
        sampled_values = []
        for col_ids, col_name in enumerate(header):
            sample_value = None
            for row in content:
                cell_val = row[col_ids]
                if len(cell_val.strip()) > 0:
                    sample_value = cell_val
                    break

            assert sample_value is not None
            sampled_values.append(sample_value)

        parsed_values = nlp_model.pipe(sampled_values)
        for col_id, sampled_value_annot in enumerate(parsed_values):
            tokenized_value = [token.text for token in sampled_value_annot]
            ner_tags = [token.ent_type_ for token in sampled_value_annot]
            pos_tags = [token.pos_ for token in sampled_value_annot]

            sample_value_entry ={
                'value': sampled_value_annot.text,
                'tokens': tokenized_value,
                'ner_tags': ner_tags
            }

            col_name = header[col_id]
            col_type = data_utils.infer_column_type_from_sampled_value(sample_value_entry)

            columns.append(Column(col_name, col_type, sample_value=sample_value_entry))

        return cls(columns, content, caption=caption)
