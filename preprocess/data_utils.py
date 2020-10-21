# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import string
import unicodedata
from typing import Dict

import spacy
from spacy.lang.en import English


__NON_CHAR_RE = re.compile(rf'[{string.punctuation}]|\w+')
def split_token_coarse(text):
    tokens = __NON_CHAR_RE.findall(text)
    return tokens


RE_HTML_TAG = re.compile('<.*?>.*?</.*?>')


def clean_cell_value(cell_val):
    val = unicodedata.normalize('NFKD', cell_val)
    val = val.encode('ascii', errors='ignore')
    val = str(val, encoding='ascii')

    val = clean_wiki_template(val)
    val = re.sub(r'\s+', ' ', val).strip()

    return val


def clean_wiki_template(text):
    if re.match(r'^{{.*}}$', text):
        text = text[2:-2].split('|')[-1]  # remove {{ }}, and retain the last
    else:
        text = re.sub(r'{{.*}}', '', text)

    return text


if __name__ == '__main__':
    print(clean_cell_value('sdf sd 我们是sdaf{{xxx}}'))


NUMERICAL_NER_TAGS = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL'}
def infer_column_type_from_sampled_value(sample_value_entry: Dict) -> str:
    if not 'ner_tags' in sample_value_entry:
        return 'text'

    num_numerical_tags = 0
    num_other_tags = 0

    for tag in sample_value_entry['ner_tags']:
        if tag in NUMERICAL_NER_TAGS:
            num_numerical_tags += 1
        else:
            num_other_tags += 1

    return 'real' if num_numerical_tags >= num_other_tags else 'text'
