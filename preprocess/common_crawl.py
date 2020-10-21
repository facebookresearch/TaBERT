# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tarfile
import traceback
from argparse import ArgumentParser

import ujson as json
import multiprocessing
import re
import os
import sys
import string
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm

import spacy
from spacy.lang.en import English

from preprocess.data_utils import split_token_coarse, RE_HTML_TAG
from preprocess.table import *


__DEBUG__ = False


def is_ascii(token):
    try:
        token.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True


def has_invalid_tokens(text):
    INVALID_CONTEXT_TOKENS = ['[', ']', '!', '{', '}', ';', '()', ');', '>', '<', '›']
    return any(
        token in text
        for token
        in INVALID_CONTEXT_TOKENS
    )


def remove_full_rowspans(rows):
    """Remove rows in which all cells have the same text."""
    return [row for row in rows if len(set(row)) > 1]


def remove_empty_columns(orig_cols):
    """Remove columns with <= 1 non-empty cells."""
    cols = []
    for col in orig_cols:
        non_empty = sum((bool(cell) for cell in col), 0)
        if non_empty >= 2:
            cols.append(col)
    return cols


def remove_invalid_columns(orig_cols):
    cols = []
    for col in orig_cols:
        is_valid_col = True
        num_invalid = 0
        for cell in col:
            cell_tokens = split_token_coarse(cell)
            if len(cell_tokens) > TABLE_CELL_MAX_TOKEN_NUM:
                is_valid_col = False
                break

            if has_invalid_tokens(cell):
                is_valid_col = False
                break

            ascii_token_count = sum(is_ascii(w) for w in cell_tokens)
            non_ascii_char_count = sum(ord(c) >= 128 and c not in ALLOWED_SPECIAL_SYMBOLS for c in cell)
            non_ascii_token_count = len(cell_tokens) - ascii_token_count
            # digit_num = sum(is_digit(w) for w in cell_tokens)
            # is_all_digits = digit_num == len(cell_tokens)
            if len(cell_tokens) > 0 and ascii_token_count == 0 or non_ascii_token_count > ascii_token_count or non_ascii_char_count >= 2:
                if __DEBUG__:
                    print('invalid cell for ascii rule: ', cell)
                is_valid_col = False
                break

            # non_alpha_token_count = sum(not w.isalpha() for w in cell_tokens)
            # if non_alpha_token_count > TABLE_CELL_MAX_NON_ALPHA_TOKEN_NUM:
            #     num_invalid += 1

        is_valid_col = is_valid_col and num_invalid / len(col) < 0.4
        if is_valid_col:
            cols.append(col)

    return cols


def are_mergeable(col1, col2):
    assert len(col1) == len(col2)
    merged = []
    for i in range(len(col1)):
        c1, c2 = col1[i], col2[i]
        if not c1:
            merged.append(c2)
        elif not c2 or c1 == c2:
            merged.append(c1)
        else:
            return None
    return merged


def merge_similar_columns(orig_cols):
    """Merge similar columns."""
    i = 0
    while i + 1 < len(orig_cols):
        merged = are_mergeable(orig_cols[i], orig_cols[i+1])
        if merged is not None:
            orig_cols[i:i+2] = [merged]
        else:
            i += 1
    return orig_cols


def is_valid_column_name(header_name):
    if has_invalid_tokens(header_name):
        return False

    header_tokens = split_token_coarse(header_name)
    token_num = len(header_tokens)
    if token_num == 0 or token_num > TABLE_HEADER_MAX_TOKEN_NUM:
        return False

    alpha_token_num = sum(w.isalpha() for w in header_tokens)
    if alpha_token_num < TABLE_HEADER_MIN_ALPHA_WORD_NUM:
        return False

    non_alpha_token_num = sum(not w.isalpha() for w in header_tokens)
    if non_alpha_token_num > TABLE_HEADER_MAX_NON_ALPHA_WORD_NUM:
        # print('invalid header for non-alpha tokens: ',header_name)
        return False

    return True


def remove_column_with_invalid_header(orig_cols, header_names):
    cols = []
    for col_id, header in enumerate(header_names):
        if is_valid_column_name(header):
            cols.append(orig_cols[col_id])

    return cols


def remove_full_rowspans(rows):
    """Remove rows in which all cells have the same text."""
    return [row for row in rows if len(set(row)) > 1]


def transpose(data):
    col_num = len(data)

    try:
        row_num = len(data[0])
    except:
        pass
    data_t = []
    for row_id in range(row_num):
        row = []
        for col_id in range(col_num):
            row.append(data[col_id][row_id])
        data_t.append(row)

    return data_t


ALLOWED_SPECIAL_SYMBOLS = {'£', '°', '§', '€'}


class ContextProcessor(object):
    def __init__(self):
        nlp = English()  # just the language with no table_bert
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)

        self.nlp = nlp

    def clean_context(self, text):
        text = RE_HTML_TAG.sub('', text)
        text = text.replace('“', '\"').replace("”", '\"').replace('’', "'").replace('—', '-').replace('•', '')
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return []

        if len(text.split()) == 1:
            return [text]

        text = self.nlp(text)
        valid_sents = []
        for sent in text.sents:
            if has_invalid_tokens(sent.text):
                continue

            non_ascii_char_count = sum(ord(c) >= 128 and c not in ALLOWED_SPECIAL_SYMBOLS for c in sent.text)
            if non_ascii_char_count >= 2:
                if __DEBUG__:
                    print('Invalid sentence: ', sent)
                continue

            num_alpha = sum(w.is_ascii and w.is_alpha for w in sent)
            if num_alpha == 0:
                continue

            num_non_alpha = len(sent) - num_alpha  # sum(not w.is_alpha for w in sent)
            if num_non_alpha >= num_alpha:
                continue

            valid_sents.append(sent.text.strip())

        return valid_sents


def process_example(example_dict: Dict, context_processor: ContextProcessor, nlp_model: English) -> Optional[Dict]:
    if example_dict['tableType'] != 'RELATION' or example_dict['hasHeader'] is False:
        return None

    # detect header information
    header_row_idx = example_dict['headerRowIndex']
    if header_row_idx < 0:
        return None

    data = example_dict['relation']
    if example_dict['tableOrientation'] == 'VERTICAL':
        cols = transpose(data)
    else:
        cols = data

    if len(cols) > TABLE_MAX_COL_NUM or cols and len(cols[0]) > TABLE_MAX_ROW_NUM:
        return None

    header_names = [cols[i][header_row_idx] for i in range(len(cols))]
    # remove full span for header
    if len(set(header_names)) == 1:
        return None

    # for efficiency issue, we first check NL context
    context_before = example_dict['textBeforeTable']
    context_after = example_dict['textAfterTable']

    context_before = context_processor.clean_context(context_before)
    context_after = context_processor.clean_context(context_after)

    if not context_before and not context_after:
        return None

    cols = remove_column_with_invalid_header(cols, header_names)
    cols = remove_empty_columns(cols)
    cols = remove_invalid_columns(cols)
    cols = merge_similar_columns(cols)

    if len(cols) == 0:
        if __DEBUG__:
            print('***** Empty Table After Cleaning ******')
            print(data)
        return None

    data = transpose(cols)
    header = data[header_row_idx]
    content = [row for i, row in enumerate(data) if i != header_row_idx]

    content = remove_full_rowspans(content)
    if len(content) == 0:
        return None

    col_num = len(content[0])
    row_num = len(content)

    if row_num < TABLE_MIN_ROW_NUM or col_num < TABLE_MIN_COL_NUM:
        return None

    table = Table.parse(header, content,
                        nlp_model=nlp_model,
                        caption=example_dict['title'])

    uuid = '{}_{}_{}'.format(example_dict['s3Link'],
                             example_dict['recordOffset'], example_dict['recordEndOffset'])

    return {'uuid': uuid,
            'table': table.to_dict(),
            'context_before': context_before,
            'context_after': context_after}


class CommonCrawlTableExtractor(multiprocessing.Process):
    def __init__(self, job_queue: multiprocessing.Queue,
                 example_queue: multiprocessing.Queue,
                 **kwargs):
        super(CommonCrawlTableExtractor, self).__init__(**kwargs)

        self.job_queue = job_queue
        self.example_queue = example_queue

    def run(self):
        self.nlp = spacy.load('en_core_web_sm')
        print(f'[Process {os.getpid()}] loaded NLP model')

        self.context_processor = ContextProcessor()

        job = self.job_queue.get()
        while job is not None:
            example_dict = json.loads(job)

            try:
                example = process_example(example_dict, self.context_processor, self.nlp)
                if example:
                    self.example_queue.put(example)
            except:
                typ, value, tb = sys.exc_info()
                print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)
                print('*' * 10 + 'Payload' + '*' * 10, file=sys.stderr)
                print(job, file=sys.stderr)
                print('*' * 10 + 'Stack Trace' + '*' * 10, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)

            job = self.job_queue.get()


def data_loader_process(input_file: Path,
                        file_filter: str,
                        job_queue: multiprocessing.Queue,
                        num_workers: int):
    if input_file.is_dir():
        files = list(input_file.glob(file_filter))
        print('Working on {}'.format([f.name for f in files]), file=sys.stderr)
        sys.stderr.flush()
    else:
        files = [input_file]

    pbar = tqdm(file=sys.stdout)
    for file in files:
        print(f'parsing {file}', file=sys.stderr)
        sys.stderr.flush()

        if file.name.endswith('.tar.gz'):
            with tarfile.open(str(file), "r:gz") as outer_tar:
                for tar_m in outer_tar.getmembers():
                    print(tar_m)
                    f_tar = outer_tar.extractfile(tar_m)
                    with tarfile.open(fileobj=f_tar, mode='r') as tar:
                        for m in tar.getmembers():
                            f = tar.extractfile(m)
                            if f is not None:
                                content = f.read()
                                job_queue.put(content)
                                pbar.update(1)
                                f.close()
        else:
            for line in file.open():
                job_queue.put(line)
                pbar.update(1)

    pbar.close()

    for i in range(num_workers):
        job_queue.put(None)


def example_writer_process(output_file, example_queue):
    data = example_queue.get()
    with output_file.open('w') as f:
        while data is not None:
            d = json.dumps(data, ensure_ascii=False)
            f.write(d + os.linesep)

            data = example_queue.get()


def process():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=Path, required=True)
    parser.add_argument('--filter', type=str, default='*.tar.gz', required=False)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument('--worker_num', type=int, default=multiprocessing.cpu_count() - 1, required=False)

    args = parser.parse_args()

    job_queue = multiprocessing.Queue(maxsize=2000)
    example_queue = multiprocessing.Queue()
    num_workers = args.worker_num

    loader = multiprocessing.Process(target=data_loader_process, daemon=True,
                                     args=(args.input_file, args.filter, job_queue, num_workers))
    loader.start()

    workers = []
    for i in range(num_workers):
        worker = CommonCrawlTableExtractor(job_queue, example_queue, daemon=True)
        worker.start()
        workers.append(worker)

    writer = multiprocessing.Process(target=example_writer_process, daemon=True, args=(args.output_file, example_queue))
    writer.start()

    for worker in workers:
        worker.join()
    loader.join()

    example_queue.put(None)
    writer.join()


def debug():
    example_dict = """
    {"relation":[["","Annual/Final Progress Report Format","Quarterly Progress Report Format","Electronic Report Submission","FY13/14 Quad Chart","Technical Reporting Requirements"],["Word","","","n/a","","n/a"],["PDF","n/a","n/a","n/a","n/a","n/a"]],"pageTitle":"eBRAP Online Application Submission","title":"","url":"https://ebrap.org/eBRAP/public/ProgramFY.htm?programFYId\\u003d12902","hasHeader":true,"headerPosition":"FIRST_ROW","tableType":"RELATION","tableNum":3,"s3Link":"common-crawl/crawl-preprocess/CC-MAIN-2015-32/segments/1438042988061.16/warc/CC-MAIN-20150728002308-00026-ip-10-236-191-2.ec2.internal.warc.gz","recordEndOffset":852028500,"recordOffset":852020298,"tableOrientation":"HORIZONTAL","textBeforeTable":"\xc2\xa0 FY14 General Application Instructions \xc2\xa0 Synopsis of PCRP Award Mechanisms Prostate Cancer Research Program (PCRP) Funding Opportunities and Forms Funding Opportunities \\u0026 Forms Register Login Electronic Biomedical Research Application Portal Serving USAMRMC, USAMRAA, CDMRP, DHP \xc2\xa0 Privacy Statement FAQ Guide Help Desk Home","textAfterTable":"n/a FY15 IND/IDE Documentation Form n/a Pre-application Budget Summary Form n/a Collaborating DoD Military Facility Budget Form n/a Common Blinding Mistakes and How to Avoid Them SOW (Statement of Work) Generic Format n/a SOW for Basic Research (Training Section optional) n/a SOW for Clinical Research (Including Trials, Special Populations) n/a SOW for Advanced Tech Development Research n/a SOW for Collaborative PI projects n/a Regulatory Document Forms \xc2\xa0 Word PDF Safety \\u0026 Environmental Resources n/a n/a Environmental Compliance Assurance Animal","hasKeyColumn":true,"keyColumnIndex":0,"headerRowIndex":0}
    """

    worker = CommonCrawlTableExtractor(None, None, daemon=True)
    example = process_example(json.loads(example_dict), ContextProcessor(), spacy.load('en_core_web_sm'))


if __name__ == '__main__':
    process()
    # debug()

