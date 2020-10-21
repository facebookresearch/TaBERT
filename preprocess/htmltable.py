#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Adapted from WikiTableQuestions dataset release (https://github.com/ppasupat/WikiTableQuestions/blob/master/table-to-csv.py)
#  Original Authors: Panupong Pasupat, Percy Liang.
#


"""Table processor.
Get statistics about a table and convert it to CSV.
"""

import sys, os, re, argparse, json
import unicodedata
from codecs import open
from collections import defaultdict
from bs4 import BeautifulSoup as BeautifulSoupOriginal

from preprocess import data_utils
from preprocess.data_utils import clean_cell_value


def BeautifulSoup(markup=""):
    return BeautifulSoupOriginal(markup, 'html.parser')


################ Dump CSV

def simple_normalize_text(text):
    return text.replace('\\', '\\\\').replace('"', r'\"').replace('\n', r'\\n').replace(u'\xa0', ' ').strip()

def dump_csv(rows, fout):
    for row in rows:
        fout.write(','.join('"%s"' % simple_normalize_text(x[1]) for x in row) + '\n')

def tab_normalize_text(text):
    return re.sub(r'\s+', ' ', text.replace('\\', '\\\\').replace('|', r'\p').replace('\n', r'\n'), re.U).strip()

def dump_tsv(rows, fout):
    for row in rows:
        fout.write('\t'.join('%s' % tab_normalize_text(x[1]) for x in row) + '\n')

def table_normalize_text(text):
    return re.sub(r'\s+', ' ', text, re.U).strip()

def dump_table(rows, fout):
    widths = defaultdict(int)
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(table_normalize_text(cell[1])) + 1)
    for row in rows:
        fout.write('|')
        for i, cell in enumerate(row):
            # wow this is so hacky
            fout.write((' %-' + str(widths[i]) + 's') % table_normalize_text(cell[1]))
            fout.write('|')
        fout.write('\n')

################ More table normalization

def debug_print(stuff):
    for x in stuff:
        print >> sys.stderr, [simple_normalize_text(y[1]) for y in x]


def horizontal_to_vertical(rows):
    if rows:
        is_vertical_table = all(cell[0] == 'th' for cell in rows[0])
        if not is_vertical_table:
            is_horizontal_table = all(row[0][0] == 'th' for row in rows)
            if is_horizontal_table and rows:
                rows = transpose(rows)

    return rows

def transpose(rows):
    cols = []
    n = max(len(row) for row in rows)
    for i in range(n):
        col = []
        for row in rows:
            try:
                col.append(row[i])
            except LookupError:
                col.append(('', ''))
        cols.append(col)
    return cols

def anti_transpose(cols):
    # All col in cols must have equal length
    assert len(set(len(col) for col in cols)) == 1
    rows = []
    n = len(cols[0])
    for i in range(n):
        row = []
        for col in cols:
            if col[i] is not None:
                row.append(col[i])
            else:
                row.append(('', ''))
        rows.append(row)
    return rows

def remove_full_rowspans(rows):
    """Remove rows in which all cells have the same text."""
    return [row for row in rows if len(set(row)) > 1]

def remove_empty_columns(orig_cols):
    """Remove columns with <= 1 non-empty cells."""
    cols = []
    for col in orig_cols:
        non_empty = sum((bool(cell[1]) for cell in col), 0)
        if non_empty >= 2:
            cols.append(col)
    return cols

#### Merge columns

def are_mergeable(col1, col2):
    assert len(col1) == len(col2)
    merged = []
    for i in range(len(col1)):
        c1, c2 = col1[i], col2[i]
        if not c1[1]:
            merged.append(c2)
        elif not c2[1] or c1 == c2:
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

def remove_empty_rows(orig_rows):
    """remove empty rows"""
    rows = []
    for row in orig_rows:
        if row:
            if row[0][0] == 'th':
                rows.append(row)
            elif sum(1 for cell in row if cell[1]) >= 2:
                rows.append(row)

    return rows

#### Merge header rows

def merge_header_rows(orig_rows):
    """Merge all header rows together."""
    header_rows, body_rows = [], []
    still_header = True
    for row in orig_rows:
        if not still_header or any(cell[0] == 'td' for cell in row):
            still_header = False
            body_rows.append(row)
        else:
            header_rows.append(row)
    if len(header_rows) < 2 or not body_rows:
        return orig_rows
    # Merge header rows with '\n'
    header_cols = transpose(header_rows)
    header_row = []
    for col in header_cols:
        texts = [None]
        for cell in col:
            if cell[1] != texts[-1]:
                texts.append(cell[1])
        header_row.append(('th', '\n'.join(texts[1:])))
    return [header_row] + body_rows


def remove_columns_by_id(rows, col_id_to_remove):
    if not col_id_to_remove:
        return rows

    new_rows = [[] for _ in rows]
    header = rows[0]
    for col_id in range(len(header)):
        if col_id not in col_id_to_remove:
            [row.append(old_row[col_id]) for row, old_row in zip(new_rows, rows)]

    return new_rows


class HtmlTable(object):
    NORM_NONE = 0
    NORM_CORNER = 1
    NORM_DUPLICATE = 2
    SOUP = BeautifulSoup()

    def __init__(self, table, normalization=0, remove_hidden=True, first_row_as_caption=True):
        """Create table from a BeautifulSoup table Tag."""
        assert table.name == 'table'
        self.table = table
        self.caption = None if table.caption is None else table.caption.text
        if self.caption:
            self.caption = clean_cell_value(self.caption)

        if first_row_as_caption:
            rows = self.table.find_all('tr', recursive=False)
            col_num = max(len(row.find_all(['th', 'td'], recursive=False)) for row in rows)
            if len(rows) >= 2:
                row_1_cells = rows[0].find_all(['th', 'td'], recursive=False)
                if row_1_cells:
                    row_1_cell = row_1_cells[0]
                    if self.get_int(row_1_cell, 'colspan') == col_num:
                        self.caption = clean_cell_value(row_1_cell.text)
                        rows[0].decompose()

        if self.caption:
            self.caption = re.sub(r'(\[\d+\])+$', '', self.caption)

        if remove_hidden:
            self.remove_hidden()
        if normalization == HtmlTable.NORM_CORNER:
            self.normalize_table()
        elif normalization == HtmlTable.NORM_DUPLICATE:
            self.normalize_table(deep=True)

        self.get_cells()

    @staticmethod
    def get_wikitable(raw_html, index=None, **kwargs):
        soup = BeautifulSoup(raw_html)
        tables = soup.find_all('table', class_='wikitable')
        if index is None:
            return [HtmlTable(x, **kwargs) for x in tables]
        else:
            return HtmlTable(tables[index], **kwargs)

    def check_hidden(self, tag):
        classes = tag.get('class', [])
        if 'reference' in classes or 'sortkey' in classes:
            return True
        if 'display:none' in tag.get('style', ''):
            return True
        return False

    def remove_hidden(self):
        """Remove hidden elements."""
        for tag in self.table.find_all(self.check_hidden):
            tag.extract()

    def get_cells(self):
        """Each cell is (tag, text)"""
        self.rows, self.cells = [], []
        for x in self.table.find_all('tr', recursive=False):
            row = []
            for y in x.find_all(['th', 'td'], recursive=False):
                row.append((y.name, clean_cell_value(y.text)))
            self.rows.append(row)
            self.cells.extend(row)

        self.num_cells = len(self.cells)
        self.cols = [[] for i in range(self.num_cols)]
        for row in self.rows:
            for i, cell in enumerate(row):
                self.cols[i].append(cell)

    @property
    def num_rows(self):
        return len(self.rows)

    @property
    def num_cols(self):
        return 0 if not self.num_rows else max(len(row) for row in self.rows)

    ################ Table normalization ################

    def get_int(self, cell, key):
        try:
            return int(cell.get(key, 1))
        except ValueError:
            try:
                return int(re.search('[0-9]+', cell[key]).group())
            except:
                return 1

    def get_cloned_cell(self, cell, rowspan=1, deep=False):
        if deep:
            # Hacky but works
            return BeautifulSoup(str(cell)).contents[0]
        tag = HtmlTable.SOUP.new_tag(cell.name)
        if rowspan > 1:
            tag['rowspan'] = rowspan
        return tag

    def normalize_table(self, deep=False):
        """Fix the table in-place."""
        # Fix colspan
        num_cols = 0
        for tr in self.table.find_all('tr', recursive=False):
            for cell in tr.find_all(['th', 'td'], recursive=False):
                colspan = self.get_int(cell, 'colspan')
                rowspan = self.get_int(cell, 'rowspan')
                if colspan <= 1:
                    continue
                cell['old-colspan'] = cell['colspan']
                del cell['colspan']
                for i in range(2, colspan + 1):
                    cell.insert_after(self.get_cloned_cell(cell, rowspan=rowspan, deep=deep))
            num_cols = max(num_cols, len(tr.find_all(['th', 'td'], recursive=False)))
        # Fix rowspan
        counts = defaultdict(int)
        spanned_cells = dict()
        for tr in self.table.find_all('tr', recursive=False):
            cell = None
            cells = tr.find_all(['th', 'td'], recursive=False)
            k = 0
            for i in range(num_cols):
                if counts[i] > 0:
                    # Create a new element caused by rowspan
                    new_cell = self.get_cloned_cell(spanned_cells[i], deep=deep)
                    if not cell:
                        tr.insert(0, new_cell)
                    else:
                        cell.insert_after(new_cell)
                    cell = new_cell
                    counts[i] -= 1
                else:
                    if k >= len(cells):  # Unfilled row
                        continue
                    cell = cells[k]
                    k += 1
                    rowspan = self.get_int(cell, 'rowspan')
                    if rowspan <= 1:
                        continue
                    counts[i] = rowspan - 1
                    spanned_cells[i] = cell
                    cell['old-rowspan'] = cell['rowspan']
                    del cell['rowspan']

    def clean(self):
        rows = self.rows
        rows = remove_full_rowspans(rows)
        rows = horizontal_to_vertical(rows)
        rows = remove_empty_rows(rows)

        if rows:
            # print('Rows: ')
            # print(rows)
            cols = transpose(rows)
            cols = merge_similar_columns(cols)
            # print('Before removing empty columns:', cols)
            cols = remove_empty_columns(cols)
            # print('After removing empty columns', cols)

            if cols:
                rows = anti_transpose(cols)
                rows = merge_header_rows(rows)

                header_row = rows[0]
                if len(rows) > 1:
                    body_rows = rows[1:]
                    col_id_to_remove = []
                    for col_id in range(len(header_row)):
                        non_empty = sum([bool(row[col_id][1]) for row in body_rows])
                        if non_empty == 0:
                            col_id_to_remove.append(col_id)

                    rows = remove_columns_by_id(rows, col_id_to_remove)

                self.rows = rows
            else:
                self.rows = []
        else:
            self.rows = []

        # delete inconsistent preprocess fields
        del self.cells
        del self.cols

    def to_dict(self):
        return {
            'caption': self.caption,
            'header': self.header,
            'data': self.rows
        }

    def annotate_schema(self, nlp_model=None):
        # assume the first row is table header
        header = []
        content_rows = self.rows[1:]
        for col_ids, (tag, col_name) in enumerate(self.rows[0]):
            sample_value = None
            for row in content_rows:
                cell_val = row[col_ids][1]
                if len(cell_val.strip()) > 0:
                    sample_value = cell_val
                    break

            sample_value_entry = {
                'value': sample_value,
            }

            assert sample_value is not None
            if nlp_model and sample_value:
                annotation = nlp_model(sample_value)
                tokenized_value = [token.text for token in annotation]
                ner_tags = [token.ent_type_ for token in annotation]
                pos_tags = [token.pos_ for token in annotation]

                sample_value_entry.update({
                    'tokens': tokenized_value,
                    'ner_tags': ner_tags,
                    'pos_tags': pos_tags
                })

            col_type = data_utils.infer_column_type_from_sampled_value(sample_value_entry)

            header.append({
                'name': col_name,
                'type': col_type,
                'sample_value': sample_value_entry
            })

        self.header = header

################ Main function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--turk-json',
                        help="json metadata file from MTurk task")
    parser.add_argument('-o', '--outfile',
                        help="output filename (default = stdout)")
    parser.add_argument('--tsv', action='store_true',
                        help='also print out tsv')
    parser.add_argument('--human', action='store_true',
                        help='also print out human-readable table')
    parser.add_argument('--html', action='store_true',
                        help='also print out cleaned html for the table')
    parser.add_argument('--keep-hidden', action='store_true',
                        help='keep hidden texts as is')
    args = parser.parse_args()
    assert not args.tsv or args.outfile.endswith('.csv')

    with open(args.turk_json) as fin:
        metadata = json.load(fin)

    # Get the path to the HTML file
    # This is kind of hacky
    match = re.match(r'^(?:json|page)/(\d+)-(?:json|page)/(\d+).json$', args.turk_json)
    batch_id, data_id = match.groups()
    inhtml = 'page/{}-page/{}.html'.format(batch_id, data_id)

    with open(inhtml, 'r', 'utf8') as fin:
        raw = fin.read()
    table = HtmlTable.get_wikitable(raw, metadata['tableIndex'],
                                    normalization=HtmlTable.NORM_DUPLICATE,
                                    remove_hidden=(not args.keep_hidden))
    if args.html:
        raw_table = HtmlTable.get_wikitable(raw, metadata['tableIndex'],
                                            remove_hidden=False).table

    rows = table.rows
    # rows = list of columns; column = list of cells; cell = (tag, text)
    # Remove redundant rows and columns
    rows = remove_full_rowspans(rows)
    cols = transpose(rows)
    cols = remove_empty_columns(cols)
    cols = merge_similar_columns(cols)
    rows = anti_transpose(cols), ''
    rows = merge_header_rows(rows)
    # Dump
    if not args.outfile:
        dump_csv(rows, sys.stdout)
    else:
        stem = re.sub('\.csv$', '', args.outfile)
        with open(args.outfile, 'w', 'utf8') as fout:
            dump_csv(rows, fout)
        if args.tsv:
            with open(stem + '.tsv', 'w', 'utf8') as fout:
                dump_tsv(rows, fout)
        if args.human:
            with open(stem + '.table', 'w', 'utf8') as fout:
                dump_table(rows, fout)
        if args.html:
            with open(stem + '.html', 'w', 'utf8') as fout:
                print >> fout, unicode(raw_table)


if __name__ == '__main__':
    pass
