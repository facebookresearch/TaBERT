# TaBERT: Learning Contextual Representations for Natural Language Utterances and Structured Tables

This repository contains source code for the [`TaBERT` model](https://arxiv.org/abs/2005.08314), a pre-trained language model for learning joint representations of natural language utterances and (semi-)structured tables for semantic parsing. `TaBERT` is pre-trained on a massive corpus of 26M Web tables and their associated natural language context, and could be used as a drop-in replacement of a semantic parsers original encoder to compute representations for utterances and table schemas (columns).

## Installation

First, install the conda environment `tabert` with supporting libraries.

```bash
bash scripts/setup_env.sh
```

Once the conda environment is created, install `TaBERT` using the following command:

```bash
conda activate tabert
pip install --editable .
```

**Integration with HuggingFace's pytorch-transformers Library** is still WIP. While all the pre-trained models were developed based on the old version of the library `pytorch-pretrained-bert`, they are compatible with the the latest version `transformers`. The conda environment will install both versions of the transformers library, and `TaBERT` will use `pytorch-pretrained-bert` by default. You could uninstall the `pytorch-pretrained-bert` library if you prefer using `TaBERT` with the latest version of `transformers`.

## Pre-trained Models

Pre-trained models could be downloaded from this [Google Drive shared folder](https://drive.google.com/drive/folders/1fDW9rLssgDAv19OMcFGgFJ5iyd9p7flg?usp=sharing).
Please uncompress the tarball files before usage.

Pre-trained models could be downloaded from command line as follows:
```shell script
pip install gdown

# TaBERT_Base_(k=1)
gdown 'https://drive.google.com/uc?id=1-pdtksj9RzC4yEqdrJQaZu4-dIEXZbM9'

# TaBERT_Base_(K=3)
gdown 'https://drive.google.com/uc?id=1NPxbGhwJF1uU9EC18YFsEZYE-IQR7ZLj'

# TaBERT_Large_(k=1)
gdown 'https://drive.google.com/uc?id=1eLJFUWnrJRo6QpROYWKXlbSOjRDDZ3yZ'

# TaBERT_Large_(K=3)
gdown 'https://drive.google.com/uc?id=17NTNIqxqYexAzaH_TgEfK42-KmjIRC-g'
```

## Using a Pre-trained Model

To load a pre-trained model from a checkpoint file:

```python
from table_bert import TableBertModel

model = TableBertModel.from_pretrained(
    'path/to/pretrained/model/checkpoint.bin',
)
```

To produce representations of natural language text and and its associated table:
```python
from table_bert import Table, Column

table = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text', sample_value='United States'),
        Column('Gross Domestic Product', 'real', sample_value='21,439,453')
    ],
    data=[
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165'],
    ]
).tokenize(model.tokenizer)

# To visualize table in an IPython notebook:
# display(table.to_data_frame(), detokenize=True)

context = 'show me countries ranked by GDP'

# model takes batched, tokenized inputs
context_encoding, column_encoding, info_dict = model.encode(
    contexts=[model.tokenizer.tokenize(context)],
    tables=[table]
)
```

For the returned tuple, `context_encoding` and `column_encoding` are PyTorch tensors 
representing utterances and table columns, respectively. `info_dict` contains useful 
meta information (e.g., context/table masks, the original input tensors to BERT) for 
downstream application.

```python
context_encoding.shape
>>> torch.Size([1, 7, 768])

column_encoding.shape
>>> torch.Size([1, 2, 768])
```

**Use Vanilla BERT** To initialize a TaBERT model from the parameters of BERT:

```python
from table_bert import TableBertModel

model = TableBertModel.from_pretrained('bert-base-uncased')
```

## Example Applications

TaBERT could be used as a general-purpose representation learning layer for semantic parsing tasks over database tables. Example applications could be found under the `examples` folder.

## Reference

If you plan to use `TaBERT` in your project, please consider citing [our paper](https://arxiv.org/abs/2005.08314):
```
@inproceedings{yin20acl,
    title = {Ta{BERT}: Pretraining for Joint Understanding of Textual and Tabular Data},
    author = {Pengcheng Yin and Graham Neubig and Wen-tau Yih and Sebastian Riedel},
    booktitle = {Annual Conference of the Association for Computational Linguistics (ACL)},
    month = {July},
    year = {2020}
}
```

## License

TaBERT is CC-BY-NC 4.0 licensed as of now.
