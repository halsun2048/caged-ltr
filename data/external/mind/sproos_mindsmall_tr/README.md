---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: query
    dtype: string
  - name: positive
    dtype: string
  - name: negative
    dtype: string
  splits:
  - name: train
    num_bytes: 7251401
    num_examples: 1419
  download_size: 4248957
  dataset_size: 7251401
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# Dataset Card for "mindsmall-tr"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)