# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import re
import os
from tqdm import tqdm
from my_py_toolkit.file.file_toolkit import get_file_paths, get_file_name, writejson
from my_py_toolkit.file.doc.doc_toolkit import get_paragraphs


def process_one_doc(file_path, data_max_len):
  train_data = []
  paragraphs = get_paragraphs(file_path)
  cur_str = ""
  for para in paragraphs:
    print(f"file_name: {get_file_name(file_path)}, The lenghth of paragraphs: {len(paragraphs)}")
    if len(cur_str) + len(para) < data_max_len - 1:
      cur_str += "\n" + para
    else:
      if not cur_str:
        cur_str = para
      else:
        train_data.append(cur_str + "\n")
        cur_str = para

  if cur_str:
    train_data.append(cur_str)

  return train_data

def generate_date(dir_name, target_path, data_max_len):
  if not os.path.exists(target_path):
    os.makedirs(target_path)
  if not os.path.exists(os.path.split(target_path)[0]):
    os.makedirs(os.path.split(target_path)[0])

  files = get_file_paths(dir_name, ["docx"])
  gene_data = {}
  for index in tqdm(range(0, len(files)), total=len(files)):
  # for i in tqdm(range(start, length + start), total=length):
    file = files[index]
    # print(file)
    if re.search("[~$]", file):
      continue
    file_name = get_file_name(file)
    data = process_one_doc(file, data_max_len)
    gene_data[file_name] = data
  
  writejson(gene_data, target_path)

if __name__ == "__main__":
  dir_name = r"D:\Work\招商new\招商所有样本 docx - 1210整理"
  target_path = "./generate_date/origin_data.json"
  data_max_len = 512
  generate_date(dir_name, target_path, data_max_len)
  pass