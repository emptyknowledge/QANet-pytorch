# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import re
import os
from tqdm import tqdm
from my_py_toolkit.file.file_toolkit import get_file_paths, get_file_name, writejson, readjson
from my_py_toolkit.file.doc.doc_toolkit import get_paragraphs


def process_one_doc(file_path, data_max_len):
  train_data = []
  paragraphs = get_paragraphs(file_path)
  cur_str = ""
  for para in paragraphs:
    # print(f"file_name: {get_file_name(file_path)}, The lenghth of paragraphs: {len(paragraphs)}")
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
    gene_data[file] = data
  
  writejson(gene_data, target_path)

def generate_unlabel_data(origin_data_path,
                          target_path="./generate_data/unlabel_data",
                          file_prefix="unlabel_data",
                          data_num_each_file=100):
  if not os.path.exists(target_path):
    os.makedirs(target_path)
  origin_data = readjson(origin_data_path)
  file_index = 1
  unlabel_data = []
  keys = list(origin_data)
  for index in tqdm(range(0, len(keys)), total=len(keys)):
  # for _, val_list in origin_data.items():
    key = keys[index]
    val_list = origin_data.get(key)
    for val in val_list:
      cur_data = {
        "context": val,
        "question": "",
        "answer": [val]
      }
      unlabel_data.append(cur_data)
      if len(unlabel_data) >= data_num_each_file:
        save_path = os.path.join(target_path,
                                 f"{file_prefix}_{str(file_index)}.json")
        writejson(unlabel_data, save_path)
        unlabel_data = []
        file_index += 1
  if len(unlabel_data)>0:
    save_path = os.path.join(target_path,
                             f"{file_prefix}_{str(file_index)}.json")
    writejson(unlabel_data, save_path)
    unlabel_data = []
    file_index += 1

def run_generate_unlabel_data():
  origin_data_path = "./generate_data/origin_data.json"
  generate_unlabel_data(origin_data_path)


def run_generate_data():
  dir_name = r"D:\Work\招商new\招商所有样本 docx - 1210整理"
  target_path = "./generate_data/origin_data.json"
  data_max_len = 512
  generate_date(dir_name, target_path, data_max_len)

if __name__ == "__main__":
  run_generate_unlabel_data()
  # run_generate_data()