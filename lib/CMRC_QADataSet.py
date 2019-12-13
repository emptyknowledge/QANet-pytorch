# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import json
import random
import torch

from lib import config
from lib.Embedding import BertEmbedding
from torch.utils.data import Dataset

def read_data(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

class CMRC_QADataSet(Dataset):
  def __init__(self, data_path="./data/train/cmrc2018/train.json",
               batch_size=1, context_len=512, question_len=20):
    self.data_path = data_path
    self.bert_embedding = BertEmbedding(config.bert_path)
    data = read_data(data_path)
    self.context_len = context_len
    self.question_len = question_len
    self.context_idx = []
    self.question_idx = []
    self.answer_idx = []
    self.ids = []
    self.batch_size = batch_size
    self.data_szie = 0
    for ids, item in enumerate(data):
      if not self.check_data(item):
        continue
      self.context_idx.append(self.padarr(self.bert_embedding.encode(item.get("context")),
                                          self.context_len))
      self.question_idx.append(self.padarr(self.bert_embedding.encode(item.get("question")),
                                           self.question_len))
      self.ids.append(ids)
      # answer = [int(v) for v in item.get("answer").split(",")]
      answer = item.get("answer_index")
      self.answer_idx.append(answer)
      self.data_szie += 1
    self.context_idx = torch.Tensor(self.context_idx).long()
    self.question_idx = torch.Tensor(self.question_idx).long()
    self.answer_idx = torch.Tensor(self.answer_idx).long()
    self.ids = torch.Tensor(self.ids).long()
  
    self.idx = list(range(self.data_szie))
    if self.batch_size > self.data_szie:
      self.idx = self.idx * (self.batch_size//self.data_szie + 1)
    random.shuffle(self.idx)

  def check_data(self, item):
    """"""
    answer = item.get("answer_index")
    if answer[0] >511 or answer[1] >511:
      return False
    else:
      return True

  def __getitem__(self, item):
    # index = self.idx[(item * self.batch_size) % self.data_szie: ((item + 1) * self.batch_size) % self.data_szie]
    if self.batch_size > len(self.idx):
      index = self.idx[(item * self.batch_size) % self.data_szie]
    else:
      start_idx = (item * self.batch_size) % self.data_szie
      end_idx = ((item + 1) * self.batch_size) % self.data_szie
      if start_idx < end_idx:
        index = self.idx[start_idx:end_idx]
      else:
        index = self.idx[start_idx:] + self.idx[:end_idx]
    index = torch.tensor(index).long()
    return (self.context_idx[index],
            self.question_idx[index],
            self.answer_idx[index],
            self.ids[index])

  def __len__(self):
    return self.data_szie

  def padarr(self, val, v_len):
    res = [0] * v_len
    res[0:min(len(val), v_len)] = val[0:min(len(val), v_len)]
    return res

  def idx2text(self, idx):
    return self.bert_embedding.decode2text(idx)

  def idx2multi_text(self, idx):
    return self.bert_embedding.decode2multi_text(idx)

    
  def get_origin_data(self, ids, key="context"):
    """"""
    # result = []
    data = read_data(self.data_path)
    return data[ids].get(key)
    # for i in ids:
    #   result.append(data[i].get(key))
    # return result


