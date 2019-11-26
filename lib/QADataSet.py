# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import json
import random
import torch

from lib.Embedding import BertEmbedding
from torch.utils.data import Dataset

def read_data(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

class QADataSet(Dataset):
  def __init__(self, data_path="./data/train/train.json",
               batch_size=1, context_len=512, question_len=20):
    self.bert_embedding = BertEmbedding()
    data = read_data(data_path)
    self.context_len = context_len
    self.question_len = question_len
    self.context_idx = []
    self.question_idx = []
    self.answer_idx = []
    self.batch_size = batch_size
    self.data_szie = len(data)
    for item in data:
      self.context_idx.append(self.padarr(self.bert_embedding.encode(item.get("context")),
                                          self.context_len))
      self.question_idx.append(self.padarr(self.bert_embedding.encode(item.get("question")),
                                           self.question_len))
      answer = [int(v) for v in item.get("answer").split(",")]
      self.answer_idx.append(answer)
    self.context_idx = torch.Tensor(self.context_idx).long()
    self.question_idx = torch.Tensor(self.question_idx).long()
    self.answer_idx = torch.Tensor(self.answer_idx).long()
  
    self.idx = list(range(len(data)))
    if self.batch_size > self.data_szie:
      self.idx = self.idx * (self.batch_size//self.data_szie + 1)
    random.shuffle(self.idx)

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
            self.answer_idx[index])

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
    


