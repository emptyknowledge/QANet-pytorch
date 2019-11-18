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
  def __init__(self, data_path="./data/train/train.json", batch_size=1):
    bert_embedding = BertEmbedding()
    data = read_data(data_path)
    self.context_idx = []
    self.question_idx = []
    self.answer_idx = []
    self.batch_size = batch_size
    self.data_szie = len(data)
    for item in data:
      self.context_idx.append(bert_embedding.encode(item.get("context")))
      self.question_idx.append(bert_embedding.encode(item.get("question")))
      answer = [int(v) for v in item.get("answer").split(",")]
      self.answer_idx.append(answer)
    self.context_idx = torch.Tensor(self.context_idx).long()
    self.question_idx = torch.Tensor(self.question_idx).long()
    self.answer_idx = torch.Tensor(self.answer_idx).long()
  
    self.idx = list(range(len(data)))
    random.shuffle(self.idx)

  def __getitem__(self, item):
    index = self.idx[(item * self.batch_size) % self.data_szie: ((item + 1) * self.batch_size) % self.data_szie]
    index = torch.LongTensor(index)
    return (self.context_idx[index], self.question_idx[index], self.answer_idx[index])

  def __len__(self):
    return self.data_szie
    
    


