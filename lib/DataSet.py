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

def read_data(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

class DataSet(object):
  def __init__(self, data_path="./train/train.json", batch_size=1):
    bert_embedding = BertEmbedding()
    data = read_data(data_path)
    self.context_idx = []
    self.question_idx = []
    self.answer_idx = []
    self.batch_size = batch_size
    self.data_szie = len(data)
    for item in data:
      self.context_idx.append(bert_embedding.encode2torch(item.get("context")))
      self.question_idx.append(bert_embedding.encode2torch(item.get("question")))
      answer = torch.Tensor(item.get("answer").split(",")).long().view(1. -1)
      self.answer_idx.append(answer)

    self.idx = random.shuffle(list(range(len(data))))

  def __getitem__(self, item):
    index = self.idx[(item * self.batch_size) % self.data_szie: ((item + 1) * self.batch_size) % self.data_szie]
    index = torch.LongTensor(index)
    return (self.context_idx[index], self.question_idx[index], self.answer_idx[index])



