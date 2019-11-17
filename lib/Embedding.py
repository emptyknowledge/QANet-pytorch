# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch
from pytorch_transformers import BertModel, BertTokenizer

class BertEmbedding():
  """"""

  def __init__(self, model_path="./data/model/bert"):
    self.model = BertModel.from_pretrained(model_path)
    self.tokenizer = BertTokenizer.from_pretrained(model_path)

  def encode(self, text):
    """"""
    return self.tokenizer.encode(text)

  def encode2torch(self, text):
    encode_text = self.tokenizer.encode(text)
    return torch.Tensor(encode_text).long().view((1, -1))

  def word_embedding(self, input):
    with torch.no_grad():
      output = self.model(input)
      embedding, _, encoded_layers, _ = output
    return embedding