# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import os
import torch
from lib import config
from pytorch_transformers import BertModel, BertTokenizer

class BertEmbedding():
  """"""

  def __init__(self, model_path="./data/model/bert"):
    self.model = BertModel.from_pretrained(model_path).to(config.device)
    self.tokenizer = BertTokenizer.from_pretrained(model_path)
    # self.trainable_embedding = trainable_embedding

  def encode(self, text):
    """"""
    return self.tokenizer.encode(text)

  def encode2torch(self, text):
    # print(f"Embedding text: {text}")
    encode_text = self.tokenizer.encode(text)
    return torch.Tensor(encode_text).long().view((1, -1))

  def word_embedding(self, ids):
    """
    返回 BERT 等预训练模型的 pretrained_embedding
    Args:
      ids:

    Returns:

    """
    with torch.no_grad():
      output = self.model(ids)
      embedding, _ = output
    return embedding

  # def word_embedding_trainable(self, ids):
  #   """
  #   返回可训练 embedding
  #   Args:
  #     ids:
  #
  #   Returns:
  #
  #   """
  #   return self.trainable_embedding[ids].to(config.device)

  def decode2text(self, idx):
    idx = idx.tolist()
    if not idx:
      return ""

    token = self.tokenizer.convert_ids_to_tokens(idx)
    return "".join(token)

  def decode2multi_text(self, idxs):
    idxs = idxs.tolist()
    if not idxs:
      return ""
    result = []
    for val in idxs:
      result.append(self.decode2text(val))
    return result



  @property
  def vocab_size(self):
    return self.tokenizer.vocab_size