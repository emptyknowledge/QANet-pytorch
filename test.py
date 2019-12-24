# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#
import torch
import torch.nn.functional as f
from pytorch_transformers import *

def test():
  a = torch.Tensor([[1,1],[0,0]])
  a = torch.randn(3,1)
  print(a)
  b = (torch.zeros_like(a) == a).float()
  print(b.shape)
  print(b)
  c = b.unsqueeze(2)
  print(c.shape)
  print(c)
  d = b.expand(1,2, 2)
  print(d.shape)
  print(d)

def test_expand():
  a = torch.randn(2,2)
  print(a.shape)
  print(a)
  b = a.unsqueeze(1).expand(2,3,2)
  # b = a.expand(2,2)
  print(b.shape)
  print(b)

def test_mul():
  a = torch.Tensor([1,1])
  b = torch.Tensor([[2],[2],[3]])
  print(a,b)
  print(torch.mul(a,b))


def test_cat():
  a = torch.randn(3,3,3)
  b = torch.randn(3,3,3)
  c = torch.cat([a,b], dim=2)
  print(c.shape)
  print(c)

def test_para():
  import math
  from torch import nn
  d_model = 3
  w = torch.empty(d_model * 3)
  lim = 1 / d_model
  nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
  ww = nn.Parameter(w)
  print(ww)

def test_bert():
  from pytorch_transformers import BertTokenizer
  from lib.handler import corresponds_index, transfer_index
  tokenizer = BertTokenizer.from_pretrained("./data/model/bert")
  text = "华为nova6系列105°前置超广角双摄了解下"
  token = tokenizer.tokenize(text)
  text = list(text)
  print(text)
  print(token)
  corr_index = corresponds_index(text, token)
  print(corr_index)
  for index, val in enumerate(text):
    index_in_token = transfer_index(text, token, index)[0]
    print(f"index_in_token: {index_in_token}")
    val_in_token = token[index_in_token] if index_in_token>-1 else ""
    print(index, val, index_in_token, val_in_token)
  # print(transfer_index(text, token, 2, 3, 9, 10))

def test():
  a = torch.Tensor(3,3)
  b = torch.Tensor(3,3)
  # a = torch.nn.init.normal_(torch.Tensor(3,3))
  print(torch.cat((a,b), -1))

if __name__ == "__main__":
  test()
