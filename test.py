# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#
import lib.config as cf
import torch
import torch.nn.functional as f
from pytorch_transformers import *
from lib.utils import load_class
def test():
  a = torch.Tensor([1]).float()
  b = torch.Tensor([2])
  # b = b.permute()
  print(torch.cat([a,b], dim=0))

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


def test_class():
  class A:
    def __init__(self):
      self.a=1
      self.b = 2

    def __getitem__(self, item):
      return item

    # def test(self):
    #   co = self.__init__.func_code
    #   print(co.co_name)


  # print(dir(A()))
  a = A()
  print(a(2,2))
  # print(A().test())

def te_soft_max():
  import torch.nn as nn

  m = nn.Softmax(dim=0)
  n = nn.Softmax(dim=1)
  k = nn.Softmax(dim=2)
  j = nn.Softmax(dim=-1)
  l = nn.Softmax(dim=-2)
  input = torch.randn(2, 2, 3)
  print(input)
  print(m(input))
  print(n(input))
  print(k(input))
  print(j(input))
  print(l(input))

def test_mask():
  a = torch.Tensor(3,3,3)
  mask = torch.Tensor([1,1,0])
  mask = mask.unsqueeze(-1)
  mask = mask.unsqueeze(-1)
  print(mask)
  print(mask.shape)
  print(a.shape)
  print(torch.mul(a, mask))



if __name__ == "__main__":
  test()
