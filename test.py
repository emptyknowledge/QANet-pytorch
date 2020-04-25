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
from lib.utils import load_class, gelu
import math
def test():
  print(torch.randn(3,3))

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

def test_init_normal():
  def linear(x,w,b): return x @ w + b
  def relu(x): return x.clamp_min(0.)

  x_train = torch.randn(784)
  # x_train = torch.nn.init.normal(x_train)
  print(x_train.mean(), x_train.std())
  nh = 50
  w1 = torch.randn(784, nh)
  b1 = torch.zeros(nh)
  z1 = linear(x_train, w1, b1)
  print("norma init=====================>")
  print(z1.mean(), z1.std())

def test_init_xavier():

  print("Xavier init ====================>")
  def linear(x,w,b): return x @ w + b
  def relu(x): return x.clamp_min(0.)

  x_train = torch.randn(784)
  # x_train = torch.nn.init.normal_(x_train)
  print(x_train.mean(), x_train.std())
  nh = 50
  w1 = torch.randn(784, nh) * math.sqrt(1/768)
  b1 = torch.zeros(nh)
  z1 = linear(x_train, w1, b1)
  scale = 1/2
  # print(0, z1.mean(), z1.std())
  z2 = relu(z1)
  print(0, z2.mean(), z2.std())
  # z2 = gelu(z1)
  # print(1, z2.mean(), z2.std())
  # for i in range(20):
  #   new_chan = max(int(nh * scale), 1)
  #   w2 = torch.randn(nh, new_chan) * math.sqrt(1/nh)
  #   b2 = torch.zeros(new_chan)
  #   z2 = linear(z1, w2, b2)
  #   nh = new_chan
  #   z1 = z2
  #   print(i+1, z2.mean(), z2.std())


def test_init_kaiming():
  print("kaiming init ==============================>")
  def linear(x,w,b): return x @ w + b
  def relu(x): return x.clamp_min(0.)

  x_train = torch.randn(784)
  # x_train = torch.nn.init.normal_(x_train)
  print(x_train.mean(), x_train.std())
  nh = 50
  w1 = torch.randn(784, nh) * math.sqrt(2 / 768)
  b1 = torch.zeros(nh)
  z1 = linear(x_train, w1, b1)
  z2 = relu(z1)
  print(z2.mean(), z2.std())
  z2 = gelu(z1)
  print(z2.mean(), z2.std())

def test_torch_buildin():
  print("torch init ===========>")
  import torch.nn.init as init
  def linear(x,w,b): return x @ w + b
  def relu(x): return x.clamp_min(0.)
  t_linear = torch.nn.Linear(784, 50)

  nh = 50
  x_train = torch.randn(784)
  W1 = torch.zeros(784, nh)
  b1 = torch.zeros(nh)
  W2 = torch.zeros(784, nh)
  b2 = torch.zeros(1)

  init.kaiming_normal_(W1, mode='fan_out', nonlinearity='relu')
  init.kaiming_normal_(W2, mode='fan_out', nonlinearity='relu')
  # init.kaiming_normal_(W2, mode='fan_out')
  z1 = t_linear(x_train)
  z2 = linear(x_train, W1, b1)
  z3 = linear(x_train, W2, b1)
  # a1 = torch.relu(z1)
  a1 = relu(z1)
  a2 = relu(z2)
  a3 = relu(z3)
  print("a1 layer1: ", a1.mean(), a1.std())
  print("a2 layer1: ", a2.mean(), a2.std())
  print("a3 layer1: ", a3.mean(), a3.std())

def test_save_features():
  from lib.handler import get_dataset
  from my_py_toolkit.file.file_toolkit import writejson
  import lib.config as cf
  dataset = get_dataset("train", cf.mode)
  writejson(dataset.convert_all_features4human_visual(), cf.path_save_feature)

def test_np():
  import numpy as np
  print(np.asarray((1,2)))

def test_loss():
  import torch.nn.functional as F
  a = torch.zeros(512)
  a[0] = 1
  print(a)
  print(F.nll_loss(1,2))

if __name__ == "__main__":
  # test_save_features()
  test_loss()
  # test_init_normal()
  # test_init_xavier()
  # test_init_kaiming()
  # test_torch_buildin()
