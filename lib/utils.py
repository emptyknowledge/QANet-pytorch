# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
#
#
# cython: language_level=3
#

import os
import torch
from pytorch_transformers.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

def write2file(content, file_path, is_continue=True):
  """"""
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))
  mode = "w+" if is_continue else "w"
  with open(file_path, mode, encoding="utf-8")as f:
    f.write("\n" + content)

def find_max_proper_batch(start_softmax, end_softmax):
  """

  Args:
    start_softmax(batch_size, dim):
    end_softmax(batch_size, dim):

  Returns:

  """
  start_index = []
  end_index = []
  max_pro = []
  for index in range(start_softmax.shape[0]):
    start, end, pro = find_max_porper(start_softmax[index], end_softmax[index])
    start_index.append(start)
    end_index.append(end)
    max_pro.append(pro)

  return start_index, end_index, max_pro



def find_max_porper(start_index_softmax, end_index_softmax):
  """
  根据 start index 与 end index 的 softmax 找出最大的联合概率: max(start_index_pro * end_index_pro)
  Args:
    start_index_softmax(dim):
    end_index_softmax(dim):

  Returns:

  """
  # b_max_pro_index[index] mean: b_softmax[index:] 中可能性最大位置的下标.
  b_max_pro_index = [-1] * end_index_softmax.shape[0]
  b_max_pro = -1
  b_max_index = -1
  for index in range(end_index_softmax.shape[0] - 1, -1, -1): # 0.1 左右
    if end_index_softmax[index] > b_max_pro:
      b_max_pro = end_index_softmax[index]
      b_max_index = index
    b_max_pro_index[index] = b_max_index


  max_start_index = -1
  max_end_index = -1
  max_pro = -1
  for start_index in range(start_index_softmax.shape[0]):
    max_start_pro = start_index_softmax[start_index]
    max_pro_end_index = b_max_pro_index[start_index]
    max_end_pro = end_index_softmax[max_pro_end_index]
    #  * 和 torch.mul 执行时间差不多
    cur_max_pro = torch.mul(max_start_pro, max_end_pro)
    if cur_max_pro > max_pro:
      max_pro = cur_max_pro
      max_start_index = start_index
      max_end_index = max_pro_end_index

  return max_start_index, max_end_index, max_pro

def tf2torch():
  """
  convert_tf_checkpoint_to_pytorch(r"\model\chinese_L-12_H-768_A-12\bert_model.ckpt",
                                   r"\model\chinese_L-12_H-768_A-12\bert_config.json",
                                   r"\model\chinese_L-12_H-768_A-12\bert_model.bin")
  Returns:

  """


def test():
  path = "./test.txt"
  write2file("11", path)

if __name__ == "__main__":
  test()