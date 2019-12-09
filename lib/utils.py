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


def find_max_porper(a_softmax, b_softmax):
  b_temp = []
  for i in range(0, b_softmax.shape[0]):
    b_temp.append(torch.argmax(b_softmax[i:]) + i)

  max_start_index = -1
  max_end_index = -1
  max_pro = -1
  for start_index in range(a_softmax.shape[0]):
    max_start_pro = a_softmax[start_index]
    max_pro_end_index = b_temp[start_index]
    max_end_pro = b_softmax[max_pro_end_index]
    cur_max_pro = max_start_pro * max_end_pro # Todo: 测试下 * 和 torch.mul 的效率
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