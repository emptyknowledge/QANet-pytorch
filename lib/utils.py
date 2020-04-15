# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
#
#
# cython: language_level=3
#

import importlib
import os
import torch
import math
import traceback
from pytorch_transformers import BertModel
from pytorch_transformers.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
from my_py_toolkit.basic_data_type.basic_data_type_toolkit import add_value4dict

def write2file(content, file_path, is_continue=True):
  """"""
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))
  mode = "a" if is_continue else "w"
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

def get_first_non_negative_index(arr, reverse=False):
  """
  获取 arr 中第一个非负值的索引. reverse 控制反向查找或正向查找
  Args:
    arr(list):
    reverse:

  Returns:

  """
  return get_first_index_greater_than_benchmark(arr, -1, reverse)


def get_first_index_greater_than_benchmark(arr, benchmark, reverse=False):
  """
  获取在数组 arr 中第一个大于基准值的索引。reverse 控制反向查找或正向查找
  Args:
    arr(list):
    benchmark():
    reverse(bool):

  Returns:

  """
  if reverse:
    start = len(arr) - 1
    stop = -1
    step = -1
  else:
    start = 0
    stop = len(arr)
    step = 1

  for i in range(start, stop, step):
    if arr[i] > benchmark:
      return i

  return -1



def tf2torch():
  """
  convert_tf_checkpoint_to_pytorch(r"\model\chinese_L-12_H-768_A-12\bert_model.ckpt",
                                   r"\model\chinese_L-12_H-768_A-12\bert_config.json",
                                   r"\model\chinese_L-12_H-768_A-12\bert_model.bin")
  Returns:

  """

def to_torch_tensor(value):
  """
  python 数据类型转 torch.
  Args:
    value:

  Returns:

  """
  return torch.Tensor(value).long()


def reshape_tensor(tensor, shape):
  return tensor.contiguous().view(*shape)

def mask(tensor, tensor_mask, mask_dim):
  """
  Mask a tensor.
  Args:
    tensor(torch.Tensor): 输入
    tensor_mask(torch.Tensor): mask 位置信息.
    mask_dim(int): 负数，指定需要 mask 的维度，example：mask_dim = -1, 表示在最后一维上做 mask 操作.

  Returns:

  """
  if not mask_dim < 0:
    raise Exception(f"Mask dim only supports negative numbers! Mask dim: {mask_dim} ")

  for i in range(-mask_dim - 1):
    tensor_mask = tensor_mask.unsqueeze(-1)
  return tensor * tensor_mask


def load_class(package, name, class_name):
  """"""
  if not name:
    return

  module_addr = f"{package}.{name}"
  module = importlib.import_module(module_addr, ".")
  class_object = getattr(module, class_name)
  return class_object


def test():
  import random
  a = [random.randrange(-1, 1) for i in range(10)]
  print(a)
  print(get_first_non_negative_index(a, True))

def get_gradient(model):
  """

  Args:
    model:

  Returns:

  """
  gradients = {}
  for name, parameter in model.named_parameters():
    keys = name.split(".")
    grad = parameter.grad
    keys.append(grad)
    add_value4dict(gradients, *keys)
  return gradients

def get_parameter_values(model):
  """

  Args:
    model:

  Returns:

  """
  parameter_values = {}
  for name, parameter in model.named_parameters():
    keys = name.split(".")
    data = parameter.data
    keys.append(data)
    add_value4dict(parameter_values, *keys)
  return parameter_values

def gelu(tensor):
  cdf = 0.5 *(1.0 + torch.erf(tensor/math.sqrt(2.0)))
  return tensor * cdf

if __name__ == "__main__":
  test()

