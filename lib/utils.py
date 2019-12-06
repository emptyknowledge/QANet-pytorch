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

def tf2torch():
  """
  convert_tf_checkpoint_to_pytorch(r"\model\chinese_L-12_H-768_A-12\bert_model.ckpt",
                                   r"\model\chinese_L-12_H-768_A-12\bert_config.json",
                                   r"\model\chinese_L-12_H-768_A-12\bert_model.bin")
  Returns:

  """