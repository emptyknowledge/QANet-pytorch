# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
#
#
# cython: language_level=3
#

import os
import torch

def write2file(file_path, content, is_continue=True):
  """"""
  if not os.path.dirname(file_path):
    os.makedirs(os.path.dirname(file_path))
  mode = "w+" if is_continue else "w"
  with open(file_path, mode, encoding="utf-8")as f:
    f.write("\n" + content)

