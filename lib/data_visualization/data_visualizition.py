# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
#
# cython: language_level=3
#

import re
from my_py_toolkit.file.file_toolkit import make_path_legal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def view_point_data(data, color="blue", is_save=False, save_path=None,
                    is_show=False):
  """

  Args:
    data(list):
    color(str):

  Returns:

  """
  fig, ax = plt.subplots()
  if data and isinstance(data[0], (list, tuple)):
    y = data[0]
    x = data[1]
  else:
    y = data
    x = list(range(len(data)))
  x = np.asarray(x)
  y = np.asarray(y)
  ax.scatter(x, y, c=color, label=color,
             alpha=0.3, edgecolors='none')
  ax.legend()
  ax.grid(True)
  if is_save and save_path:
    make_path_legal(save_path)
    plt.savefig(save_path)
  if is_show:
    plt.show()

def view_loss():
  data_path = "../../log/train/losses.txt"
  with open(data_path, "r", encoding="utf-8") as f:
    data = f.read()
  data = [float(v) for v in data.split(",")]
  view_point_data(data, is_save=True, save_path="./test.png", is_show=True)

def view_limited_loss(min_loss=None, max_loss=None):
  losses = []
  data_path = "../../log/train/losses.txt"
  with open(data_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
    # line = f.readline()
      line = re.sub("[\\s]*", "", line)
      if not line:
        continue
      data = [float(v) for v in line.split(",")]
      if min_loss is not None:
        data = [v for v in data if v > min_loss]

      if max_loss is not None:
        data = [v for v in data if v < max_loss]

      losses += data
  view_point_data(losses, is_save=True, save_path="./test.png", is_show=True)


if __name__ == "__main__":
  view_limited_loss(min_loss=1000)