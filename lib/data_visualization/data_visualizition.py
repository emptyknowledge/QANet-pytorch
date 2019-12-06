# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
#
# cython: language_level=3
#

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

if __name__ == "__main__":
  data_path = "../../log/losses_log.txt"
  with open(data_path, "r", encoding="utf-8") as f:
    data = f.read()
  data = [float(v) for v in data.split(",")]
  view_point_data(data, is_save=True, save_path="./test.png", is_show=True)