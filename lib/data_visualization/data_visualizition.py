# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
#
# cython: language_level=3
#

import re
from my_py_toolkit.file.file_toolkit import make_path_legal
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import matplotlib

from lib import config

myfont = fm.FontProperties(fname=config.font_file)


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


def draw_bar(data, labels, x_label, y_label, title, width=1, interval=1,
             save_path="./test_bar.png"):
  """

  Args:
    data(numpy.object): 二维数据, The size:（context_length, question_length）
    labels(dict): {x_label: [], y_label: []}
    x_label(str):  x 轴标题.
    y_label(str):  y 轴标题.
    title(str): 表标题.
    interval(str): 间隔, 指定两组数据之间的间隔，默认为1，当为 1 的时候， width > 1
    会造成两组数据有重叠，部分被覆盖.

  Returns:

  """
  plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
  matplotlib.rcParams['font.family'] = 'sans-serif'
  plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
  x_tick_labels = labels.get("x_tick_labels", [])
  y_tick_labels = labels.get("y_tick_labels", [])

  context_length = len(data)

  rects = []
  

  fig, ax = plt.subplots()
  x = np.arange(len(x_tick_labels)) * interval

  averge_width = width / context_length
  for index, c_q_attention in enumerate(data):
    rect = ax.bar(x - averge_width * (context_length/2 - index),
                  c_q_attention,
                  averge_width,
                  label=y_tick_labels[index])
    rects.append(rect)

  ax.set_ylabel(y_label)
  ax.set_xlabel(x_label + "".join(x_tick_labels))
  ax.set_title(title)
  ax.set_xticks(x)
  # 这里需要单独指定字体，否则输出是乱码
  ax.set_xticklabels(x_tick_labels, fontdict={"fontproperties": myfont})
  # ax.legend()

  # for rect in rects:
  #   autolabel(rect, ax)
  fig.tight_layout()
  plt.savefig(save_path)

def autolabel(rects, ax):
  """Attach a text label above each bar in *rects*, displaying its height."""
  for rect in rects:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

def test_draw_bar():
  import numpy as np
  labels = {
    "x_tick_labels": list("张三是男是女？"),
    "y_tick_labels": list("张三是个30岁男青年。")
  }
  data = np.random.rand(len(labels["y_tick_labels"]),
                        len(labels["x_tick_labels"]))
  x_label = "question_chart"
  y_label = "context_attention"
  title = "Attention"
  width = 0.9
  draw_bar(data, labels, x_label, y_label, title, width)
  pass

if __name__ == "__main__":
  # view_limited_loss(max_loss=100)
  test_draw_bar()

