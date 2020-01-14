import lib.config as config
from math import log2
import os
import numpy as np
# import ujson as json
import json
import re
from collections import Counter
import string
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
import traceback
from torch.utils.data import Dataset
from lib.QADataSet import read_data
import argparse
from lib.config import logger

from lib.QADataSet import QADataSet
from lib.handler import *
from lib.utils import *
from lib.handler import *
from my_py_toolkit.file.file_toolkit import writejson
from my_py_toolkit.decorator.decorator import fn_timer

'''
Some functions are from the official evaluation script.
'''

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
device = config.device


class EMA(object):
  """
  EMA（Exponential Moving Average）是指数移动平均值, 在参数更新时考虑了前 n - 1 次的参数值，
  不直接使用梯度更新的参数.
  """
  def __init__(self, decay):
    self.decay = decay
    self.shadows = {}
    self.devices = {}

  def __len__(self):
    return len(self.shadows)

  def get(self, name: str):
    return self.shadows[name].to(self.devices[name])

  def set(self, name: str, param: nn.Parameter):
    self.shadows[name] = param.data.to('cpu').clone()
    self.devices[name] = param.data.device

  def update_parameter(self, name: str, param: nn.Parameter):
    if name in self.shadows:
      data = param.data
      new_shadow = self.decay * data + (1.0 - self.decay) * self.get(name)
      param.data.copy_(new_shadow)
      self.shadows[name] = new_shadow.to('cpu').clone()

@fn_timer(logger)
def train(model, optimizer, scheduler, ema, dataset, start_step, steps_num, epoch):
  model.train()
  clamped_losses = []
  origin_losses = []
  logger.info("start_step train:")
  for step in tqdm(range(start_step, steps_num), total=steps_num - start_step):
    optimizer.zero_grad()
    Cwid, Qwid, answer, ids = dataset[step]
    Cwid, Qwid = Cwid.to(device), Qwid.to(device)
    p1, p2 = model(Cwid, Qwid)
    y1, y2 = answer[:, 0].view(-1).to(device), answer[:, 1].view(-1).to(device)
    loss1 = F.nll_loss(p1, y1, reduction='mean')
    loss2 = F.nll_loss(p2, y2, reduction='mean')
    loss = (loss1 + loss2) / 2
    logger.info(f"Origin Loss: {loss}, step: {step}")
    origin_losses.append(loss.item())
    loss = torch.clamp(loss, min=config.min_loss, max=config.max_loss)
    logger.info(f"Clamped Loss: {loss}, step: {step}")
    clamped_losses.append(loss.item())
    loss.backward()
    optimizer.step()
    scheduler.step()
    for name, p in model.named_parameters():
      if p.requires_grad: ema.update_parameter(name, p)

    if step % config.interval_save == 0:
      save_model(model, step)
      record_info(origin_losses, r_type="train", epoch=step)
      origin_losses = []
  loss_avg = np.mean(clamped_losses)
  logger.info("Epoch {:8d} loss {:8f}\n".format(epoch, loss_avg))

@fn_timer(logger)
def valid(model, dataset, epoch=0):
  model.eval()
  valid_result = []
  losses = []
  logger.info("start valid:")
  loss, metrics = test_model(dataset, losses, model, valid_result, dataset_type="valid")
  record_info(losses, f1=[metrics["f1"]], em=[metrics["exact_match"]],
              valid_result=valid_result,
              epoch=epoch,
              r_type="valid")
  logger.info("VALID loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"],
                                                            metrics["exact_match"]))


def test_model(dataset, losses, model, valid_result, dataset_type="valid"):
  steps = min(get_steps(dataset_type, config.mode), dataset.data_szie)
  with torch.no_grad():
    for i in tqdm(range(0, steps), total=steps):
      Cwid, Qwid, answer, ids = dataset[i]
      Cwid, Qwid = Cwid.to(device), Qwid.to(device)
      y1, y2 = answer[:, 0].view(-1).to(device), answer[:, 1].view(-1).to(
        device)
      p1, p2 = model(Cwid, Qwid)
      y1, y2 = y1.to(device), y2.to(device)
      loss1 = F.nll_loss(torch.log(p1), y1, reduction='mean')
      loss2 = F.nll_loss(torch.log(p2), y2, reduction='mean')
      loss = (loss1 + loss2) / 2
      losses.append(loss.item())
      # TODO: 这里不能直接使用 argmax, 应该找 p1(start) * p2(end) 值最大且 start < end
      # yp1 = torch.argmax(p1, 1)
      # yp2 = torch.argmax(p2, 1)
      # yps = torch.stack([yp1, yp2], dim=1)
      # ymin, _ = torch.min(yps, 1)
      # ymax, _ = torch.max(yps, 1)
      pre_start, pre_end, _ = find_max_proper_batch(p1, p2)
      valid_result.extend(
        convert_valid_result(Cwid, Qwid, y1, y2, pre_start, pre_end, dataset, ids))
  loss = np.mean(losses)
  metrics = evaluate_valid_result(valid_result)
  metrics["loss"] = loss
  return loss, metrics


@fn_timer(logger)
def test(model, dataset, epoch=0):
  model.eval()
  losses = []
  valid_result = []
  print("start test:")
  loss, metrics = test_model(dataset, losses, model, valid_result, dataset_type="test")
  record_info(losses,f1=[metrics["f1"]], em=[metrics["exact_match"]],
              valid_result=valid_result, epoch=epoch,
              r_type="test")
  print("TEST loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"],
                                                     metrics["exact_match"]))
  return metrics

@fn_timer(logger)
def classify(model, dataset):
  # model.eval()
  # losses = []
  # valid_result = []
  less_loss_data = []
  high_loss_data = []
  print("start classify:")
  model.eval()
  with torch.no_grad():
    for i in tqdm(range(0, dataset.data_szie, dataset.batch_size), total=dataset.data_szie):
      logger.info(f"Classify index : {i}")
      Cwid, Qwid, answer, ids = dataset[i]
      Cwid, Qwid = Cwid.to(device), Qwid.to(device)
      y1, y2 = answer[:, 0].view(-1).to(device), answer[:, 1].view(-1).to(
        device)
      p1, p2 = model(Cwid, Qwid)
      y1, y2 = y1.to(device), y2.to(device)
      loss1 = F.nll_loss(torch.log(p1), y1, reduction='none')
      loss2 = F.nll_loss(torch.log(p2), y2, reduction='none')
      loss = (loss1 + loss2) / 2
      origin_data = dataset.get_origin_data(ids, "whole")
      for index, l in enumerate(loss):
        if l < 10e10:
          less_loss_data.append(origin_data[index])
        else:
          high_loss_data.append(origin_data[index])


      
      # losses.append(loss.item())
  # loss, metrics = test_model(dataset, losses, model, valid_result)
  # record_info(losses,f1=[metrics["f1"]], em=[metrics["exact_match"]],
  #             valid_result=valid_result, r_type="test")
  # print("TEST loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"],
  #                                                    metrics["exact_match"]))
  writejson(less_loss_data, config.less_loss_path)
  writejson(high_loss_data, config.high_loss_path)

  # return




def train_entry():
  from lib.models import QANet

  logger.info("Building model...")

  train_dataset = get_dataset("train")
  dev_dataset = get_dataset("dev")
  trial_dataset = get_dataset("trial")

  lr = config.learning_rate
  base_lr = 1.0
  warm_up = config.lr_warm_up_num

  model = get_model().to(device)

  ema = EMA(config.ema_decay)
  for name, p in model.named_parameters():
    if p.requires_grad: ema.set(name, p)
  params = filter(lambda param: param.requires_grad, model.parameters())
  optimizer = optim.Adam(lr=base_lr, betas=(config.beta1, config.beta2),
                         eps=1e-7, weight_decay=3e-7, params=params)
  cr = lr / log2(warm_up)
  scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                          lr_lambda=lambda ee: cr * log2(
                                            ee + 1) if ee < warm_up else lr)
  epochs = config.epochs
  best_f1 = best_em = patience = 0
  start_index = 0 if not config.is_continue else config.continue_checkpoint
  for epoch in range(config.start_epoch, epochs):
    logger.info(f"Epoch: {epoch}")
    train(model, optimizer, scheduler, ema, train_dataset, start_index,
          get_steps("train", config.mode), epoch)
          # 1, epoch) # todo: debug 完删掉
    valid(model, dev_dataset, epoch)
    metrics = test(model, trial_dataset, epoch)
    logger.info("Learning rate: {}".format(scheduler.get_lr()))
    dev_f1 = metrics["f1"]
    dev_em = metrics["exact_match"]
    if dev_f1 < best_f1 and dev_em < best_em:
      patience += 1
      if patience > config.early_stop: break
    else:
      patience = 0
      best_f1 = max(best_f1, dev_f1)
      best_em = max(best_em, dev_em)
    start_index = 0
    train_dataset.shuffle()


# def test_entry():
#   with open(config.dev_eval_file, "r") as fh:
#     dev_eval_file = json.load(fh)
#   dev_dataset = SQuADDataset(config.dev_record_file, -1, config.batch_size)
#   fn = os.path.join(config.save_dir, "model.pt")
#   model = torch.load(fn)
#   test(model, dev_dataset, dev_eval_file)

def classify_data():
  from lib.models import QANet

  logger.info("Building model...")
  train_dataset = get_dataset("train")
  model = get_model().to(device)
  classify(model, train_dataset)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", action="store", dest="mode", default="train",
                      help="train/test/debug")
  pargs = parser.parse_args()
  mode = config.mode # pargs.mode
  logger.info("Current device is {}".format(device))
  if mode == "train":
    train_entry()
  elif mode == "debug":
    config.batch_size = 2
    config.num_steps = 32
    config.test_num_batches = 2
    config.val_num_batches = 2
    config.checkpoint = 2
    config.period = 1
    train_entry()
  elif mode == "test":
    test_entry()
  elif mode == "classify":
    classify_data()
  else:
    print("Unknown mode")
    exit(0)

if __name__ == '__main__':
  try:
    main()
  except Exception:
    logger.error(traceback.format_exc())
