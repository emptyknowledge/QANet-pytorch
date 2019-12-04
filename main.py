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
from torch.utils.data import Dataset
from lib.QADataSet import read_data
import argparse
from lib.config import logger

from lib.QADataSet import QADataSet
from lib.handler import *
from lib.utils import *
from my_py_toolkit.file.file_toolkit import writejson
from my_py_toolkit.decorator.decorator import fn_timer

'''
Some functions are from the official evaluation script.
'''

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
device = config.device


class EMA(object):
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
def train(model, optimizer, scheduler, ema, dataset, start, length):
  model.train()
  losses = []
  logger.info("start train:")
  for i in tqdm(range(start, length + start), total=length):
    optimizer.zero_grad()
    Cwid, Qwid, answer = dataset[i]
    Cwid, Qwid = Cwid.to(device), Qwid.to(device)
    p1, p2 = model(Cwid, Qwid)
    y1, y2 = answer[:, 0].view(-1).to(device), answer[:, 1].view(-1).to(device)
    loss1 = F.nll_loss(p1, y1, reduction='mean')
    loss2 = F.nll_loss(p2, y2, reduction='mean')
    loss = (loss1 + loss2) / 2
    logger.info(f"Origin Loss: {loss}")
    loss = torch.clamp(loss, min=config.min_loss, max=config.max_loss)
    logger.info(f"Clamped Loss: {loss}")
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    scheduler.step()
    for name, p in model.named_parameters():
      if p.requires_grad: ema.update_parameter(name, p)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
  loss_avg = np.mean(losses)
  logger.info("STEP {:8d} loss {:8f}\n".format(i + 1, loss_avg))
  return losses

@fn_timer(logger)
def valid(model, dataset, eval_file):
  model.eval()
  answer_dict = {}
  valid_result = []
  losses = []
  num_batches = config.val_num_batches
  logger.info("start valid:")
  with torch.no_grad():
    for i in tqdm(
      random.sample(range(0, len(dataset)),
                    min(num_batches, len(dataset))),
      total=num_batches):
      Cwid, Qwid, answer = dataset[i]
      Cwid, Qwid = Cwid.to(device), Qwid.to(device)
      y1, y2 = answer[:, 0].view(-1).to(device), answer[:, 1].view(-1).to(
        device)
      p1, p2 = model(Cwid, Qwid)
      y1, y2 = y1.to(device), y2.to(device)
      loss1 = F.nll_loss(p1, y1, reduction='mean')
      loss2 = F.nll_loss(p2, y2, reduction='mean')
      loss = (loss1 + loss2) / 2
      losses.append(loss.item())
      # TODO: 这里不能直接使用 argmax, 应该找 p1(start) * p2(end) 值最大且 start < end
      yp1 = torch.argmax(p1, 1)
      yp2 = torch.argmax(p2, 1)
      yps = torch.stack([yp1, yp2], dim=1)
      ymin, _ = torch.min(yps, 1)
      ymax, _ = torch.max(yps, 1)
      valid_result.extend(
        convert_valid_result(Cwid, Qwid, y1, y2, yp1, yp2, dataset))
  loss = np.mean(losses)
  # metrics = evaluate(eval_file, answer_dict)
  metrics = evaluate_valid_result(valid_result)
  metrics["loss"] = loss
  logger.info("VALID loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"],
                                                            metrics["exact_match"]))
  return valid_result


@fn_timer(logger)
def test(model, dataset, eval_file):
  model.eval()
  answer_dict = {}
  losses = []
  valid_result = []
  num_batches = config.test_num_batches
  print("start test:")
  with torch.no_grad():
    for i in tqdm(range(num_batches), total=min(num_batches, len(dataset))):
      Cwid, Qwid, answer = dataset[i]
      Cwid, Qwid = Cwid.to(device), Qwid.to(device)
      y1, y2 = answer[:, 0].view(-1).to(device), answer[:, 1].view(-1).to(
        device)
      p1, p2 = model(Cwid, Qwid)
      y1, y2 = y1.to(device), y2.to(device)
      p1, p2 = model(Cwid, Qwid)
      y1, y2 = y1.to(device), y2.to(device)
      loss1 = F.nll_loss(p1, y1, reduction='mean')
      loss2 = F.nll_loss(p2, y2, reduction='mean')
      loss = (loss1 + loss2) / 2
      losses.append(loss.item())
      yp1 = torch.argmax(p1, 1)
      yp2 = torch.argmax(p2, 1)
      yps = torch.stack([yp1, yp2], dim=1)
      ymin, _ = torch.min(yps, 1)
      ymax, _ = torch.max(yps, 1)
      valid_result.extend(
        convert_valid_result(Cwid, Qwid, y1, y2, yp1, yp2, dataset))
      # answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
      # answer_dict.update(answer_dict_)
  loss = np.mean(losses)
  # metrics = evaluate(eval_file, answer_dict)
  metrics = evaluate_valid_result(valid_result)
  f = open("log/answers.json", "w")
  json.dump(valid_result, f)
  f.close()
  metrics["loss"] = loss
  print("TEST loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"],
                                                     metrics["exact_match"]))
  return metrics




def train_entry():
  from lib.models import QANet

  logger.info("Building model...")

  # train_dataset = QADataSet(batch_size=config.batch_size)
  # dev_dataset = QADataSet(batch_size=config.batch_size)
  train_dataset = get_dataset()
  dev_dataset = get_dataset()
  train_eval_file = read_data(config.train_eval_file)
  dev_eval_file = read_data(config.dev_eval_file)

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
  L = config.checkpoint
  N = config.num_steps
  epochs = config.epochs
  best_f1 = best_em = patience = 0
  loss_of_each_sample = []
  for epoch in range(epochs):
    logger.info(f"Epoch: {epoch}")
    for iter in range(config.continue_checkpoint + L, N, L):
      logger.info(f"Iter: {iter}")
      loss_of_each_sample.extend(train(model, optimizer, scheduler, ema,
                                       train_dataset, iter,
                                       train_dataset.data_szie))
      valid_result = valid(model, train_dataset, train_eval_file)
      metrics = test(model, dev_dataset, dev_eval_file)
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
      if iter % config.interval_save == 0:
        save_model(model, iter)
        valid_res_path = os.path.join(config.valid_result_dir,
                                      f"valid_result_{iter}.json")
        writejson(valid_result, valid_res_path)
      write2file(config.losses_path,
                 ",".join([str(v) for v in loss_of_each_sample]),
                 is_continue=config.is_continue)
      loss_of_each_sample = []


# def test_entry():
#   with open(config.dev_eval_file, "r") as fh:
#     dev_eval_file = json.load(fh)
#   dev_dataset = SQuADDataset(config.dev_record_file, -1, config.batch_size)
#   fn = os.path.join(config.save_dir, "model.pt")
#   model = torch.load(fn)
#   test(model, dev_dataset, dev_eval_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", action="store", dest="mode", default="train",
                      help="train/test/debug")
  pargs = parser.parse_args()
  logger.info("Current device is {}".format(device))
  if pargs.mode == "train":
    train_entry()
  elif pargs.mode == "debug":
    config.batch_size = 2
    config.num_steps = 32
    config.test_num_batches = 2
    config.val_num_batches = 2
    config.checkpoint = 2
    config.period = 1
    train_entry()
  elif pargs.mode == "test":
    test_entry()
  else:
    print("Unknown mode")
    exit(0)
