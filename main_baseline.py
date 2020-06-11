import lib.config as config
import copy
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
from lib.tokenization import FullTokenizer
from lib.QADataSet import QADataSet
from lib.handler import *
from lib.utils import *
from lib.pytorch_optimization import get_optimization
from lib.utils_bert_finetune import torch_show_all_params, torch_init_model
from lib.handler import *
from lib.data_visualization.data_visualizition import visual_tensorboard
from my_py_toolkit.file.file_toolkit import writejson
from my_py_toolkit.decorator.decorator import fn_timer

from lib.model_baseline import BertConfig
from lib.tokenizations import official_tokenization as tokenization
from torch.utils.data import TensorDataset, DataLoader

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
def train(model, optimizer, scheduler, ema, bert_config, start_step, steps_num, epoch):
  model.train()
  tokenizer, train_dataloader = load_data(bert_config)
  clamped_losses = []
  origin_losses = []
  exact_match_total = 0
  f1_total = 0
  exact_match = 0
  f1 = 0
  logger.info("start_step train:")
  softmax = torch.nn.Softmax(dim=-1)
  log_sofmax = torch.nn.LogSoftmax(dim=-1)
  for step, batch in enumerate(train_dataloader):
    try:
      optimizer.zero_grad()
      batch = tuple(t.to(device) for t in batch)
      input_ids, input_mask, segment_ids, start_positions, end_positions = batch
      start_positions, end_positions = start_positions.to(
        device), end_positions.to(device)
      input_mask = input_mask.float()
      start_embeddings, end_embeddings = model(input_ids, input_mask,
                                               segment_ids)
      loss1 = F.nll_loss(log_sofmax(start_embeddings), start_positions,
                         reduction='mean')
      loss2 = F.nll_loss(log_sofmax(end_embeddings), end_positions,
                         reduction='mean')
      loss = (loss1 + loss2) / 2
      logger.info(f"Origin Loss: {loss}, epoch: {epoch}, step: {step}")
      origin_losses.append(loss.item())

      pre_start, pre_end, probabilities = find_max_proper_batch(
        softmax(start_embeddings), softmax(end_embeddings))
      pre_loss = loss
      cur_res = convert_pre_res(input_ids, pre_start, pre_end, start_positions, end_positions, probabilities, tokenizer)

      loss = torch.clamp(loss, min=config.min_loss, max=config.max_loss)
      logger.info(f"Clamped Loss: {loss}, epoch: {epoch}, step: {step}")
      clamped_losses.append(loss.item())
      loss.backward()
      record_info(valid_result=cur_res, epoch=epoch, is_continue=True)
      exact_match_total, f1_total, exact_match, f1 = evaluate_valid_result(cur_res, exact_match_total, f1_total, (step+1) * config.n_batch)
      visual_data(model, loss, pre_loss, optimizer, epoch, step, exact_match_total, f1_total, exact_match, f1, label="train")
      optimizer.step()
      scheduler.step()
      if config.use_ema:
        for name, p in model.named_parameters():
          if p.requires_grad: ema.update_parameter(name, p)

      if step % config.interval_save == 0:
        save_model(model, optimizer, step)
        record_info(origin_losses, r_type="train", epoch=epoch)
        origin_losses = []
    except Exception:
      logger.error(traceback.format_exc())
  loss_avg = np.mean(clamped_losses)
  logger.info("Epoch {:8d} loss {:8f}\n".format(epoch, loss_avg))


def cal_pre_loss(pre_start, pre_end, start_positions, end_positions,
                 start_embeddings, end_embeddings, log_softmax):
  """"""
  loss1 = F.nll_loss(covert_position_embeddings(start_embeddings, pre_start, log_softmax).to(device),
                     start_positions,
                     reduction="mean")
  loss2 = F.nll_loss(covert_position_embeddings(end_embeddings, pre_end, log_softmax).to(device),
                     end_positions,
                     reduction="mean")
  loss = (loss1 + loss2) / 2
  loss = loss.to("cpu")
  return loss


@fn_timer(logger)
def visual_data(model, loss, pre_loss, optimizer, epoch, step, exact_match_total, f1_total, exact_match, f1, label="train"):
  """
  可视化数据
  Args:
    model:
    loss:
    epoch:
    step:

  Returns:

  """
  if config.visual_gradient:
    gradient = get_gradient(model)
    gradient = transfer_multi_layer_dict(gradient)
    visual_tensorboard(config.visual_gradient_dir, f"{label}_gradient", gradient, epoch, step)
  if config.visual_parameter:
    parameter_values = get_parameter_values(model)
    parameter_values = transfer_multi_layer_dict(parameter_values)
    visual_tensorboard(config.visual_parameter_dir, f"{label}_parameter_values", parameter_values, epoch, step)
  if config.visual_loss:
    visual_tensorboard(config.visual_loss_dir, f"{label}_loss", {"loss": [loss.item()]}, epoch, step)
  if config.visual_optimizer:
    visual_tensorboard(config.visual_optimizer_dir, f"{label}_optimizer", process_optimizer_info(optimizer), epoch, step)
  if config.visual_valid_result:
    visual_tensorboard(config.visual_valid_result_dir, f"{label}_valid", {
      "exact_match_total": [exact_match_total],
      "exact_match": [exact_match],
      "f1_total": [f1_total],
      "f1": [f1]
    }, epoch, step)


def process_optimizer_info(optimizer):
  val = copy.deepcopy(optimizer.defaults)
  result = {}
  for k,v in val.items():
    if not isinstance(v, list):
      v = [v]
    result[k] = np.asarray(v)
  return result

@fn_timer(logger)
def valid(model, bert_config, epoch=0):
  model.eval()
  valid_result = []
  losses = []
  logger.info("start valid:")

  loss, extract_match, f1 = test_model(bert_config, losses, model, valid_result, epoch, label="valid")
  record_info(loss, f1=f1, em=extract_match,
              # valid_result=valid_result,
              epoch=epoch,
              r_type="valid")
  logger.info("VALID loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, f1,
                                                            extract_match))


def test_model(bert_config, losses, model, valid_result, epoch, label="valid"):
  logger.info("start_step test or valid:")
  tokenizer, dev_dataloader = load_data(bert_config, config.dev_dir1)
  softmax = torch.nn.Softmax(dim=-1)
  log_sofmax = torch.nn.LogSoftmax(dim=-1)
  exact_match_total = 0
  f1_total = 0
  exact_match = 0
  f1 = 0
  with torch.no_grad():
    for step, batch in enumerate(dev_dataloader):
      batch = tuple(t.to(device) for t in batch)
      input_ids, input_mask, segment_ids, start_positions, end_positions = batch

      start_embeddings, end_embeddings = model(input_ids, input_mask,
                                               segment_ids)
      # start_embeddings = mask(start_embeddings, input_span_mask, -1)
      # end_embeddings = mask(end_embeddings, input_span_mask, -1)
      loss1 = F.nll_loss(log_sofmax(start_embeddings), start_positions,
                         reduction='mean')
      loss2 = F.nll_loss(log_sofmax(end_embeddings), end_positions,
                         reduction='mean')
      loss = (loss1 + loss2) / 2
      logger.info(f"Origin Loss: {loss}, step: {step}")
      losses.append(loss.item())
      pre_start, pre_end, probabilities = find_max_proper_batch(
        softmax(start_embeddings), softmax(end_embeddings))
      cur_res = convert_pre_res(input_ids, pre_start, pre_end, start_positions,
                                end_positions, probabilities, tokenizer)
      record_info(valid_result=cur_res, epoch=epoch, is_continue=True, r_type=label)
      visual_data(model, loss, loss, None, epoch, step,
                  exact_match_total, f1_total, exact_match, f1, label=label)

  loss = np.mean(losses)
  exact_match_total, f1_total, exact_match, f1 = evaluate_valid_result(valid_result, 0, 0, len(dev_dataloader.dataset))
  # metrics["loss"] = loss
  return loss, exact_match, f1


@fn_timer(logger)
def test(model, dataset, epoch=0):
  model.eval()
  losses = []
  valid_result = []
  print("start test:")
  loss, extract_match, f1 = test_model(dataset, losses, model, valid_result, label="test")
  record_info(loss,f1=f1, em=extract_match,
              # valid_result=valid_result,
              epoch=epoch,
              r_type="test")
  print("TEST loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, f1,
                                                     extract_match))
  return 

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

def record_features(dataset):
  if config.is_save_features:
    writejson(dataset.convert_all_features4human_visual(), config.path_save_feature)


def train_entry():
  logger.info("Building model...")
  bert_config = BertConfig.from_json_file(config.bert_config)

  lr = config.learning_rate
  # num_train_steps = int(train_dataset.features_size/ config.batch_size * config.epochs)
  # warm_up = int(num_train_steps * config.warmup_proportion)

  model = get_model(config.model_package, config.model_name, config.model_class_name).to(device)

  ema = EMA(config.ema_decay)
  params = get_model_trainabel_param(ema, model)
  optimizer = get_optimizer(lr, params=params)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   config.T_MAX,
                                                   config.min_lr)
  epochs = config.epochs
  best_f1 = best_em = patience = 0
  start_index = 0 if not config.is_continue else config.continue_checkpoint
  for epoch in range(config.start_epoch, epochs):
    logger.info(f"Epoch: {epoch}")
    train(model, optimizer, scheduler, ema, bert_config, start_index,
          get_steps("train", config.mode), epoch)

    if config.is_test_with_test_dev_dataset:
      valid(model, bert_config, epoch)
      metrics = test(model, bert_config, epoch)
      # logger.info("Learning rate: {}".format(scheduler.get_lr()))
      # dev_f1 = metrics["f1"]
      # dev_em = metrics["exact_match"]
      # if dev_f1 < best_f1 and dev_em < best_em:
      #   patience += 1
      #   if patience > config.early_stop: break
      # else:
      #   patience = 0
      #   best_f1 = max(best_f1, dev_f1)
      #   best_em = max(best_em, dev_em)


def get_model_trainabel_param(ema, model):
  for name, p in model.named_parameters():
    if p.requires_grad: ema.set(name, p)
  params = filter(lambda param: param.requires_grad, model.parameters())
  return params


def load_data(bert_config, feature_path=config.train_dir):
  ############################################################################
  # load data for bert cn finetune
  ############################################################################
  # tokenizer = tokenization.BertTokenizer(vocab_file=config.vocab_file,
  #                                        do_lower_case=True)
  tokenizer = FullTokenizer(
    vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)
  if not os.path.exists(config.train_dir):
    json2features(config.train_file,
                  [config.train_dir.replace('_features_', '_examples_'),
                   config.train_dir],
                  tokenizer, is_training=True,
                  max_seq_length=bert_config.max_position_embeddings)
  if not os.path.exists(config.dev_dir1) or not os.path.exists(config.dev_dir2):
    json2features(config.dev_file, [config.dev_dir1, config.dev_dir2],
                  tokenizer,
                  is_training=False,
                  max_seq_length=bert_config.max_position_embeddings)
  train_features = json.load(open(feature_path, 'r'))
  # dev_examples = json.load(open(config.dev_dir1, 'r'))
  # dev_features = json.load(open(config.dev_dir2, 'r'))
  all_input_ids = torch.tensor([f['input_ids'] for f in train_features],
                               dtype=torch.long)
  all_input_mask = torch.tensor([f['input_mask'] for f in train_features],
                                dtype=torch.long)
  all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features],
                                 dtype=torch.long)
  # true label
  all_start_positions = torch.tensor(
    [f['start_position'] for f in train_features], dtype=torch.long)
  all_end_positions = torch.tensor([f['end_position'] for f in train_features],
                                   dtype=torch.long)
  train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             all_start_positions, all_end_positions)
  train_dataloader = DataLoader(train_data, batch_size=config.n_batch,
                                shuffle=True)
  ############################################################################
  ############################################################################
  ############################################################################
  return tokenizer, train_dataloader


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
  # parser = argparse.ArgumentParser()
  # parser.add_argument("--mode", action="store", dest="mode", default="train",
  #                     help="train/test/debug")
  # pargs = parser.parse_args()
  mode = config.mode
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
