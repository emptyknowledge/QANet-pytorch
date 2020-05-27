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

from lib.QADataSet import QADataSet
from lib.handler import *
from lib.utils import *
from lib.pytorch_optimization import get_optimization
from lib.utils_bert_finetune import torch_show_all_params, torch_init_model
from lib.handler import *
from lib.data_visualization.data_visualizition import visual_tensorboard
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
def train(model, optimizer, scheduler, ema, train_dataloader, start_step, steps_num, epoch):
  model.train()
  clamped_losses = []
  origin_losses = []
  extract_result = []
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
      # input_ids, input_mask, segment_ids, start_positions, end_positions, index
      # input_ids, input_mask, input_span_mask, segment_ids, start_positions, end_positions, index = dataset.get(
      #   step, config.batch_size)
      # logger.info(f"Start positions: {start_positions}, End positions: {end_positions}")
      # input_ids, input_mask, input_span_mask, segment_ids = input_ids.to(device), input_mask.to(
      #   device), input_span_mask.to(device), segment_ids.to(device)
      start_positions, end_positions = start_positions.to(
        device), end_positions.to(device)
      input_mask = input_mask.float()
      start_embeddings, end_embeddings = model(input_ids, input_mask,
                                               segment_ids)
      # start_embeddings = mask(start_embeddings, input_span_mask, -1)
      # end_embeddings = mask(end_embeddings, input_span_mask, -1)
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
      # cur_res = dataset.convert_predict_values_with_batch_feature_index(index,
      #                                                           pre_start,
      #                                                           pre_end,
      #                                                           probabilities)

      loss = torch.clamp(loss, min=config.min_loss, max=config.max_loss)
      logger.info(f"Clamped Loss: {loss}, epoch: {epoch}, step: {step}")
      clamped_losses.append(loss.item())
      loss.backward()
      # record_info(valid_result=cur_res, epoch=epoch, is_continue=True)
      # exact_match_total, f1_total, exact_match, f1 = evaluate_valid_result(cur_res, exact_match_total, f1_total, step + config.batch_size)
      visual_data(model, loss, pre_loss, optimizer, epoch, step, exact_match_total, f1_total, exact_match, f1)
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
  # metrics = evaluate_valid_result(extract_result)
  # metrics["loss"] = loss_avg
  # record_info(origin_losses, f1=[metrics["f1"]], em=[metrics["exact_match"]],
  #             valid_result=extract_result,
  #             epoch=epoch,
  #             r_type="train")
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
def visual_data(model, loss, pre_loss, optimizer, epoch, step, exact_match_total, f1_total, exact_match, f1):
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
    visual_tensorboard(config.visual_gradient_dir, "gradient", gradient, epoch, step)
  if config.visual_parameter:
    parameter_values = get_parameter_values(model)
    parameter_values = transfer_multi_layer_dict(parameter_values)
    visual_tensorboard(config.visual_parameter_dir, "parameter_values", parameter_values, epoch, step)
  if config.visual_loss:
    visual_tensorboard(config.visual_loss_dir, "loss", {"loss": [loss.item()]}, epoch, step)
  if config.visual_optimizer:
    visual_tensorboard(config.visual_optimizer_dir, "optimizer", process_optimizer_info(optimizer), epoch, step)
  if config.visual_valid_result:
    visual_tensorboard(config.visual_valid_result_dir, "valid", {
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
def valid(model, dataset, epoch=0):
  model.eval()
  valid_result = []
  losses = []
  logger.info("start valid:")
  losses, metrics = test_model(dataset, losses, model, valid_result, dataset_type="valid")
  record_info(losses, f1=[metrics["f1"]], em=[metrics["exact_match"]],
              valid_result=valid_result,
              epoch=epoch,
              r_type="valid")
  logger.info("VALID loss {:8f} F1 {:8f} EM {:8f}\n".format(metrics["loss"], metrics["f1"],
                                                            metrics["exact_match"]))


def test_model(dataset, losses, model, valid_result, dataset_type="valid"):
  steps = min(get_steps(dataset_type, config.mode), dataset.features_size)
  logger.info("start_step test or valid:")
  softmax = torch.nn.Softmax(dim=-1)
  log_sofmax = torch.nn.LogSoftmax(dim=-1)
  # valid_result = []
  with torch.no_grad():
    for step in tqdm(range(0, steps, config.batch_size), total=steps):
      input_ids, input_mask, input_span_mask, segment_ids, start_positions, end_positions, index = dataset.get(
        step, config.batch_size)
      input_ids, input_mask, input_span_mask, segment_ids = input_ids.to(device), input_mask.to(
        device), input_span_mask.to(device), segment_ids.to(device)
      start_positions, end_positions = start_positions.to(
        device), end_positions.to(device)
      input_mask = input_mask.float()
      start_embeddings, end_embeddings = model(input_ids, input_mask,
                                               segment_ids)
      start_embeddings = mask(start_embeddings, input_span_mask, -1)
      end_embeddings = mask(end_embeddings, input_span_mask, -1)
      loss1 = F.nll_loss(log_sofmax(start_embeddings), start_positions,
                         reduction='mean')
      loss2 = F.nll_loss(log_sofmax(end_embeddings), end_positions,
                         reduction='mean')
      loss = (loss1 + loss2) / 2
      logger.info(f"Origin Loss: {loss}, step: {step}")
      losses.append(loss.item())
      pre_start, pre_end, probabilities = find_max_proper_batch(
        softmax(start_embeddings), softmax(end_embeddings))
      valid_result.extend(
        dataset.convert_predict_values_with_batch_feature_index(index,
                                                                pre_start,
                                                                pre_end,
                                                                probabilities)
      )
      # valid_result.extend(
      #   convert_valid_result(Cwid, Qwid, y1, y2, pre_start, pre_end, dataset, ids))
  loss = np.mean(losses)
  metrics = evaluate_valid_result(valid_result)
  metrics["loss"] = loss
  return losses, metrics


@fn_timer(logger)
def test(model, dataset, epoch=0):
  model.eval()
  losses = []
  valid_result = []
  print("start test:")
  losses, metrics = test_model(dataset, losses, model, valid_result, dataset_type="test")
  record_info(losses,f1=[metrics["f1"]], em=[metrics["exact_match"]],
              valid_result=valid_result, epoch=epoch,
              r_type="test")
  print("TEST loss {:8f} F1 {:8f} EM {:8f}\n".format(metrics["loss"], metrics["f1"],
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

def record_features(dataset):
  if config.is_save_features:
    writejson(dataset.convert_all_features4human_visual(), config.path_save_feature)


def train_entry():
  from lib.models import QANet
  from lib.model_baseline import BertConfig
  from lib.tokenizations import official_tokenization as tokenization
  from torch.utils.data import TensorDataset, DataLoader
  logger.info("Building model...")
  bert_config = BertConfig.from_json_file(config.bert_config)
  ############################################################################
  # load data for bert cn finetune
  ############################################################################
  tokenizer = tokenization.BertTokenizer(vocab_file=config.vocab_file,
                                         do_lower_case=True)
  if not os.path.exists(config.train_dir):
    json2features(config.train_file,
                  [config.train_dir.replace('_features_', '_examples_'),
                   config.train_dir],
                  tokenizer, is_training=True,
                  max_seq_length=bert_config.max_position_embeddings)

  if not os.path.exists(config.dev_dir1) or not os.path.exists(config.dev_dir2):
    json2features(config.dev_file, [config.dev_dir1, config.dev_dir2], tokenizer,
                  is_training=False,
                  max_seq_length=bert_config.max_position_embeddings)
  train_features = json.load(open(config.train_dir, 'r'))
  dev_examples = json.load(open(config.dev_dir1, 'r'))
  dev_features = json.load(open(config.dev_dir2, 'r'))
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
  lr = config.learning_rate
  base_lr = 5e-5
  base_lr =lr
  # num_train_steps = int(train_dataset.features_size/ config.batch_size * config.epochs)
  # warm_up = int(num_train_steps * config.warmup_proportion)

  model = get_model(config.model_package, config.model_name, config.model_class_name).to(device)

  ema = EMA(config.ema_decay)
  for name, p in model.named_parameters():
    if p.requires_grad: ema.set(name, p)
  params = filter(lambda param: param.requires_grad, model.parameters())
  optimizer = get_optimizer(base_lr, params=params)
  # optimizer = optim.Adam(lr=base_lr, betas=(config.beta1, config.beta2),
  #                        eps=1e-8, weight_decay=3e-7, params=params)
  # optimizer = optim.SGD(lr=lr, params=params)
  # cr = lr / log2(warm_up) if config.mode=="train" else lr
  # scheduler = optim.lr_scheduler.LambdaLR(optimizer,
  #                                         lr_lambda=lambda ee: cr * log2(
  #                                           ee + 1) if ee < warm_up else lr)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   config.T_MAX,
                                                   config.min_lr)
  epochs = config.epochs
  best_f1 = best_em = patience = 0
  start_index = 0 if not config.is_continue else config.continue_checkpoint
  for epoch in range(config.start_epoch, epochs):
    logger.info(f"Epoch: {epoch}")
    train(model, optimizer, scheduler, ema, train_dataloader, start_index,
          get_steps("train", config.mode), epoch)
          # 1, epoch) # todo: debug 完删掉

    # if config.is_test_with_test_dev_dataset:
    #   valid(model, dev_dataset, epoch)
    #   metrics = test(model, trial_dataset, epoch)
    #   logger.info("Learning rate: {}".format(scheduler.get_lr()))
    #   dev_f1 = metrics["f1"]
    #   dev_em = metrics["exact_match"]
    #   if dev_f1 < best_f1 and dev_em < best_em:
    #     patience += 1
    #     if patience > config.early_stop: break
    #   else:
    #     patience = 0
    #     best_f1 = max(best_f1, dev_f1)
    #     best_em = max(best_em, dev_em)
    # start_index = 0
    # train_dataset.shuffle()


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
