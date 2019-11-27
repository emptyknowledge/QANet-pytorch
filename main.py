import lib.config as config
from math import log2
import os
import numpy as np
import ujson as json
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
from my_py_toolkit.decorator.decorator import fn_timer

'''
Some functions are from the official evaluation script.
'''

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
device = config.device


class SQuADDataset(Dataset):
  def __init__(self, npz_file, num_steps, batch_size):
    super().__init__()
    data = np.load(npz_file)
    self.context_idxs = torch.from_numpy(data["context_idxs"]).long()
    self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
    self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
    self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()
    self.y1s = torch.from_numpy(data["y1s"]).long()
    self.y2s = torch.from_numpy(data["y2s"]).long()
    self.ids = torch.from_numpy(data["ids"]).long()
    num = len(self.ids)
    self.batch_size = batch_size
    self.num_steps = num_steps if num_steps >= 0 else num // batch_size
    num_items = num_steps * batch_size
    idxs = list(range(num))
    self.idx_map = []
    i, j = 0, num

    while j <= num_items:
      random.shuffle(idxs)
      self.idx_map += idxs.copy()
      i = j
      j += num
    random.shuffle(idxs)
    self.idx_map += idxs[:num_items - i]

  def __len__(self):
    return self.num_steps

  def __getitem__(self, item):
    idxs = torch.LongTensor(self.idx_map[item:item + self.batch_size])
    res = (self.context_idxs[idxs],
           self.context_char_idxs[idxs],
           self.ques_idxs[idxs],
           self.ques_char_idxs[idxs],
           self.y1s[idxs],
           self.y2s[idxs], self.ids[idxs])
    return res


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


def convert_tokens(eval_file, qa_id, pp1, pp2):
  answer_dict = {}
  remapped_dict = {}
  for qid, p1, p2 in zip(qa_id, pp1, pp2):
    context = eval_file[str(qid)]["context"]
    spans = eval_file[str(qid)]["spans"]
    uuid = eval_file[str(qid)]["uuid"]
    l = len(spans)
    if p1 >= l or p2 >= l:
      ans = ""
    else:
      start_idx = spans[p1][0]
      end_idx = spans[p2][1]
      ans = context[start_idx: end_idx]
    answer_dict[str(qid)] = ans
    remapped_dict[uuid] = ans
  return answer_dict, remapped_dict


def evaluate_valid_result(valid_result):
  f1 = exact_match = total = 0
  for item in valid_result:
    total += 1
    ground_truths = item.get("labelled_answer")
    prediction = item.get("predict_answer")
    exact_match += metric_max_over_ground_truths(exact_match_score, prediction,
                                                 ground_truths)
    f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total
  return {'exact_match': exact_match, 'f1': f1}


def evaluate(eval_file, answer_dict):
  f1 = exact_match = total = 0
  for key, value in answer_dict.items():
    total += 1
    ground_truths = eval_file[key]["answers"]
    prediction = value
    exact_match += metric_max_over_ground_truths(exact_match_score, prediction,
                                                 ground_truths)
    f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total
  return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
  # prediction_tokens = normalize_answer(prediction).split()
  # ground_truth_tokens = normalize_answer(ground_truth).split()
  prediction_tokens = prediction.split()
  ground_truth_tokens = ground_truth.split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def exact_match_score(prediction, ground_truth):
  # return (normalize_answer(prediction) == normalize_answer(ground_truth))
  return prediction == ground_truth


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths)

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
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    scheduler.step()
    for name, p in model.named_parameters():
      if p.requires_grad: ema.update_parameter(name, p)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
  loss_avg = np.mean(losses)
  logger.info("STEP {:8d} loss {:8f}\n".format(i + 1, loss_avg))

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
      yp1 = torch.argmax(p1, 1)
      yp2 = torch.argmax(p2, 1)
      yps = torch.stack([yp1, yp2], dim=1)
      ymin, _ = torch.min(yps, 1)
      ymax, _ = torch.max(yps, 1)
      valid_result.extend(
        convert_valid_result(Cwid, Qwid, y1, y2, yp1, yp2, dataset))
      # answer_dict_, _ = convert_tokens(eval_file, ymin.tolist(), ymax.tolist())
      # answer_dict.update(answer_dict_)
  loss = np.mean(losses)
  # metrics = evaluate(eval_file, answer_dict)
  metrics = evaluate_valid_result(valid_result)
  metrics["loss"] = loss
  logger.info("VALID loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"],
                                                            metrics[
                                                              "exact_match"]))


def convert_valid_result(Cwids, Qwids, y1s, y2s, p1s, p2s, dataset):
  """"""
  result = []
  for Cwid, Qwid, y1, y2, p1, p2 in zip(Cwids, Qwids, y1s, y2s, p1s, p2s):
    Cw_text = dataset.idx2text(Cwid)
    Qw_text = dataset.idx2text(Qwid)
    y1, y2 = int(y1), int(y2)
    p1, p2 = int(p1), int(p2)
    result.append({
      "context": Cw_text,
      "question": Qw_text,
      "labelled_answer": Cw_text[y1: y2],
      "predict_answer": Cw_text[p1: p2]
    })
  return result

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


def load_model(model_dir, check_point):
  """

  Returns:

  """
  model_path = os.path.join(model_dir, f"model_{check_point}.pt")
  return torch.load(model_path)


def get_model():
  from lib.models import QANet
  if not config.is_continue:
    return QANet()
  else:
    logger.info(f"Continue train, continue_checkpoint: {config.continue_checkpoint}")
    model_path = os.path.join(config.model_dir,
                              f"model_{config.continue_checkpoint}.pt")
    model = torch.load(model_path, map_location=config.device)
    return model


def train_entry():
  from lib.models import QANet

  # with open(config.word_emb_file, "r") as fh:
  #     word_mat = np.array(json.load(fh), dtype=np.float32)
  # with open(config.char_emb_file, "r") as fh:
  #     char_mat = np.array(json.load(fh), dtype=np.float32)
  # with open(config.train_eval_file, "r") as fh:
  #     train_eval_file = json.load(fh)
  # with open(config.dev_eval_file, "r") as fh:
  #     dev_eval_file = json.load(fh)

  logger.info("Building model...")

  train_dataset = QADataSet(batch_size=config.batch_size)
  dev_dataset = QADataSet(batch_size=config.batch_size)
  train_eval_file = read_data(config.train_eval_file)
  dev_eval_file = read_data(config.dev_eval_file)

  # train_dataset = SQuADDataset(config.train_record_file, config.num_steps, config.batch_size)
  # dev_dataset = SQuADDataset(config.dev_record_file, config.test_num_batches, config.batch_size)

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
  for epoch in range(epochs):
    logger.info(f"Epoch: {epoch}")
    for iter in range(config.continue_checkpoint + L, N, L):
      logger.info(f"Iter: {iter}")
      train(model, optimizer, scheduler, ema, train_dataset, iter, L)
      valid(model, train_dataset, train_eval_file)
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
        fn = os.path.join(config.save_dir, f"model_{str(iter)}.pt")
        torch.save(model, fn)


def test_entry():
  with open(config.dev_eval_file, "r") as fh:
    dev_eval_file = json.load(fh)
  dev_dataset = SQuADDataset(config.dev_record_file, -1, config.batch_size)
  fn = os.path.join(config.save_dir, "model.pt")
  model = torch.load(fn)
  test(model, dev_dataset, dev_eval_file)


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
