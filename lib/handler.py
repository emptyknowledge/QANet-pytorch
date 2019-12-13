# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author:
#
# cython: language_level=3
#

import re
import os
import torch
from collections import Counter
import lib.config as config
from lib.CMRC_QADataSet import CMRC_QADataSet
from lib.QADataSet import QADataSet
from lib.config import logger
from my_py_toolkit.file.file_toolkit import *
from lib.utils import write2file
from my_py_toolkit.decorator.decorator import fn_timer


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


def convert_valid_result(Cwids, Qwids, y1s, y2s, p1s, p2s, dataset, ids):
  """"""
  result = []
  for Cwid, Qwid, y1, y2, p1, p2 in zip(Cwids, Qwids, y1s, y2s, p1s, p2s, ids):
    Cw_text = dataset.get_origin_data(ids, "context")
    Qw_text = dataset.get_origin_data(ids, "question")
    Aw_text = dataset.get_origin_data(ids, "question")
    # y1, y2 = int(y1), int(y2)
    p1, p2 = int(p1), int(p2)
    result.append({
      "context": Cw_text,
      "question": Qw_text,
      "labelled_answer": Aw_text,
      "predict_answer": Cw_text[p1: p2]
    })
  return result


def load_model(model_dir, check_point):
  """

  Returns:

  """
  model_path = os.path.join(model_dir, f"model_{check_point}.pt")
  return torch.load(model_path)


@fn_timer(logger)
def get_model():
  """
  Gets models.
  Returns:

  """
  from lib.models import QANet
  if not config.is_continue:
    return QANet()
  else:
    logger.info(f"Continue train, continue_checkpoint: {config.continue_checkpoint}")
    model_path = os.path.join(config.model_dir,
                              f"model_{config.continue_checkpoint}.pkl")
    if not config.is_only_save_params:
      model = torch.load(model_path, map_location=config.device)
    else:
      model = QANet()
      model.load_state_dict(torch.load(model_path))
    return model
  
def save_model(model, steps=0):
  model_path = os.path.join(config.model_dir, f"model_{str(steps)}.pkl")
  make_path_legal(model_path)
  if not config.is_only_save_params:
    torch.save(model, model_path)
  else:
    torch.save(model.state_dict(), model_path)

@fn_timer(logger)
def get_dataset(data_type="train"):
  """

  Args:
    data_type(str): train, dev, trial

  Returns:

  """
  data_dir = config.dataset_dir.get(config.dataset_name)
  data_path = [path for path in get_file_paths(data_dir) if data_type in get_file_name(path)][0]
  if "cmcr" in config.dataset_name:
    dataset = CMRC_QADataSet(data_path,
                             config.batch_size)
    return dataset
  else:
    dataset = QADataSet(data_path,
                        config.batch_size)
    return dataset

def record_info(losses, f1=[], em=[], valid_result={}, iter_num=0,
                r_type="train"):
  """
  记录训练中的 loss, f1, em 值.
  Args:
    losses:
    f1:
    em:
    r_type:

  Returns:

  """
  dir_name = f"./log/{r_type}/"
  losses = [str(v) for v in losses]
  f1 = [str(v) for v in f1]
  em = [str(v) for v in em]
  if losses:
    write2file(",".join(losses), f"{dir_name}losses.txt")
  if f1:
    write2file(",".join(f1), f"{dir_name}f1.txt")
  if em:
    write2file(",".join(em), f"{dir_name}em.txt")

  if valid_result:
    writejson(valid_result, f"{dir_name}valid_result_{iter_num}.json")