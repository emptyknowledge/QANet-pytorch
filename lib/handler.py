# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author:
#
# cython: language_level=3
#

import os
import torch
from collections import Counter
import lib.config as config
from lib.config import logger



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
                              f"model_{config.continue_checkpoint}.pkl")
    if not config.is_only_save_params:
      model = torch.load(model_path, map_location=config.device)
    else:
      model = QANet()
      model.load_state_dict(torch.load(model_path))
    return model
  
def save_model(model, steps=0):
  model_path = os.path.join(config.model_dir, f"model_{str(steps)}.pkl")
  if not config.is_only_save_params:
    torch.save(model, model_path)
  else:
    torch.save(model.state_dict(), model_path)
    