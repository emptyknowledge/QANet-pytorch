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
import importlib
from my_py_toolkit.file.file_toolkit import *
from lib.utils import *
from my_py_toolkit.decorator.decorator import fn_timer
from pytorch_transformers import BertModel

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
  """

  Args:
    prediction(str):
    ground_truth(str):

  Returns:

  """
  # prediction_tokens = normalize_answer(prediction).split()
  # ground_truth_tokens = normalize_answer(ground_truth).split()
  prediction_tokens = list(prediction)
  ground_truth_tokens = list(ground_truth)
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
  """
  计算准确率.
  Args:
    metric_fn:
    prediction(str):
    ground_truths(str):

  Returns:

  """
  # todo: 代码有问题需要修改
  # scores_for_ground_truths = []
  # for ground_truth in ground_truths:
  #   score = metric_fn(prediction, ground_truth)
  #   scores_for_ground_truths.append(score)
  # return max(scores_for_ground_truths)
  return metric_fn(prediction, ground_truths)


def convert_valid_result(Cwids, Qwids, y1s, y2s, p1s, p2s, dataset, ids):
  """"""
  result = []
  for Cwid, Qwid, y1, y2, p1, p2, id in zip(Cwids, Qwids, y1s, y2s, p1s, p2s, ids):
    Cw_text = dataset.get_origin_data(id, "context")
    Qw_text = dataset.get_origin_data(id, "question")
    Aw_text = dataset.get_origin_data(id, "answer_text")
    # y1, y2 = int(y1), int(y2)
    p1, p2 = int(p1), int(p2)
    result.append({
      "context": Cw_text,
      "question": Qw_text,
      "labelled_answer": Aw_text,
      "predict_answer": Cw_text[p1: p2]
    })
  return result

def convert_valid_result_baseline(y1s, y2s, p1s, p2s, dataset, ids):
  """"""
  result = []
  for y1, y2, p1, p2, id in zip(y1s, y2s, p1s, p2s, ids):
    input_feature = dataset.input_features(id)
    # Cw_text = dataset.get_origin_data(id, "context")
    # Qw_text = dataset.get_origin_data(id, "question")
    # Aw_text = dataset.get_origin_data(id, "answer_text")
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

def load_bert(model_path, device):
  """
  加载与训练的 bert.
  Args:
    model_path:
    device:

  Returns:

  """
  return BertModel.from_pretrained(model_path).to(device)

@fn_timer(logger)
def get_model(package, name, class_name):
  """
  Gets models.
  Returns:

  """
  model_class = load_class(package, name, class_name)
  if not config.is_continue:
    return model_class()
  else:
    logger.info(f"Continue train, continue_checkpoint: {config.continue_checkpoint}")
    model_path = os.path.join(config.model_dir,
                              f"model_{config.continue_checkpoint}.pkl")
    if not config.is_only_save_params:
      model = torch.load(model_path, map_location=config.device)
    else:
      model = model_class()
      model.load_state_dict(torch.load(model_path))
    return model
  
def save_model(model, steps=0):
  model_path = os.path.join(config.model_dir, f"model_{str(steps)}.pkl")
  make_path_legal(model_path)
  if not config.is_only_save_params:
    torch.save(model, model_path)
    # torch.save(model.embedding.trainable_embedding,
    #            config.embedding_trainable_model)
    # torch.save(data_set.trainable_embedding, config.embedding_trainable_model)
  else:
    torch.save(model.state_dict(), model_path)
    # torch.save(model.embedding.trainable_embedding,
    #            config.embedding_trainable_model)
    # torch.save(data_set.trainable_embedding, config.embedding_trainable_model)

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
                             config.is_train)
    return dataset
  else:
    dataset = QADataSet(data_path,
                        config.batch_size)
    return dataset

def record_info(losses, f1=[], em=[], valid_result={}, epoch=0,
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
    writejson(valid_result, f"{dir_name}valid_result_{epoch}.json")


def corresponds_index(origin, token):
  """
  将 BertTokenizer.tokenize 出的 token 与原始字符串对应回去，
  主要处理情况：对英文切割不好(eg: luminoso 被拆分为：'lu', '##min', '##os', '##o',)，
  在使用了原始句子下标(index)的情况下，
  BertTokenizer.tokenize 后结果 index 与原句对应不上， index 需要纠正。
  Args:
    origin(list):
    token(list):

  Returns:

  """
  origin_index_in_token = []
  index_in_token = 0
  token_len = len(token)
  for index, char in enumerate(origin):
    if index_in_token > token_len - 1 or re.search("[a-zA-Z\d]", char):
      origin_index_in_token.append(-1)
      continue

    for j in range(index_in_token, token_len):
      if char == token[j]:
        origin_index_in_token.append(j)
        index_in_token = j
        break

    while len(origin_index_in_token) < index + 1:
      origin_index_in_token.append(-1)

  return origin_index_in_token

def transfer_index(origin_content, tokenize_content, *indexes):
  """
  将 origin content 中的 index 映射到 tokenize_content 中的 Index.
  TODO: 目前英文单词后面紧结数字，返回 index 不对, 待修复。
  Args:
    origin_content:
    tokenize_content:
    *indexes:

  Returns:

  """
  text_index_in_token = corresponds_index(origin_content, tokenize_content)
  result = []
  for i in indexes:
    if text_index_in_token[i] > -1:
      result.append(text_index_in_token[i])
    else:
      first_non_negative_index = get_first_non_negative_index(text_index_in_token[:i], reverse=True)
      if first_non_negative_index > -1:
        result.append(text_index_in_token[first_non_negative_index] + 1)
      else:
        result.append(-1)
  return result


def get_trainable_embedding(vocab_size):
  """
  获取可训练 embedding.
  Returns:

  """
  embedding = torch.Tensor(vocab_size, config.embedding_trainable_dim)
  embedding = torch.nn.Parameter(torch.nn.init.normal_(embedding).to(config.device))
  return embedding
  # if not config.is_continue or not os.path.exists(config.embedding_trainable_model):
  #   # torch.nn.init.normal_(torch.Tensor(3,3)).requires_grad_(True)
  #   embedding = torch.Tensor(vocab_size, config.embedding_trainable_dim)
  #   embedding = torch.nn.Parameter(torch.nn.init.normal_(embedding)).to(config.device).requires_grad_(True)
  #   return embedding
  # else:
  #   return torch.load(config.embedding_trainable_model).to(config.device).requires_grad_(True)


def adapt_conv2D(input_tensor, origin_shape):
  """
  将输入转换为适合二维 conv （卷积）的维度.
  词向量输入： batch_size, length, word_dim, 二维卷积需要的维度: batch_size, length, word_dim, channel,
  将输入增加一维。
  Args:
    input_tensor:

  Returns:

  """
  origin_shape = list(origin_shape)
  if len(origin_shape) == 3:
    origin_shape.append(1)
    input_tensor = input_tensor.contiguous().view(*origin_shape)

  return input_tensor

def adapt_normal(input_tensor, origin_shape):
  """
  调整 normal 操作所需的 input tensor shape.
  Args:
    input_tensor:

  Returns:

  """
  origin_shape = list(origin_shape)
  input_tensor = input_tensor.contiguous().view(origin_shape[0], -1)

  return input_tensor

def adapt_line(input_tensor, origin_shape):
  """
  调整 normal 操作所需的 input tensor shape.
  Args:
    input_tensor:

  Returns:

  """
  origin_shape = list(origin_shape)
  new_shape = origin_shape
  if new_shape[-1] != 1:
    new_shape.append(1)
  input_tensor = input_tensor.contiguous().view(*new_shape)

  return input_tensor


def get_steps(data_set_type, mode="train"):
  """

  Args:
    data_set_type:
    mode:train, debug

  Returns:

  """
  if mode == "debug":
    return 1

  if data_set_type == "train":
    return config.num_steps

  if data_set_type == "valid":
    return config.val_num_steps

  if data_set_type == "test":
    return config.test_num_steps