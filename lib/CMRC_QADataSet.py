# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import json
import numpy as np
import random
import torch

from lib import config
from lib.Embedding import BertEmbedding
from torch.utils.data import Dataset
from lib.handle_data import *
from lib.handler import *
from lib.utils import *

def read_data(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

class CMRC_QADataSet_Old(Dataset):
  def __init__(self, data_path="./data/train/cmrc2018/train.json",
               batch_size=1, context_len=512, question_len=20):
    self.data_path = data_path
    self.bert_embedding = BertEmbedding(config.bert_path)
    data = read_data(data_path)
    self.context_len = context_len
    self.question_len = question_len
    self.context_idx = []
    self.question_idx = []
    self.context_trainable_idx = []
    self.question_trainable_idx = []
    self.answer_idx = []
    self.ids = []
    self.batch_size = batch_size
    self.data_szie = 0
    for ids, item in enumerate(data):
      if not self.check_data(item):
        continue
      self.context_idx.append(self.padarr(self.bert_embedding.encode(item.get("context")),
                                          self.context_len))
      self.question_idx.append(self.padarr(self.bert_embedding.encode(item.get("question")),
                                           self.question_len))

      self.ids.append(ids)
      answer = item.get("answer_index")
      self.answer_idx.append(answer)
      self.data_szie += 1
    self.context_idx = torch.Tensor(self.context_idx).long()
    self.question_idx = torch.Tensor(self.question_idx).long()
    self.answer_idx = torch.Tensor(self.answer_idx).long()
    self.ids = torch.Tensor(self.ids).long()
  
    self.idx = list(range(self.data_szie))
    if self.batch_size > self.data_szie:
      self.idx = self.idx * (self.batch_size//self.data_szie + 1)
    random.shuffle(self.idx)

  def shuffle(self):
    """打乱数据顺序"""
    random.shuffle(self.idx)

  def check_data(self, item):
    """"""
    answer = item.get("answer_index")
    if answer[0] >511 or answer[1] >511:
      return False
    else:
      return True

  def __getitem__(self, item):
    # index = self.idx[(item * self.batch_size) % self.data_szie: ((item + 1) * self.batch_size) % self.data_szie]
    # TODO: return trainable embedding.
    if self.batch_size > len(self.idx):
      index = self.idx[(item * self.batch_size) % self.data_szie]
    else:
      start_idx = (item * self.batch_size) % self.data_szie
      end_idx = ((item + 1) * self.batch_size) % self.data_szie
      if start_idx < end_idx:
        index = self.idx[start_idx:end_idx]
      else:
        index = self.idx[start_idx:] + self.idx[:end_idx]
    index = torch.tensor(index).long()
    return (self.context_idx[index],
            self.question_idx[index],
            self.answer_idx[index],
            self.ids[index])

  def __len__(self):
    return self.data_szie

  def padarr(self, val, v_len):
    res = [0] * v_len
    res[0:min(len(val), v_len)] = val[0:min(len(val), v_len)]
    return res

  def idx2text(self, idx):
    return self.bert_embedding.decode2text(idx)

  def idx2multi_text(self, idx):
    return self.bert_embedding.decode2multi_text(idx)

    
  def get_origin_data(self, ids, key="context"):
    """"""
    if isinstance(ids, torch.Tensor):
      ids = ids.tolist()

    data = read_data(self.data_path)
    if isinstance(ids, (str, int)):
      if key == "whole":
        return data[ids]
      else:
        return data[ids].get(key)

    if isinstance(ids, list):
      result = []
      for id in ids:
        if key == "whole":
          result.append(data[id])
        else:
          result.append(data[id].get(key))

      return result


class CMRC_QADataSet(Dataset):
  def __init__(self, data_file, is_train, mode="train"):
    self.tokenizer = tokenizer = FullTokenizer(
      vocab_file=cf.vocab_file, do_lower_case=cf.do_lower_case)
    self.examples = convert2example(data_file, is_train)
    self.mode = mode
    if self.mode == "debug":
      self.examples = self.examples[:10]
    if config.data_size:
      self.examples = self.examples[:int(len(self.examples) * config.data_size) + 1]
    # self.input_features = convert_examples_to_features(
    #     examples=self.examples,
    #     tokenizer=tokenizer,
    #     max_seq_length=cf.context_length_limit,
    #     doc_stride=cf.doc_stride,
    #     max_query_length=cf.ques_length_limit,
    #     is_training=True,
    #     output_fn=None)
    self.input_features = convert2feature(self.examples, tokenizer)
    self.input_features = np.asarray(self.input_features, dtype=np.object)
    self.features_size = len(self.input_features)
    self.idx = [i for i in range(self.features_size)]

  def shuffle(self):
    """打乱数据顺序"""
    random.shuffle(self.idx)

  def covert_features4model_input(self, features, index):
    """"""
    input_ids = []
    input_mask = []
    input_span_mask = []
    segment_ids = []
    start_positions = []
    end_positions = []
    for feature in features:
      input_ids.append(torch.unsqueeze(feature.input_ids, 0))
      input_mask.append(torch.unsqueeze(feature.input_mask, 0))
      input_span_mask.append(torch.unsqueeze(feature.input_span_mask, 0))
      segment_ids.append(torch.unsqueeze(feature.segment_ids, 0))
      start_positions.append(feature.start_position)
      end_positions.append(feature.end_position)

    input_ids = torch.cat(input_ids, dim=0).long()
    input_mask = torch.cat(input_mask, dim=0).long()
    input_span_mask = torch.cat(input_span_mask, dim=0).float()
    segment_ids = torch.cat(segment_ids, dim=0).long()
    start_positions = torch.cat(start_positions, dim=0).long()
    end_positions = torch.cat(end_positions, dim=0).long()
    # index = to_torch_tensor(index)

    return input_ids, input_mask, input_span_mask, segment_ids, start_positions, end_positions, index


  def get(self, item, batch_size=1):
    # index = self.idx[(item * self.batch_size) % self.data_szie: ((item + 1) * self.batch_size) % self.data_szie]
    # TODO: return trainable embedding.
    if batch_size > len(self.idx):
      index = self.idx[(item * batch_size) % self.features_size:]
    else:
      start_idx = (item * batch_size) % self.features_size
      end_idx = ((item + 1) * batch_size) % self.features_size
      if start_idx < end_idx:
        index = self.idx[start_idx:end_idx]
      else:
        index = self.idx[start_idx:] + self.idx[:end_idx]

    features = self.input_features[index]
    return self.covert_features4model_input(features, index)


  def convert_predict_values(self, features, predict_start, predict_end, probability):
    """
    '[SEP]'
    Args:
      features:
      predict_start:
      predict_end:

    Returns:

    """
    tokens = features.tokens
    question = "".join(tokens[:tokens.index("[SEP]")])
    context = "".join(tokens[tokens.index("[SEP]"):])
    label_answer = "".join(tokens[features.start_position:features.end_position])
    predict_answer = "".join(tokens[predict_start:predict_end])
    return {
      "context": context,
      "question": question,
      "label_answer":label_answer,
      "predict_answer": predict_answer,
      "probability": probability
    }

  def convert_predict_values_with_feature_index(self, feature_index, predict_start, predict_end, probability):
    """

    Args:
      features:
      predict_start:
      predict_end:

    Returns:

    """
    feature = self.input_features[feature_index]
    tokens = feature.tokens
    question = "".join(tokens[:tokens.index("[SEP]")])
    context = "".join(tokens[tokens.index("[SEP]"):])
    label_answer = "".join(tokens[feature.start_position:feature.end_position + 1])
    predict_answer = "".join(tokens[predict_start:predict_end + 1])
    return {
      "context": context,
      "question": question,
      "label_answer":label_answer,
      "predict_answer": predict_answer,
      "is_correct": label_answer==predict_answer,
      "probability": float(probability)
    }

  def convert_predict_values_with_batch_feature_index(self, feature_indexs, predict_starts, predict_ends, probabilities):
    """

    Args:
      features:
      predict_start:
      predict_end:

    Returns:

    """
    res = []
    for feature_index, predict_start, predict_end, property in zip(feature_indexs, predict_starts, predict_ends, probabilities):
      res.append(self.convert_predict_values_with_feature_index(feature_index, predict_start, predict_end, property))
    return res

  def convert_all_features4human_visual(self):
    result = []
    for i in range(self.features_size):
      result.append(self.convert_predict_values_with_feature_index(i, 0, 0, 0))
    return result



