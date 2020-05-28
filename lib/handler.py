# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author:
#
# cython: language_level=3
#

import json
import re
import os
import numpy as np
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
from tqdm import tqdm
import collections
from lib.tokenizations import official_tokenization as tokenization

SPIECE_UNDERLINE = '▁'

def evaluate_valid_result(valid_result, exact_match_total=0, f1_total=0, total=0):
  # f1 = exact_match = total = 0
  for item in valid_result:
    total += 1
    ground_truths = item.get("label_answer")
    prediction = item.get("predict_answer")
    exact_match_total += metric_max_over_ground_truths(exact_match_score, prediction,
                                                       ground_truths)
    f1_total += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
  exact_match = 100.0 * exact_match_total / total
  f1 = 100.0 * f1_total / total
  return exact_match_total, f1_total, exact_match, f1



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

def load_bert(model_path, device, use_pretrained_bert=True, bert_conf=None,
              use_segments_embeddings=True, local_bert_class=None):
  """
  加载与训练的 bert.
  Args:
    model_path:
    device:

  Returns:

  """
  if use_pretrained_bert:
    return BertModel.from_pretrained(model_path).to(device)
  else:
    return local_bert_class(bert_conf, use_segments_embeddings).to(device)

@fn_timer(logger)
def get_model(package, name, class_name):
  """
  Gets models.
  Returns:

  """
  model_class = load_class(package, name, class_name)
  if not config.is_continue:
    return model_class(config.bert_path, config.device, config.dropout)
  else:
    logger.info(f"Continue train, continue_checkpoint: {config.continue_checkpoint}")
    model_path = os.path.join(config.model_dir,
                              f"model_{config.continue_checkpoint}.pkl")
    if not config.is_only_save_params:
      model = torch.load(model_path, map_location=config.device)
    else:
      model = model_class(config.bert_path, config.device, config.dropout)
      model.load_state_dict(torch.load(model_path))
    return model


@fn_timer(logger)
def get_optimizer(base_lr, params):
  """
  Gets models.
  Returns:

  """
  if not config.is_continue:
    return torch.optim.Adam(lr=base_lr, betas=(config.beta1, config.beta2),
                         eps=1e-8, weight_decay=3e-7, params=params)
  else:
    logger.info(f"Continue train, continue_checkpoint: {config.continue_checkpoint}")
    optimizer_path = os.path.join(config.model_dir,
                              f"optimizer_{config.continue_checkpoint}.pkl")
    if not config.is_only_save_params:
      optimizer = torch.load(optimizer_path, map_location=config.device)
    else:
      optimizer = torch.optim.Adam(lr=base_lr, betas=(config.beta1, config.beta2),
                         eps=1e-8, weight_decay=3e-7, params=params)
      optimizer.load_state_dict(torch.load(optimizer_path))
    return optimizer
  
def save_model(model, optmizer, steps=0):
  model_path = os.path.join(config.model_dir, f"model_{str(steps)}.pkl")
  optimizer_path = os.path.join(config.model_dir, f"optimizer_{str(steps)}.pkl")
  make_path_legal(model_path)
  make_path_legal(optimizer_path)
  if not config.is_only_save_params:
    torch.save(model, model_path)
    torch.save(optmizer, optimizer_path)
    # torch.save(model.embedding.trainable_embedding,
    #            config.embedding_trainable_model)
    # torch.save(data_set.trainable_embedding, config.embedding_trainable_model)
  else:
    torch.save(model.state_dict(), model_path)
    torch.save(optmizer.state_dict(), optimizer_path)
    # torch.save(model.embedding.trainable_embedding,
    #            config.embedding_trainable_model)
    # torch.save(data_set.trainable_embedding, config.embedding_trainable_model)

@fn_timer(logger)
def get_dataset(data_type="train", mode="train"):
  """

  Args:
    data_type(str): train, dev, trial

  Returns:

  """
  data_dir = config.dataset_dir.get(config.dataset_name)
  data_path = [path for path in get_file_paths(data_dir) if data_type in get_file_name(path)][0]
  if "cmcr" in config.dataset_name:
    dataset = CMRC_QADataSet(data_path,
                             config.is_train, mode)
    return dataset
  else:
    dataset = QADataSet(data_path,
                        config.batch_size, mode)
    return dataset

def record_info(losses=[], f1=[], em=[], valid_result={}, epoch=0,
                r_type="train", is_continue=True):
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
    write2file(",".join(losses), f"{dir_name}losses.txt", is_continue=is_continue)
  if f1:
    write2file(",".join(f1), f"{dir_name}f1.txt", is_continue=is_continue)
  if em:
    write2file(",".join(em), f"{dir_name}em.txt", is_continue=is_continue)

  if valid_result:
    write2file(json.dumps(valid_result, ensure_ascii=False, indent=2),
               f"{dir_name}valid_result_{epoch}.json", is_continue=is_continue)


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


def get_vocab_size(vocab_file):
  with open(vocab_file, "r", encoding="utf-8") as f:
    context = f.read()
    return len(context.split("\n"))

def is_leaf(value):
  keys = "".join([k for k,_ in value.items()])
  if "." in keys:
    return False
  else:
    return True

def process_leaf(value):
  value = [v for k,v in value.items()]
  value = [reshape_tensor(v,[-1]) for v in value if v is not None]
  if value:
    value = torch.cat(value, dim=0)
  return value

def handler_gradient(gradient):
  result ={}
  leaf = True
  for k,v in gradient.items():
    if not isinstance(v, dict):
      result[k] = v
      continue

    if isinstance(list(v.values())[0], dict):
      v = handler_gradient(v)
      leaf = False

    if leaf and is_leaf(v):
      v = process_leaf(v)
      result[k] = v
    else:
      for sub_key, sub_v in v.items():
        result[f"{k}.{sub_key}"] = sub_v

  return result

def is_last_layer(value):
  if not isinstance(value, dict):
    return True
  for k,v in value.items():
    if isinstance(v, dict):
      return False
  return True

def transfer_multi_layer_dict(dict_value):
  """
  将多层 dict 转换为 1 层 dict.
  Args:
    dict_value:

  Returns:

  """
  result = {}
  for key, value in dict_value.items():
    if not isinstance(value, dict):
      result[key] = value
      continue


    if not is_last_layer(value):
      value = transfer_multi_layer_dict(value)

    for sub_key, sub_value in value.items():
      result[f"{key}.{sub_key}"] = sub_value

  return result


def covert_position_embeddings(position_embeddings, predict, log_softmax):
  for index, v in enumerate(predict):
    position_embeddings[index][v] += 1
  position_embeddings = log_softmax(log_softmax)
  return position_embeddings

def convert_one_hot(input_array, class_num):
  result = torch.zeros((len(input_array), class_num))
  for index, v in enumerate(input_array):
    result[index][v] = 1
  return result

def truncated_normal_(tensor, mean=0, std=0.09):
  with torch.no_grad():
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def json2features(input_file, output_files, tokenizer, is_training=False,
                  repeat_limit=3, max_query_length=64,
                  max_seq_length=512, doc_stride=128):
  with open(input_file, 'r') as f:
    train_data = json.load(f)
    train_data = train_data['data']

  def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
      (cp >= 0x3400 and cp <= 0x4DBF) or  #
      (cp >= 0x20000 and cp <= 0x2A6DF) or  #
      (cp >= 0x2A700 and cp <= 0x2B73F) or  #
      (cp >= 0x2B740 and cp <= 0x2B81F) or  #
      (cp >= 0x2B820 and cp <= 0x2CEAF) or
      (cp >= 0xF900 and cp <= 0xFAFF) or  #
      (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def is_fuhao(c):
    if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
      or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
      or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
      or c == '‘' or c == '’':
      return True
    return False

  def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if _is_chinese_char(cp) or is_fuhao(char):
        if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
          output.append(SPIECE_UNDERLINE)
        output.append(char)
        output.append(SPIECE_UNDERLINE)
      else:
        output.append(char)
    return "".join(output)

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(
      c) == 0x202F or c == SPIECE_UNDERLINE:
      return True
    return False

  # to examples
  examples = []
  mis_match = 0
  for article in tqdm(train_data):
    for para in article['paragraphs']:
      context = para['context']
      context_chs = _tokenize_chinese_chars(context)
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in context_chs:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        if c != SPIECE_UNDERLINE:
          char_to_word_offset.append(len(doc_tokens) - 1)

      for qas in para['qas']:
        qid = qas['id']
        ques_text = qas['question']
        ans_text = qas['answers'][0]['text']

        start_position_final = None
        end_position_final = None
        if is_training:
          count_i = 0
          start_position = qas['answers'][0]['answer_start']

          end_position = start_position + len(ans_text) - 1
          while context[
                start_position:end_position + 1] != ans_text and count_i < repeat_limit:
            start_position -= 1
            end_position -= 1
            count_i += 1

          while context[start_position] == " " or context[
            start_position] == "\t" or \
            context[start_position] == "\r" or context[
            start_position] == "\n":
            start_position += 1

          start_position_final = char_to_word_offset[start_position]
          end_position_final = char_to_word_offset[end_position]

          if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".",
                                                  ","}:
            start_position_final += 1

          actual_text = "".join(
            doc_tokens[start_position_final:(end_position_final + 1)])
          cleaned_answer_text = "".join(
            tokenization.whitespace_tokenize(ans_text))

          if actual_text != cleaned_answer_text:
            print(actual_text, 'V.S', cleaned_answer_text)
            mis_match += 1
            # ipdb.set_trace()

        examples.append({'doc_tokens': doc_tokens,
                         'orig_answer_text': ans_text,
                         'qid': qid,
                         'question': ques_text,
                         'answer': ans_text,
                         'start_position': start_position_final,
                         'end_position': end_position_final})

  print('examples num:', len(examples))
  print('mis_match:', mis_match)
  os.makedirs('/'.join(output_files[0].split('/')[0:-1]), exist_ok=True)
  json.dump(examples, open(output_files[0], 'w'))

  # to features
  features = []
  unique_id = 1000000000
  for (example_index, example) in enumerate(tqdm(examples)):
    query_tokens = tokenizer.tokenize(example['question'])
    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example['doc_tokens']):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training:
      tok_start_position = orig_to_tok_index[
        example['start_position']]  # 原来token到新token的映射，这是新token的起点
      if example['end_position'] < len(example['doc_tokens']) - 1:
        tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
        example['orig_answer_text'])

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    doc_spans = []
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      start_position = None
      end_position = None
      if is_training:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        if tok_start_position == -1 and tok_end_position == -1:
          start_position = 0  # 问题本来没答案，0是[CLS]的位子
          end_position = 0
        else:  # 如果原本是有答案的，那么去除没有答案的feature
          out_of_span = False
          doc_start = doc_span.start  # 映射回原文的起点和终点
          doc_end = doc_span.start + doc_span.length - 1

          if not (
            tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
            out_of_span = True
          if out_of_span:
            start_position = 0
            end_position = 0
          else:
            doc_offset = len(query_tokens) + 2
            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset

      features.append({'unique_id': unique_id,
                       'example_index': example_index,
                       'doc_span_index': doc_span_index,
                       'tokens': tokens,
                       'token_to_orig_map': token_to_orig_map,
                       'token_is_max_context': token_is_max_context,
                       'input_ids': input_ids,
                       'input_mask': input_mask,
                       'segment_ids': segment_ids,
                       'start_position': start_position,
                       'end_position': end_position})
      unique_id += 1

  print('features num:', len(features))
  json.dump(features, open(output_files[1], 'w'))


def convert_pre_res(input_ids, pre_start, pre_end, ori_start, ori_end, probabilities, tokenizer):
  """"""
  result = []
  for input, p_start, p_end, o_start, o_end, probability in zip(input_ids, pre_start, pre_end, ori_start, ori_end, probabilities):
    tokens = tokenizer.convert_ids_to_tokens(input)
    question = "".join(tokens[:tokens.index("[SEP]")])
    context = "".join(tokens[tokens.index("[SEP]"):])
    label_answer = "".join(
      tokens[o_start:ori_end + 1])
    predict_answer = "".join(tokens[p_start:p_end + 1])
    cur_res = {
      "context": context,
      "question": question,
      "label_answer": label_answer,
      "predict_answer": predict_answer,
      "is_correct": label_answer == predict_answer,
      "probability": float(probability)
    }
    result.append(cur_res)
