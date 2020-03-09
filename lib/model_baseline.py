# -*- coding: utf-8 -*-
# Copyright .
# Author:
#
# cython: language_level=3
#

import math
import torch
from lib.handler import load_bert
import lib.config as cf
from lib.utils import reshape_tensor

class Attention(torch.nn.Module):
  """
  Attention
  """
  def __int__(self, dim, attention_head_num, attention_probs_dropout_prob,
              use_bias=False):
    self.dim = dim
    self.attention_head_num = attention_head_num
    self.use_bias = use_bias
    self.dropout = torch.nn.Dropout(attention_probs_dropout_prob)
    if not self.dim % self.attention_head_num == 0:
      raise Exception(f"The dim({self.dim}) % attention_head_num({self.attention_head_num}) != 0")
    self.size_per_head = self.dim / self.attention_head_num
    self.query_layer = torch.nn.Linear(self.dim, self.dim, self.use_bias)
    self.key_layer = torch.nn.Linear(self.dim, self.dim, self.use_bias)
    self.value_layer = torch.nn.Linear(self.dim, self.dim, self.use_bias)
    self.softmax = torch.nn.Softmax(dim=3)

  def transpose4score(self, tensor, shape):
    """
    为计算 score 对 tensor 进行转换.
    Args:
      tensor:
      shape:

    Returns:

    """
    tensor = reshape_tensor(tensor, shape)
    tensor = tensor.permute(0, 2, 1, 3)
    return tensor

  def forward(self, query_tensor, value_tensor, attention_mask=None):
    """"""
    batch_size, quert_length, _, _ = query_tensor.shape
    _, value_length, _, _ = value_tensor.shape

    query_tensor = reshape_tensor((-1, self.dim))
    value_tensor = reshape_tensor((-1, self.dim))
    query_tensor = self.query_layer(query_tensor)
    key_tensor = self.key_layer(value_tensor)
    value_tensor = self.value_layer(value_tensor)

    query_tensor = self.transpose4score(query_tensor, (batch_size, quert_length,
                                                       self.attention_head_num, self.size_per_head))
    key_tensor  =self.transpose4score(key_tensor, (batch_size, value_length, self.attention_head_num, self.size_per_head))
    attention_scores =  torch.mm(query_tensor, key_tensor.permute(0, 1, 3, 2))
    # batch_size, attention_head_num, query_length, value_length
    attention_scores = attention_scores * (1 / math.sqrt(float(self.size_per_head)))

    if attention_mask:
      attention_mask = torch.unsqueeze(attention_mask, 1)
      attention_mask = torch.unsqueeze(attention_mask, 1)
      attention_mask = attention_mask.expand(batch_size, self.attention_head_num, quert_length, value_length)
      attention_scores = attention_scores * attention_mask

    attention_scores = self.softmax(attention_scores)
    attention_scores = self.dropout(attention_scores)

    value_tensor = reshape_tensor(value_tensor, (batch_size, value_length,
                                                 self.attention_head_num, self.size_per_head))

    value_tensor = value_tensor.permute(0, 2, 1, 3)

    # batch_size, attention_head_num, query_length, size_per_head
    attention = attention_scores * value_tensor

    attention = attention.permute(0, 2, 1, 3)
    attention = reshape_tensor(attention, (batch_size, quert_length, self.dim))

    return attention





class ModelBaseline(torch.nn.Module):
  """
  The model of baseline.
  """
  def __init__(self, bert_path, device, dropout, use_position_embedding=True,
               max_postion=cf.max_postion, pos_dim=cf.bert_dim,
               encoder_hidden_layers=cf.encoder_hidden_layers,
               encoder_intermediate_dim=cf.encoder_intermediate_dim,
               encoder_dropout_prob=cf.encoder_dropout_prob,
               attention_head_num=cf.num_heads,
               attention_probs_dropout_prob=cf.attention_probs_dropout_prob,
               attention_use_bias=cf.attention_use_bias):
    """"""
    # embedding
    self.bert = load_bert(bert_path, device)
    self.dropout = torch.nn.Dropout(dropout)
    self.layer_normal = torch.nn.LayerNorm(max_postion * pos_dim)
    self.use_position_embedding = use_position_embedding
    self.encoder_hidden_layers = encoder_hidden_layers
    if self.use_position_embedding:
      self.init_positon_embedding(max_postion, pos_dim)

    self.attention_layer = Attention(pos_dim, attention_head_num, attention_probs_dropout_prob, attention_use_bias)

    # encoder
    self.encoder_linear_1 = torch.nn.ModuleList([torch.nn.Linear(self.dim, self.dim)
                                                 for i in range(self.encoder_hidden_layers)])
    self.encoder_line_intermidia = torch.nn.ModuleList([torch.nn.Linear(self.dim, encoder_intermediate_dim)
                                                        for i in range(self.encoder_hidden_layers)])
    self.encoder_line_2 = torch.nn.ModuleList([torch.nn.Linear(encoder_intermediate_dim, self.dim)
                                               for i in range(self.encoder_hidden_layers)])
    self.encoder_normal = torch.nn.ModuleList([torch.nn.LayerNorm() for _ in range(self.encoder_hidden_layers)])


  def init_positon_embedding(self, max_postion, pos_dim):
    posi_embedding = torch.Tensor(max_postion, pos_dim)
    posi_embedding = torch.nn.init.kaiming_normal(posi_embedding, a=0, mode='fan_in', nonlinearity='leaky_relu')
    self.position_embedding = torch.nn.Parameter(posi_embedding)


  def embedding(self, input_ids, segment_ids):
    """
    Embedding for input.
    Args:
      input_ids:
      segment_ids:

    Returns:

    """
    embeddings, _ = self.bert(input_ids, segment_ids)
    if self.use_position_embedding:
      embeddings = embeddings + self.position_embedding
    embeddings = self.dropout(embeddings)
    embeddings = self.layer_normal(embeddings)
    return embeddings

  def encoder(self, embeddings, input_mask):
    prelayer_output = embeddings
    for index in range(self.encoder_hidden_layers):
      embeddings = self.attention_layer(embeddings, embeddings, input_mask)
      embeddings = self.encoder_linear_1[index](embeddings)
      embeddings = self.encoder_line_intermidia[index](embeddings)
      embeddings = self.encoder_line_2[index](embeddings)







    



  def forward(self, input_ids, input_mask, segment_ids):
    embedding = self.embedding(input_ids, segment_ids)
