# -*- coding: utf-8 -*-
# Copyright .
# Author:
#
# cython: language_level=3
#

import torch
from lib.handler import load_bert
import lib.config as cf
from lib.utils import reshape_tensor

class ModelBaseline(torch.nn.Module):
  """
  The model of baseline.
  """
  def __init__(self, bert_path, device, dropout, use_position_embedding=True,
               max_postion=cf.max_postion, pos_dim=cf.bert_dim):
    """"""
    # embedding
    self.bert = load_bert(bert_path, device)
    self.dropout = torch.nn.Dropout(dropout)
    self.layer_normal = torch.nn.LayerNorm(max_postion * pos_dim)
    self.use_position_embedding = use_position_embedding
    if self.use_position_embedding:
      self.init_positon_embedding(max_postion, pos_dim)

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

  def encoder(self, embeddings, input_mask, attention_heads):
    pass

  def attention(self, from_tensor, to_tensor, input_mask, attention_head_num):
    """"""
    # batch_size, context_length, dim
    batch_size, f_context_length, dim = from_tensor.shape
    t_context_length = to_tensor.shape[1]
    if dim % attention_head_num != 0:
      raise Exception("attention heads 应该被 dim 整除")

    head_dim = dim / attention_head_num
    from_tensor = reshape_tensor(from_tensor, [batch_size, f_context_length,
                                               attention_head_num, head_dim])
    to_tensor = reshape_tensor(to_tensor, [batch_size, t_context_length,
                                           attention_head_num, head_dim])

    torch.nn.Linear
    attention_score = torch.mm(from_tensor, to_tensor.permute(0,1,3,2))





    



  def forward(self, input_ids, input_mask, segment_ids):
    embedding = self.embedding(input_ids, segment_ids)
