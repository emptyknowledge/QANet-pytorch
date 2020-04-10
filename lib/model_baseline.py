# -*- coding: utf-8 -*-
# Copyright .
# Author:
#
# cython: language_level=3
#

import math
import torch
from lib.handler import load_bert, get_vocab_size
import lib.config as cf
from lib.utils import reshape_tensor, mask, find_max_proper_batch
from torch.nn import functional
from my_py_toolkit.file.file_toolkit import readjson


class LocalLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(LocalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Attention(torch.nn.Module):
  """
  Attention
  """
  def __init__(self, dim, attention_head_num, attention_probs_dropout_prob,
              use_bias=False):
    super(Attention, self).__init__()
    self.dim = dim
    self.attention_head_num = attention_head_num
    self.use_bias = use_bias
    self.dropout = torch.nn.Dropout(attention_probs_dropout_prob)
    if not self.dim % self.attention_head_num == 0:
      raise Exception(f"The dim({self.dim}) % attention_head_num({self.attention_head_num}) != 0")
    self.size_per_head = int(self.dim / self.attention_head_num)
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
    batch_size, quert_length, _ = query_tensor.shape
    _, value_length, _ = value_tensor.shape

    query_tensor = reshape_tensor(query_tensor, (-1, self.dim))
    value_tensor = reshape_tensor(value_tensor, (-1, self.dim))
    query_tensor = self.query_layer(query_tensor)
    key_tensor = self.key_layer(value_tensor)
    value_tensor = self.value_layer(value_tensor)

    query_tensor = self.transpose4score(query_tensor, (batch_size, quert_length,
                                                       self.attention_head_num, self.size_per_head))
    key_tensor  =self.transpose4score(key_tensor, (batch_size, value_length, self.attention_head_num, self.size_per_head))
    attention_scores =  torch.matmul(query_tensor, key_tensor.permute(0, 1, 3, 2))
    # batch_size, attention_head_num, query_length, value_length
    attention_scores = attention_scores * (1 / math.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
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
    attention = torch.matmul(attention_mask, value_tensor)

    attention = attention.permute(0, 2, 1, 3)
    attention = reshape_tensor(attention, (batch_size, quert_length, self.dim))

    return attention


class LocalBert(torch.nn.Module):
  def __init__(self, bert_config, use_segments_embedding=False):
    super(LocalBert, self).__init__()
    self.bert_config = readjson(bert_config)
    self.vocab_size = self.bert_config.get("vocab_size")
    self.type_vocab_size = self.bert_config.get("type_vocab_size")
    self.embedding_dim = self.bert_config.get("hidden_size")
    self.use_segments_embedding = use_segments_embedding
    self.word_embeddings = torch.nn.Parameter(torch.Tensor(self.vocab_size, self.embedding_dim))
    if self.use_segments_embedding:
      self.segments_embedding = torch.nn.Parameter(torch.Tensor(self.type_vocab_size, self.embedding_dim) )

    self.init_para()


  def init_para(self):
    self.word_embeddings = torch.nn.init.kaiming_normal(self.word_embeddings, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
    self.segments_embedding = torch.nn.init.kaiming_normal(self.segments_embedding, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')


  def forward(self, input_ids, segment_ids):
    """"""
    batch_size, sqe_length = input_ids.shape
    input_ids = reshape_tensor(input_ids, [-1])
    segment_ids = reshape_tensor(segment_ids, [-1])
    word_embedding = self.word_embeddings[input_ids]
    segment_embedding = self.segments_embedding[segment_ids]
    word_embedding = word_embedding + segment_embedding
    word_embedding = reshape_tensor(word_embedding, [batch_size, sqe_length, -1])
    return (word_embedding, None)









class ModelBaseLine(torch.nn.Module):
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
               attention_use_bias=cf.attention_use_bias,
               training=True,
               use_pretrained_bert=cf.use_pretrained_bert):
    """"""
    super(ModelBaseLine, self).__init__()
    self.training = training
    # embedding
    self.dim = pos_dim
    self.bert = load_bert(bert_path, device, use_pretrained_bert, cf.bert_config, cf.use_segment_embedding, LocalBert)
    self.dropout = torch.nn.Dropout(dropout)
    self.layer_normal = torch.nn.LayerNorm(pos_dim)
    self.use_position_embedding = use_position_embedding
    self.encoder_hidden_layers = encoder_hidden_layers
    if self.use_position_embedding:
      self.init_positon_embedding(max_postion, pos_dim)

    self.attention_layer = Attention(pos_dim, attention_head_num, attention_probs_dropout_prob, attention_use_bias)

    # encoder
    self.encoder_dropout_prob = encoder_dropout_prob
    self.encoder_linear_1 = torch.nn.ModuleList([torch.nn.Linear(self.dim, self.dim)
                                                 for i in range(self.encoder_hidden_layers)])
    self.encoder_line_intermidia = torch.nn.ModuleList([torch.nn.Linear(self.dim, encoder_intermediate_dim)
                                                        for i in range(self.encoder_hidden_layers)])
    self.encoder_line_2 = torch.nn.ModuleList([torch.nn.Linear(encoder_intermediate_dim, self.dim)
                                               for i in range(self.encoder_hidden_layers)])
    
    self.encoder_normal = torch.nn.ModuleList([torch.nn.LayerNorm(pos_dim) for _ in range(self.encoder_hidden_layers)])

    # pointer
    self.pointer_linear = torch.nn.Linear(self.dim, 2)
    # self.pointer_softmax = torch.nn.Softmax(dim=-2)


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
      # batchsize, sequence_length, posi_duim
      embeddings = self.attention_layer(embeddings, embeddings, input_mask)
      embeddings = self.encoder_linear_1[index](embeddings)
      embeddings = self.encoder_line_intermidia[index](embeddings)
      embeddings = self.encoder_line_2[index](embeddings)
      embeddings += prelayer_output
      # todo: dropout、 normal
      embeddings = functional.dropout(embeddings, self.encoder_dropout_prob, self.training)
      embeddings = self.encoder_normal[index](embeddings)
      prelayer_output = embeddings
    return embeddings




  def pointer(self, embeddings, input_mask):
    """"""
    # size: batch_size, seq_length, 2
    embeddings = self.pointer_linear(embeddings)
    embeddings = mask(embeddings, input_mask, -2)
    start_embeddings = embeddings[:, :, 0].squeeze(dim=-1)
    end_embeddings = embeddings[:, :, 1].squeeze(dim=-1)
    return start_embeddings, end_embeddings

    # embeddings = self.pointer_softmax(embeddings)
    # start_softmax = embeddings[:,:,0]
    # end_softmax = embeddings[:,:,1]
    # start, end, pro = find_max_proper_batch(start_softmax, end_softmax)
    # return start, end, pro


  def forward(self, input_ids, input_mask, segment_ids):
    embedding = self.embedding(input_ids, segment_ids)
    embedding = self.encoder(embedding, input_mask)
    start, end = self.pointer(embedding, input_mask)
    return start, end

    
if __name__ == "__main__":
  input = torch.Tensor([[1,23,3],[4,5,6]]).long()
  segment = torch.Tensor([[1,1,0], [1,0,1]]).long()
  local_bert  = LocalBert(r"F:\Study\Github\QANet-pytorch\data\model\bert\config.json",
                          True)
  local_bert(input, segment)