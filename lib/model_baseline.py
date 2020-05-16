# -*- coding: utf-8 -*-
# Copyright .
# Author:
#
# cython: language_level=3
#

import copy
import json
import six
import math
import torch
from lib.handler import load_bert, get_vocab_size
import lib.config as cf
from lib.utils import reshape_tensor, mask, find_max_proper_batch, gelu
from torch.nn import functional
from my_py_toolkit.file.file_toolkit import readjson



class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    return cls.from_dict(readjson(json_file))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

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

    def __init__(self, in_features, out_features, bias=False):
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
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_out')
        if self.bias is not None:
          # torch.nn.init.kaiming_normal_(self.bias, mode='fan_out')
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
    self.softmax = torch.nn.Softmax(dim=-1)

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
                                                       self.attention_head_num,
                                                       self.size_per_head))
    key_tensor = self.transpose4score(key_tensor, (batch_size, value_length,
                                                   self.attention_head_num,
                                                   self.size_per_head))
    attention_scores = torch.matmul(query_tensor, key_tensor.permute(0, 1, 3, 2))
    # batch_size, attention_head_num, query_length, value_length
    attention_scores = attention_scores / math.sqrt(float(self.size_per_head))

    if attention_mask is not None:
      # batch_size, 1, sqe_len
      attention_mask = torch.unsqueeze(attention_mask, 1)
      # batch_size, 1, sqe_len, 1
      attention_mask = torch.unsqueeze(attention_mask, -1)
      # batch_size, attention_head_num, squ_len
      attention_mask = attention_mask.expand(batch_size, self.attention_head_num, quert_length, value_length)
      attention_scores = attention_scores * attention_mask

    attention_scores = self.softmax(attention_scores)
    # attention_scores = self.dropout(attention_scores)

    value_tensor = reshape_tensor(value_tensor, (batch_size, value_length,
                                                 self.attention_head_num, self.size_per_head))

    value_tensor = value_tensor.permute(0, 2, 1, 3)
    attention = torch.matmul(attention_scores, value_tensor)

    # batch_size, attention_head_num, query_length, size_per_head
    # attention = torch.matmul(attention_mask, value_tensor)

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
    torch.nn.init.kaiming_normal_(self.word_embeddings, mode='fan_out')
    torch.nn.init.kaiming_normal_(self.segments_embedding, mode='fan_out')
    # self.word_embeddings = torch.nn.init.kaiming_normal(self.word_embeddings, mode='fan_out', nonlinearity)
    # self.segments_embedding = torch.nn.init.kaiming_normal(self.segments_embedding, a=math.sqrt(5))
    # self.word_embeddings = torch.nn.init.uniform_(self.word_embeddings, a=-0.02, b=0.02)
    # self.segments_embedding = torch.nn.init.uniform_(self.segments_embedding, a=-0.02, b=0.02)


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
    self.layer_normal = torch.nn.LayerNorm([max_postion, pos_dim])
    self.use_position_embedding = use_position_embedding
    self.encoder_hidden_layers = encoder_hidden_layers
    if self.use_position_embedding:
      self.init_positon_embedding(max_postion, pos_dim)

    # encoder
    self.attention_layer = torch.nn.ModuleList([
      Attention(pos_dim, attention_head_num, attention_probs_dropout_prob, attention_use_bias)
      for i in range(self.encoder_hidden_layers)
    ])
    self.encoder_dropout_prob = encoder_dropout_prob
    self.encoder_linear_1 = torch.nn.ModuleList([torch.nn.Linear(self.dim, self.dim)
                                                 for i in range(self.encoder_hidden_layers)])
    self.encoder_line_intermidia = torch.nn.ModuleList([torch.nn.Linear(self.dim, encoder_intermediate_dim)
                                                        for i in range(self.encoder_hidden_layers)])
    self.encoder_line_2 = torch.nn.ModuleList([torch.nn.Linear(encoder_intermediate_dim, self.dim)
                                               for i in range(self.encoder_hidden_layers)])
    
    self.encoder_normal = torch.nn.ModuleList([torch.nn.LayerNorm([max_postion, pos_dim]) for _ in range(self.encoder_hidden_layers)])

    # pointer
    self.pointer_linear = torch.nn.Linear(self.dim, 2)
    # self.pointer_softmax = torch.nn.Softmax(dim=-2)


  def init_positon_embedding(self, max_postion, pos_dim):
    posi_embedding = torch.Tensor(max_postion, pos_dim)
    # posi_embedding = torch.nn.init.kaiming_normal(posi_embedding, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
    self.position_embedding = torch.nn.Parameter(posi_embedding)
    torch.nn.init.kaiming_normal_(self.position_embedding, mode='fan_out')


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
    embeddings = self.layer_normal(embeddings)
    # embeddings = self.dropout(embeddings)
    return embeddings

  def encoder(self, embeddings, input_mask):
    prelayer_output = embeddings
    for index in range(self.encoder_hidden_layers):
      # batchsize, sequence_length, posi_duim
      embeddings = self.attention_layer[index](embeddings, embeddings, input_mask)
      embeddings = self.encoder_linear_1[index](embeddings)
      embeddings = torch.relu(embeddings)
      embeddings = self.encoder_line_intermidia[index](embeddings)
      # embeddings = gelu(embeddings)
      embeddings = torch.relu(embeddings)
      embeddings = self.encoder_line_2[index](embeddings)
      embeddings = torch.relu(embeddings)
      embeddings += prelayer_output
      # todo: dropout、 normal
      embeddings = self.encoder_normal[index](embeddings)
      # embeddings = functional.leaky_relu(embeddings)
      # embeddings = functional.dropout(embeddings, self.encoder_dropout_prob, self.training)
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