
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import lib.config as config
import torchsnooper

from lib.handler import *

from lib.Embedding import BertEmbedding

d_model = config.d_model
n_head = config.num_heads
d_word = config.bert_dim
d_char = config.bert_dim
d_trainable_emb = config.embedding_trainable_dim
batch_size = config.batch_size
dropout = config.dropout
dropout_char = config.dropout_char

d_k = d_model // n_head
d_cq = d_model * 4
len_c = config.context_length_limit
len_q = config.ques_length_limit


def mask_logits(target, mask):
  return target * (1 - mask) + mask * (-1e30)


class PosEncoder(nn.Module):
  def __init__(self, length):
    super().__init__()
    freqs = torch.Tensor(
      [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model)
       for i in range(d_model)]).unsqueeze(dim=1)
    phases = torch.Tensor(
      [0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(
      dim=1)
    pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
    self.pos_encoding = nn.Parameter(
      torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

  def forward(self, x):
    x = x.transpose(1,2)
    x = x + self.pos_encoding
    x = x.transpose(1,2)
    return x


class DepthwiseSeparableConv(nn.Module):
  def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
    super().__init__()
    self.dim = dim
    self.in_ch = in_ch
    self.out_ch = out_ch
    if dim == 1:
      self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch,
                                      kernel_size=k, groups=in_ch,
                                      padding=k // 2, bias=bias)
      self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                      kernel_size=1, padding=0, bias=bias)
    elif dim == 2:
      self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
                                      kernel_size=k, groups=in_ch,
                                      padding=k // 2, bias=bias)
      self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                      kernel_size=1, padding=0, bias=bias)
    else:
      raise Exception("Wrong dimension for Depthwise Separable Convolution!")
    nn.init.kaiming_normal_(self.depthwise_conv.weight)
    nn.init.constant_(self.depthwise_conv.bias, 0.0)
    nn.init.kaiming_normal_(self.pointwise_conv.weight)
    nn.init.constant_(self.pointwise_conv.bias, 0.0)

  def forward(self, x):
    # if self.dim == 2:
    #   x = adapt_conv2D(x)
    # else:
    #   x = adapt_normal(x)

    x_shape = list(x.shape)
    if len(x_shape) < 4:
      new_shape = copy.deepcopy(x_shape)
      new_shape.append(1)
      x = x.contiguous().view(* new_shape)
    x = x.transpose(1,2)
    x = self.pointwise_conv(self.depthwise_conv(x))
    x = x.transpose(1,2)
    x_shape[2] = self.out_ch
    x = x.contiguous().view(*x_shape)
    return x


class Highway(nn.Module):
  def __init__(self, layer_num: int, size: int):
    super().__init__()
    self.n = layer_num
    self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
    self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

  def forward(self, x):
    # x = adapt_normal(x)
    # x = x.transpose(1, 2)
    # x_shape = x.shape
    # x = adapt_line(x, x_shape)
    for i in range(self.n):
      # 线性变换
      gate = torch.sigmoid(self.gate[i](x))
      # 非线性变换
      nonlinear = F.relu(self.linear[i](x))
      x = gate * nonlinear + (1 - gate) * x
    # x = x.transpose(1, 2)
    # x = x.view(*x_shape)
    return x


class SelfAttention(nn.Module):
  def __init__(self):
    super().__init__()
    Wo = torch.empty(d_model, d_k * n_head)
    Wqs = [torch.empty(d_model, d_k) for _ in range(n_head)]
    Wks = [torch.empty(d_model, d_k) for _ in range(n_head)]
    Wvs = [torch.empty(d_model, d_k) for _ in range(n_head)]
    nn.init.kaiming_uniform_(Wo)
    for i in range(n_head):
      nn.init.xavier_uniform_(Wqs[i])
      nn.init.xavier_uniform_(Wks[i])
      nn.init.xavier_uniform_(Wvs[i])
    self.Wo = nn.Parameter(Wo)
    self.Wqs = nn.ParameterList([nn.Parameter(X) for X in Wqs])
    self.Wks = nn.ParameterList([nn.Parameter(X) for X in Wks])
    self.Wvs = nn.ParameterList([nn.Parameter(X) for X in Wvs])

  def forward(self, x, mask):
    WQs, WKs, WVs = [], [], []
    sqrt_d_k_inv = 1 / math.sqrt(d_k)
    x = x.transpose(1, 2)
    hmask = mask.unsqueeze(1)
    vmask = mask.unsqueeze(2)
    for i in range(n_head):
      WQs.append(torch.matmul(x, self.Wqs[i]))
      WKs.append(torch.matmul(x, self.Wks[i]))
      WVs.append(torch.matmul(x, self.Wvs[i]))
    heads = []
    for i in range(n_head):
      out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
      out = torch.mul(out, sqrt_d_k_inv)
      # not sure... I think `dim` should be 2 since it weighted each column of `WVs[i]`
      out = mask_logits(out, hmask)
      out = F.softmax(out, dim=2) * vmask
      headi = torch.bmm(out, WVs[i])
      heads.append(headi)
    head = torch.cat(heads, dim=2)
    out = torch.matmul(head, self.Wo)
    return out.transpose(1, 2)


class MultiHeadAttention(nn.Module):
  def __init__(self):
    super().__init__()

    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(d_model, d_model)
    self.a = 1 / math.sqrt(d_k)

  def forward(self, x, mask):
    # adapt_normal(x): batch_size, length, dim
    x_shape = list(x.shape)
    x = adapt_normal(x, x_shape)
    x = x.contiguous().view(* x_shape)
    bs, l_x, _ = x_shape
    # x = x.transpose(1, 2)
    k = self.k_linear(x).contiguous().view(bs, l_x, n_head, d_k)
    q = self.q_linear(x).contiguous().view(bs, l_x, n_head, d_k)
    v = self.v_linear(x).contiguous().view(bs, l_x, n_head, d_k)
    q = q.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)
    k = k.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)
    v = v.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)

    # mask shape: bs * n_head, l_x, l_x
    mask = mask.unsqueeze(1).expand(-1, l_x, -1).repeat(n_head, 1, 1)

    # attn shape: bs * n_head, l_x, l_x
    attn = torch.bmm(q, k.transpose(1, 2)) * self.a
    attn = mask_logits(attn, mask)
    attn = F.softmax(attn, dim=2)
    attn = self.dropout(attn)

    out = torch.bmm(attn, v)
    out = out.contiguous().view(n_head, bs, l_x, d_k).permute(1, 2, 0, 3).contiguous().view(
      bs, l_x, d_model)

    # out = adapt_normal(out, x_shape)
    out = self.fc(out)
    out = self.dropout(out)
    out = out.contiguous().view(*x_shape)
    # return out.transpose(1, 2)
    return out


class Embedding(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv2d = DepthwiseSeparableConv(d_char + d_trainable_emb,
                                         d_char + d_trainable_emb,
                                         5, dim=2)
    self.high = Highway(2, d_char + d_trainable_emb)

  def forward(self, wd_emb):
    # ch_emb = ch_emb.permute(0, 3, 1, 2)
    # ch_emb = F.dropout(ch_emb, p=dropout_char, training=self.training)
    # ch_emb = self.conv2d(ch_emb)
    # ch_emb = F.relu(ch_emb)
    # ch_emb, _ = torch.max(ch_emb, dim=3)
    # ch_emb = ch_emb.squeeze()

    # wd_embedding shape: batch_size, length, dim
    wd_emb = F.dropout(wd_emb, p=dropout, training=self.training)
    # wd_emb = wd_emb.transpose(1, 2)
    wd_emb = self.conv2d(wd_emb)
    # emb = wd_emb
    # emb = torch.squeeze(emb, -1)
    wd_emb = self.high(wd_emb)
    return wd_emb


class EncoderBlock(nn.Module):
  def __init__(self, conv_num: int, ch_num: int, k: int, length: int):
    super().__init__()
    self.convs = nn.ModuleList(
      [DepthwiseSeparableConv(ch_num, ch_num, k, dim=2) for _ in
       range(conv_num)])
    self.self_att = MultiHeadAttention()
    # fc 的使用有点没看明白
    self.fc = nn.Linear(ch_num, ch_num, bias=True)
    self.pos = PosEncoder(length)
    # self.norm = nn.LayerNorm([d_model, length])
    self.normb = nn.LayerNorm(ch_num * length)
    self.norms = nn.ModuleList(
      [nn.LayerNorm(ch_num * length) for _ in range(conv_num)])
    self.norme = nn.LayerNorm(ch_num * length)
    self.L = conv_num

  def forward(self, x, mask):
    x_shape = list(x.shape)

    out = self.pos(x)

    out = adapt_normal(out, x_shape)
    res = out
    out = self.normb(out)

    for i, conv in enumerate(self.convs):
      out = adapt_conv2D(out, x_shape)
      out = conv(out)
      out = adapt_normal(out, x_shape)
      out = F.relu(out)
      out = out + res
      if (i + 1) % 2 == 0:
        p_drop = dropout * (i + 1) / self.L
        out = F.dropout(out, p=p_drop, training=self.training)
      res = out
      out = self.norms[i](out)
    out = out.contiguous().view(*x_shape)
    # print("Before attention: {}".format(out.size()))
    out = self.self_att(out, mask)
    # print("After attention: {}".format(out.size()))
    out = out + res.contiguous().view(*x_shape)
    out = F.dropout(out, p=dropout, training=self.training)

    res = out
    out = adapt_normal(out, x_shape)
    out = self.norme(out).contiguous().view(*x_shape)
    out = self.fc(out)
    out = F.relu(out)
    out = out + res
    out = F.dropout(out, p=dropout, training=self.training)
    out = out.contiguous().view(*x_shape)
    return out


class CQAttention(nn.Module):
  def __init__(self):
    super().__init__()
    w = torch.empty(d_model * 3)
    lim = 1 / d_model
    nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
    self.w = nn.Parameter(w)

  def forward(self, C, Q, cmask, qmask):
    """

    Args:
      C: shape: batch_size, dim, length
      Q: shape: batch_size, dim, length
      cmask: shape: batch_size, length
      qmask: shape: batch_size, length

    Returns:

    """
    ss = []
    # C = C.transpose(1, 2)  # shape: batch_size, context_length, dim
    # Q = Q.transpose(1, 2)  # shape: batch_size, question_length, dim
    cmask = cmask.unsqueeze(2)  # 升维, shape: batch_size, length, new_dim
    qmask = qmask.unsqueeze(1)  # 升维， shape: batch_size, new_dim, length

    # batch_size, conetxt_length, question_length, dim
    shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
    # C.unsqueeze(2).shape: batch_size, context_length, new_dim(值：1), dim
    # C.unsqueeze(2).expand(shape): 表示  context 的最后一维 dim 变为
    # (qestion_length, dim), 最后一维被复制了 question_length 份
    Ct = C.unsqueeze(2).expand(shape)
    # Q.unsqueeze(1).shape: batch_size, new_dim(值：1), question_length, dim
    # Q.unsqueeze(1).expand(shape): 表示 question 的后两维由 (question_length, dim)
    # 变为 (context_length ,question_length, dim), 表示每个 question 被复制为了
    #  context_length 份
    Qt = Q.unsqueeze(1).expand(shape)
    # CQ 计算点积
    CQ = torch.mul(Ct, Qt)
    # S.shape: batch_size, conetxt_length, question_length, dim * 3
    S = torch.cat([Ct, Qt, CQ], dim=3)
    # s.shape: batch_size, conetxt_length, question_length, 1
    S = torch.matmul(S, self.w)
    # S1 在 question_lenth 这维 soft max， Context 使用, shape: batch_size, conetxt_length, question_length, 1
    S1 = F.softmax(mask_logits(S, qmask), dim=2)
    # S2 在 context_lenth 这维 soft max， Question 使用, shape: batch_size, conetxt_length, question_length, 1
    S2 = F.softmax(mask_logits(S, cmask), dim=1)
    # A context attention. shape: batch_size, context_length, dim
    A = torch.bmm(S1, Q)
    # B question attention。
    # torch.bmm(S1, S2.transpose(1, 2)) shape: batch_size, context_length, context_length
    # B shape: batch_size, context_length, dim
    B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
    # out shape : batch_size, context_length, dim * 4
    out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
    out = F.dropout(out, p=dropout, training=self.training)
    # out shape: batch_size, dim * 4, context_length
    return out


class Pointer(nn.Module):
  def __init__(self):
    super().__init__()
    w1 = torch.empty(d_model * 2)
    w2 = torch.empty(d_model * 2)
    lim = 3 / ( d_model * 2)
    nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
    nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
    self.w1 = nn.Parameter(w1)
    self.w2 = nn.Parameter(w2)

  def forward(self, M1, M2, M3, mask):
    X1 = torch.cat([M1, M2], dim=-1)
    X2 = torch.cat([M1, M3], dim=-1)
    Y1 = torch.matmul(X1, self.w1)
    Y2 = torch.matmul(X2, self.w2)
    Y1 = mask_logits(Y1, mask)
    Y2 = mask_logits(Y2, mask)
    # if self.training:
    #   p1 = F.log_softmax(Y1, dim=1)
    #   p2 = F.log_softmax(Y2, dim=1)
    # else:
    p1 = F.softmax(Y1, dim=1)
    p2 = F.softmax(Y2, dim=1)
    return p1, p2


class ContextConv(nn.Module):
  def __init__(self, input_channel, out_channel, kernel_szie=5):
    super().__init__()
    self.convs = nn.ModuleList()
    self.input_channel = input_channel
    self.out_channel = out_channel
    channel = input_channel
    while channel > 2 * out_channel:
      # 查查原因，为什么这里需要 to(device) ，否则就会异常？
      # 使用 nn.ModuleList 就不需要单独加 to(device)
      self.convs.append(DepthwiseSeparableConv(channel,
                                               channel // 2,
                                               kernel_szie,
                                               dim=2))
      channel = channel // 2

    self.convs.append(DepthwiseSeparableConv(channel,
                                             out_channel,
                                             kernel_szie,
                                             dim=2))

  def forward(self, x):
    # x_shape = list(x.shape)
    # if len(x_shape) < 4:
    #   new_shape = copy.deepcopy(x_shape)
    #   new_shape.append(1)
    #   x.view(*new_shape)
    #   x = x.transpose(1,2)
    for conv in self.convs:
      x = conv(x)
    # x = x.transpose(1,2) 
    return x


class QANet(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding = BertEmbedding(config.bert_path)
    self.trainable_embedding = get_trainable_embedding(
      self.embedding.vocab_size)
    self.emb = Embedding()
    # self.emb_conetxt = Embedding(config.context_length_limit)
    # self.emb_question = Embedding(config.ques_length_limit)
    self.context_conv = ContextConv(d_word + d_trainable_emb, d_model,
                                    config.context_kernel_size)
    self.question_conv = ContextConv(d_word + d_trainable_emb, d_model,
                                     config.question_kernel_size)
    self.c_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_c)
    self.q_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_q)
    self.cq_att = CQAttention()
    self.cq_resizer = DepthwiseSeparableConv(d_model * 4,
                                             d_model,
                                             5,
                                             dim=2)
    enc_blk = EncoderBlock(conv_num=2, ch_num=d_model, k=5, length=len_c)
    self.model_enc_blks = nn.ModuleList([enc_blk] * 7)
    self.out = Pointer()

  # @torchsnooper.snoop()
  def forward(self, Cwid, Qwid):
    # 输入数据有 pad （即补 0 使数据长度一致, mask 拿到补 0 的位置信息）
    cmask = (torch.zeros_like(Cwid) == Cwid).float()
    qmask = (torch.zeros_like(Qwid) == Qwid).float()
    Cw = self.embedding.word_embedding(Cwid)
    Qw = self.embedding.word_embedding(Qwid)
    Cw_trainable = self.trainable_embedding.data[Cwid]
    Qw_trainable = self.trainable_embedding.data[Qwid]
    Cw = torch.cat((Cw, Cw_trainable), -1)
    Qw = torch.cat((Qw, Qw_trainable), -1)
    # C,Q emb 后 shape: batch_size, length, embedding_dim
    C, Q = self.emb(Cw), self.emb(Qw)
    # C, Q = self.emb_conetxt(Cw), self.emb_question(Qw)
    # C,Q  context_conv 后 shape: batch_size, length, d_model
    C = self.context_conv(C)
    Q = self.question_conv(Q)
    # Ce, Qe shape:  batch_size, length, d_model
    Ce = self.c_emb_enc(C, cmask)
    Qe = self.q_emb_enc(Q, qmask)

    # Attention,  X shape: batch_size, dim * 4, context_length
    X = self.cq_att(Ce, Qe, cmask, qmask)
    # M1 shape: batch_size, dim, context_length
    M1 = self.cq_resizer(X)
    for enc in self.model_enc_blks: M1 = enc(M1, cmask)
    M2 = M1
    for enc in self.model_enc_blks: M2 = enc(M2, cmask)
    M3 = M2
    for enc in self.model_enc_blks: M3 = enc(M3, cmask)
    p1, p2 = self.out(M1, M2, M3, cmask)
    return p1, p2
