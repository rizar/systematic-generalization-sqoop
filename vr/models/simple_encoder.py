#!/usr/bin/env python3

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vr.embedding import expand_embedding_vocab
from vr.models.layers import init_modules
from torch.nn.init import uniform, xavier_uniform, constant

class SimpleEncoderBinary(nn.Module):
  '''
  SimpleEncoder takes a question extracts the important info (for binary relation VQA: s1, r, s2)
  embeds them into a learnt vector space and then concatenates them
  '''

  def __init__(self, vocab, embedding_dim_in, hidden_dim, embedding_dim_out):
    super(SimpleEncoder, self).__init__() 
    self.encoder = nn.Embedding(vocab, embedding_dim_in)
 
    self.feed_forward_1 = nn.Linear(3*embedding_dim_in, hidden_dim)
    self.feed_forward_2 = nn.Linear(hidden_dim, embedding_dim_out)

    self.tanh = nn.Tanh()

  def forward(self, x): 
    s1_embed = self.encoder(x[:, 4]) # B x D
    s2_embed = self.encoder(x[:, 6]) # B x D 
    relation_embed = self.encoder(x[:, 5]) # B x D

    return self.feed_forward_2(self.tanh(self.feed_forward_1(torch.cat([s1_embed, s2_embed, relation_embed], dim = -1))))


