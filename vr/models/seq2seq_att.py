#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import (pack_padded_sequence,
                                pad_packed_sequence)

from vr.embedding import expand_embedding_vocab



class Attn(nn.Module):
  def __init__(self, method, wordvec_size, hidden_size):
    super(Attn, self).__init__()

    self.method = method
    self.hidden_size = hidden_size
    self.wordvec_size = wordvec_size

    if self.method == 'general':
      self.attn = nn.Linear(self.hidden_size, hidden_size)

    elif self.method == 'concat':
      self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
      self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

  def forward(self, hidden, encoder_outputs):
    max_len = encoder_outputs.size(0)
    this_batch_size = encoder_outputs.size(1)

    # Create variable to store attention energies
    attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S
    attn_energies = attn_energies.cuda()

    # For each batch of encoder outputs
    for b in range(this_batch_size):
      # Calculate energy for each encoder output
      for i in range(max_len):
        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

    # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
    return F.softmax(attn_energies).unsqueeze(1)

  def score(self, hidden, encoder_output):
    if self.method == 'dot':
      energy = hidden.dot(encoder_output)
      return energy

    elif self.method == 'general':
      energy = self.attn(encoder_output)
      energy = hidden.dot(energy)
      return energy

    elif self.method == 'concat':
      energy = self.attn(torch.cat((hidden, encoder_output), 1))
      energy = self.v.dot(energy)
      return energy


class Seq2SeqAtt(nn.Module):
  def __init__(self,
    null_token=0,
    start_token=1,
    end_token=2,
    encoder_vocab_size=100,
    decoder_vocab_size=100,
    wordvec_dim=300,
    hidden_dim=256,
    rnn_num_layers=2,
    rnn_dropout=0,
  ):
    super().__init__()
    self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
    self.encoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                               dropout=rnn_dropout, batch_first=True)
    self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
    self.decoder_rnn = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
                               dropout=rnn_dropout, batch_first=True)
    self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
    self.decoder_attn = Attn('general', wordvec_dim, hidden_dim)
    self.rnn_num_layers = rnn_num_layers
    self.NULL = null_token
    self.START = start_token
    self.END = end_token
    self.multinomial_outputs = None

  def expand_encoder_vocab(self, token_to_idx, word2vec=None, std=0.01):
    expand_embedding_vocab(self.encoder_embed, token_to_idx,
                           word2vec=word2vec, std=std)

  def get_dims(self, x=None, y=None):
    V_in = self.encoder_embed.num_embeddings
    V_out = self.decoder_embed.num_embeddings
    D = self.encoder_embed.embedding_dim
    H = self.encoder_rnn.hidden_size
    L = self.encoder_rnn.num_layers

    N = x.size(0) if x is not None else None
    N = y.size(0) if N is None and y is not None else N
    T_in = x.size(1) if x is not None else None
    T_out = y.size(1) if y is not None else None
    return V_in, V_out, D, H, L, N, T_in, T_out

  def before_rnn(self, x, replace=0):
    # TODO: Use PackedSequence instead of manually plucking out the last
    # non-NULL entry of each sequence; it is cleaner and more efficient.
    N, T = x.size()
    idx = torch.LongTensor(N).fill_(T - 1)

    # Find the last non-null element in each sequence. Is there a clean
    # way to do this?
    x_cpu = x.cpu()
    for i in range(N):
      for t in range(T - 1):
        if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t + 1] == self.NULL:
          idx[i] = t
          break
    idx = idx.type_as(x.data)
    # x[x.data == self.NULL] = replace
    return x, Variable(idx)

  def encoder(self, x, x_lengths):
    V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)

    embed = self.encoder_embed(x)
    packed = pack_padded_sequence(embed, x_lengths, batch_first=True)
    out_packed, (hn, cn) = self.encoder_rnn(packed)
    out, _ = pad_packed_sequence(out_packed, batch_first=True)
    hn = hn.transpose(1,0)
    cn = cn.transpose(1,0)

    # TODO(mnoukhov) there is a better way to do this
    # Pull out the hidden state for the last non-null value in each input
    last_idx = Variable(torch.cuda.LongTensor(x_lengths) - 1)
    last_idx = last_idx.view(N, 1, 1).expand(N, 1, H)
    out = out.gather(1, last_idx).view(N, H)

    return out, (hn, cn)

  def decoder(self, encoded, y, h0=None, c0=None):
    V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
    if h0 is None:
      h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
    if c0 is None:
      c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))

    # embed current input
    word_embedded = self.decoder_embed(y)

    # calculate attention weights, apply to encoder
    attn_weights = self.decoder_attn(h0, c0, encoded)

    # create input of concat(previous true state, encoder_out)
    encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
    rnn_input = torch.cat([encoded_repeat, y_embed], 2)
    rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

    rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
    output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)

    return output_logprobs, ht, ct

  def decoder_with_hidden(self, encoder_outputs, word_inputs, prev_hidden):
    word_embedded = self.decoder_embed(word_inputs)

    attn_weights = self.attn(prev_hidden[-1], encoder_outputs)
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
    context = context.transpose(0, 1) # 1 x B x N

    # Combine embedded input word and attended context, run through RNN
    rnn_input = torch.cat((word_embedded, context), 2)
    output, hidden = self.decoder_rnn(rnn_input, prev_hidden)

    # Final output layer
    output = output.squeeze(0) # B x N
    output = F.log_softmax(self.decoder_linear(torch.cat((output, context), 1)))

    return output, hidden, attn_weights

  def compute_loss(self, output_logprobs, y):
    """
    Compute loss. We assume that the first element of the output sequence y is
    a start token, and that each element of y is left-aligned and right-padded
    with self.NULL out to T_out. We want the output_logprobs to predict the
    sequence y, shifted by one timestep so that y[0] is fed to the network and
    then y[1] is predicted. We also don't want to compute loss for padded
    timesteps.

    Inputs:
    - output_logprobs: Variable of shape (N, T_out, V_out)
    - y: LongTensor Variable of shape (N, T_out)
    """
    self.multinomial_outputs = None
    V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
    mask = y.data != self.NULL
    y_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
    y_mask[:, 1:] = mask[:, 1:]
    y_masked = y[y_mask]
    out_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
    out_mask[:, :-1] = mask[:, 1:]
    out_mask = out_mask.view(N, T_out, 1).expand(N, T_out, V_out)
    out_masked = output_logprobs[out_mask].view(-1, V_out)
    loss = F.cross_entropy(out_masked, y_masked)
    return loss

  def forward(self, x, x_lengths, y):
    V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)

    encoder_out, (encoder_hn, encoder_cn) = self.encoder(x, x_lengths)
    output_logprobs, _, _ = self.decoder(encoder_out, y)
    loss = self.compute_loss(output_logprobs, y)
    return loss

  def sample(self, x, x_lengths, max_length=50):
    # TODO: Handle sampling for minibatch inputs
    # TODO: Beam search?
    self.multinomial_outputs = None
    assert x.size(0) == 1, "Sampling minibatches not implemented"
    encoded = self.encoder(x, x_lengths)
    y = [self.START]
    h0, c0 = None, None
    while True:
      cur_y = Variable(torch.LongTensor([y[-1]]).type_as(x.data).view(1, 1))
      logprobs, h0, c0 = self.decoder(encoded, cur_y, h0=h0, c0=c0)
      _, next_y = logprobs.data.max(2, keepdim=True)
      y.append(next_y[0, 0, 0])
      if len(y) >= max_length or y[-1] == self.END:
        break
    return y

  def reinforce_sample(self, x, x_lengths, max_length=30, temperature=1.0, argmax=False):
    N, T = x.size(0), max_length
    encoded = self.encoder(x, x_lengths)
    y = torch.LongTensor(N, T).fill_(self.NULL)
    done = torch.ByteTensor(N).fill_(0)
    cur_input = Variable(x.data.new(N, 1).fill_(self.START))
    h, c = None, None
    self.multinomial_outputs = []
    self.multinomial_probs = []
    for t in range(T):
      # logprobs is N x 1 x V
      logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
      logprobs = logprobs / temperature
      probs = F.softmax(logprobs.view(N, -1), dim=1) # Now N x V
      if argmax:
        _, cur_output = probs.max(1, keepdim=True)
      else:
        cur_output = probs.multinomial() # Now N x 1
      self.multinomial_outputs.append(cur_output)
      self.multinomial_probs.append(probs)
      cur_output_data = cur_output.data.cpu()
      not_done = logical_not(done)
      y[:, t][not_done] = cur_output_data[not_done]
      done = logical_or(done, cur_output_data.cpu() == self.END)
      cur_input = cur_output
      if done.sum() == N:
        break
    return Variable(y.type_as(x.data))

  def reinforce_backward(self, reward, output_mask=None):
    """
    If output_mask is not None, then it should be a FloatTensor of shape (N, T)
    giving a multiplier to the output.
    """
    assert self.multinomial_outputs is not None, 'Must call reinforce_sample first'
    grad_output = []

    def gen_hook(mask):
      def hook(grad):
        return grad * mask.contiguous().view(-1, 1).expand_as(grad)
      return hook

    if output_mask is not None:
      for t, probs in enumerate(self.multinomial_probs):
        mask = Variable(output_mask[:, t])
        probs.register_hook(gen_hook(mask))

    for sampled_output in self.multinomial_outputs:
      sampled_output.reinforce(reward)
      grad_output.append(None)
    torch.autograd.backward(self.multinomial_outputs, grad_output, retain_variables=True)


def logical_or(x, y):
  return (x + y).clamp_(0, 1)

def logical_not(x):
  return x == 0
