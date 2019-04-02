#!/usr/bin/env python3

# Copyright 2019-present, Mila
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import (pack_padded_sequence,
                                pad_packed_sequence)

from vr.embedding import expand_embedding_vocab


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, encoder_outputs, hidden):
        '''
        :param hidden:
          previous hidden state of the decoder, in shape (layers*directions, H)
        :param encoder_outputs:
          encoder outputs from Encoder, in shape (B,T,H)
        :return
          attention energies in shape (B,T)
        '''
        seq_len = encoder_outputs.size(1)
        H = hidden.repeat(seq_len, 1, 1).transpose(0,1)
        attn_energies = self.score(H, encoder_outputs) # B*1*T
        return F.softmax(attn_energies, dim=2)

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v, energy) # [B*1*T]
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
        self.decoder_attn = Attn(hidden_dim)
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

    def encoder(self, x):
        x, x_lengths, inverse_index = sort_for_rnn(x, null=self.NULL)
        embed = self.encoder_embed(x)
        packed = pack_padded_sequence(embed, x_lengths, batch_first=True)
        out_packed, hidden = self.encoder_rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        out = out[inverse_index]
        hidden = [h[:,inverse_index] for h in hidden]

        return out, hidden

    def decoder(self, word_inputs, encoder_outputs, prev_hidden):
        hn, cn = prev_hidden
        word_embedded = self.decoder_embed(word_inputs).unsqueeze(1) # batch x 1 x embed

        attn_weights = self.decoder_attn(encoder_outputs, hn[-1])
        context = attn_weights.bmm(encoder_outputs) # batch x 1 x hidden

        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.decoder_rnn(rnn_input, prev_hidden)

        output = output.squeeze(1) # batch x hidden
        output = self.decoder_linear(output)

        return output, hidden

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

    def forward(self, x, y):
        max_target_length = y.size(1)

        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_inputs = y
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for t in range(max_target_length):
            decoder_out, decoder_hidden = self.decoder(
                decoder_inputs[:,t], encoder_outputs, decoder_hidden)
            decoder_outputs.append(decoder_out)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        loss = self.compute_loss(decoder_outputs, y)
        return loss

    def sample(self, x, max_length=50):
        # TODO: Handle sampling for minibatch inputs
        # TODO: Beam search?
        self.multinomial_outputs = None
        assert x.size(0) == 1, "Sampling minibatches not implemented"

        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_hidden = encoder_hidden
        sampled_output = [self.START]
        for t in range(max_length):
            decoder_input = Variable(torch.cuda.LongTensor([sampled_output[-1]]))
            decoder_out, decoder_hidden = self.decoder(
                decoder_input, encoder_outputs, decoder_hidden)
            _, argmax = decoder_out.data.max(1)
            output = argmax[0]
            sampled_output.append(output)
            if output == self.END:
                break

        return sampled_output

    def reinforce_sample(self, x, max_length=30, temperature=1.0, argmax=False):
        N, T = x.size(0), max_length
        encoder_outputs, encoder_hidden = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        done = torch.ByteTensor(N).fill_(0)
        cur_input = Variable(x.data.new(N, 1).fill_(self.START))
        decoder_hidden = encoder_hidden
        self.multinomial_outputs = []
        self.multinomial_probs = []
        for t in range(T):
            # logprobs is N x 1 x V
            logprobs, decoder_hidden = self.decoder(cur_input, encoder_outputs, decoder_hidden)
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

def sort_for_rnn(x, null=0):
    lengths = torch.sum(x != null, dim=1).long()
    sorted_lengths, sorted_idx = torch.sort(lengths, dim=0, descending=True)
    sorted_lengths = sorted_lengths.data.tolist() # remove for pytorch 0.4+
    # ugly
    inverse_sorted_idx = torch.LongTensor(sorted_idx.shape).fill_(0).to(device)
    for i, v in enumerate(sorted_idx):
        inverse_sorted_idx[v.data] = i

    return x[sorted_idx], sorted_lengths, inverse_sorted_idx

