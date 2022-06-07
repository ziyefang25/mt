import torch
from torch import nn
from basicModel import BernoulliGate, RCNNCell, WordLSTMEncoder
import numpy as np


class IndependentGenerator(nn.Module):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(self,
                 vocab_size,
                 emb_size: int = 200,
                 hidden_size: int = 200,
                 dropout: float = 0.1,
                 layer: str = "rcnn"
                 ):

        super(IndependentGenerator, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.embed_layer = nn.Sequential(
            self.embeddings,
            nn.Dropout(p=dropout)
        )
        enc_size = hidden_size * 2

        self.enc_layer = WordLSTMEncoder(emb_size, hidden_size)

        self.z_layer = BernoulliGate(enc_size)

        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)

    def forward(self, docs, doc_lengths, sent_lengths, num_samples=1):

        # encode sentence
        docs = self.embed_layer(docs)  # [B, T, E]
        h, _ = self.enc_layer(docs, doc_lengths, sent_lengths)

        # compute parameters for Bernoulli p(z|x)
        z_dist = self.z_layer(h)

        if self.training:  # sample
            z = z_dist.sample()  # [B, T, 1]
        else:  # deterministic
            z = (z_dist.probs >= 0.5).float()  # [B, T, 1]

        z = z.squeeze(-1)  # [B, T, 1]  -> [B, T]

        self.z = z
        self.z_dists = [z_dist]

        return z


class DependentGenerator(nn.Module):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(self,
                 vocab_size,
                 emb_size: int = 200,
                 hidden_size: int = 200,
                 dropout: float = 0.1,
                 layer: str = "rcnn",
                 z_rnn_size: int = 30,
                 ):

        super(DependentGenerator, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.embed_layer = nn.Sequential(
            self.embeddings,
            nn.Dropout(p=dropout)
        )
        enc_size = hidden_size * 2

        self.enc_layer = WordLSTMEncoder(emb_size, hidden_size)

        self.z_cell = RCNNCell(enc_size + 1, z_rnn_size)
        self.z_layer = BernoulliGate(enc_size + z_rnn_size)

        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)
        self.logits_all = []

    def forward(self, docs, doc_lengths, sent_lengths, num_samples=1):
        docs = self.embed_layer(docs)
        # encode sentence
        batch_size, sentence, time, emb = docs.size()
        h, _ = self.enc_layer(docs, doc_lengths, sent_lengths)
        # predict z for each time step conditioning on previous z
        h = h.transpose(0, 2)  # [T, S, B, E)
        h = h.transpose(1, 2)  # [T, B, S, E)
        z = []
        z_dists = []
        logits_all = []

        # initial states  [1, B, z_rnn_dim]
        state = torch.zeros([3 * batch_size, sentence, self.z_cell.hidden_size],
                            device=docs.device).chunk(3)

        for h_t, t in zip(h, range(time)):

            # compute Binomial z distribution for this time step
            logits = torch.cat([h_t, state[0]], dim=-1)
            z_t_dist = self.z_layer(logits)  # [B, S, 1]
            z_dists.append(z_t_dist)
            logits_all.append(logits)

            if self.training:
                # sample (once since we already repeated the state)
                z_t = z_t_dist.sample().detach()  # [B, 1]
            else:
                z_t = (z_t_dist.probs >= 0.5).float().detach()
            assert (z_t < 0.).sum().item() == 0, "cannot be smaller than 0."
            z.append(z_t)  # [B, S, 1]

            # update cell state (to make dependent decisions)
            rnn_input = torch.cat([h_t, z_t], dim=-1)  # [B, S, 2D+1]
            state = self.z_cell(rnn_input, state)

        z = torch.stack(z, dim=2).squeeze(-1)  # [B, S, T]

        self.z = z
        self.z_dists = z_dists
        self.logits_all = logits_all
        return z
