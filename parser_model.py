#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParserModel(nn.Module):
    def __init__(self, embeddings, n_features=36, hidden_size=200, n_classes=3, dropout_prob=0.5):
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))

        self.embed_to_hidden = nn.Linear(self.n_features * self.embed_size, self.hidden_size, bias=True)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes, bias=True)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        self.dropout = nn.Dropout(p=dropout_prob)

    def embedding_lookup(self, t):
        x = self.pretrained_embeddings(t).view(t.shape[0], -1)
        return x

    def forward(self, t):
        embeddings = self.embedding_lookup(t)
        h = F.relu(self.embed_to_hidden(embeddings))
        logits = self.hidden_to_logits(self.dropout(h))
        return logits
