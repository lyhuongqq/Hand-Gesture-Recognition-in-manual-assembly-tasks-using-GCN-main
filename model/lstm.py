import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import utils.adj_mat

from config import CFG


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, batch_size, num_layers ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = batch_size

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(int(hidden_dim/2), output_dim),
        )

    def forward(self, x):
        #h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        #c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #out = torch.flatten(out,start_dim= 1, end_dim= 2)
        #out = self.fc(out) 
        #return out
        x = x.reshape(self.batch_size, self.seq_len, CFG.num_feats*21)
        #x = x.view(x.size(0), self.seq_len, CFG.num_feats*21)
        x, (h, c) = self.lstm(x)
        x = x[:, -1, :]
        x = self.hidden2label(x)
        return x