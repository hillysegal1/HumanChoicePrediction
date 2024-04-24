import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.usersvectors import UsersVectors

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_outputs).squeeze(2), dim=1)
        # attn_weights shape: (batch_size, seq_len)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        # context_vector shape: (batch_size, hidden_dim)
        return context_vector, attn_weights


class SpecialLSTM(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dropout, logsoftmax=True, input_twice=False):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.input_twice = input_twice
        self.attention = Attention(hidden_dim)  # Instantiate the attention module

        self.input_fc = nn.Sequential(nn.Linear(input_dim, input_dim * 2),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(input_dim * 2, self.hidden_dim),
                                      nn.Dropout(dropout),
                                      nn.ReLU())

        self.main_task = nn.LSTM(input_size=self.hidden_dim,
                                 hidden_size=self.hidden_dim,
                                 batch_first=True,
                                 num_layers=self.n_layers,
                                 dropout=dropout)

        seq = [nn.Linear(self.hidden_dim, self.hidden_dim // 2),
               nn.ReLU(),
               nn.Linear(self.hidden_dim // 2, self.output_dim)]
        if logsoftmax:
            seq += [nn.LogSoftmax(dim=-1)]

        self.output_fc = nn.Sequential(*seq)

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)

    def forward(self, input_vec, game_vector, user_vector):
        lstm_input = self.input_fc(input_vec)
        lstm_output, (game_hidden, user_hidden) = self.main_task(lstm_input.contiguous(),
                                                                 (game_vector.contiguous(),
                                                                  user_vector.contiguous()))
        # Apply attention
        context_vector, attn_weights = self.attention(lstm_output)

        if self.input_twice:
            context_vector = torch.cat([context_vector, input_vec], dim=-1)

        output = self.output_fc(context_vector)
        return {"output": output, "game_vector": game_hidden, "user_vector": user_hidden, "attn_weights": attn_weights}




    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)
