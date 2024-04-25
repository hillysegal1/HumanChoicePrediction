import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.usersvectors import UsersVectors

class Attention(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
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
        self.attention = Attention(hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        # Instantiate the attention module

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

        lstm_shape = lstm_input.shape
        shape = user_vector.shape
        assert game_vector.shape == shape
        if len(lstm_shape) != len(shape):
            lstm_input = lstm_input.reshape((1,) * (len(shape) - 1) + lstm_input.shape)
        user_vector = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))

        # Pass the inputs and hidden states to the LSTM
        lstm_output, (game_vector, user_vector) = self.main_task(lstm_input.contiguous(),
                                                             (game_vector.contiguous(),
                                                              user_vector.contiguous()))
        user_vector = user_vector.reshape(shape)
        game_vector = game_vector.reshape(shape)

        # Apply attention
        context_vector, attn_weights = self.attention.forward(lstm_output)
        batch_size, seq_len, hidden_dim = lstm_output.size()
        context_vector = context_vector.unsqueeze(1).expand(batch_size, seq_len, hidden_dim)

        if hasattr(self, "input_twice") and self.input_twice:
            context_vector = torch.cat([context_vector, input_vec], dim=-1)

        output = self.output_fc(context_vector)
        #output = output.view(lstm_shape[0], lstm_shape[1], -1)  # Reshape to [4, 10, 2]

        #output = output.view(-1, self.output_dim)  # Reshape to [batch_size * DATA_ROUNDS_PER_GAME, -1]

        if len(output.shape) != len(lstm_shape):
            output.reshape(-1, output.shape[-1])

        if self.training:
            return {"output": output, "game_vector": game_vector, "user_vector": user_vector, "attn_weights": attn_weights}
        else:
            return {"output": output, "game_vector": game_vector.detach(), "user_vector": user_vector.detach(),"attn_weights": attn_weights}



    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)
