import torch
import torch.nn as nn
from environments.transformer_env import transformer_env_ARC
from utils.usersvectors import UsersVectors

class SpecialLSTM(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dropout, logsoftmax=True, input_twice=False):
        super().__init__()
        # Your existing initialization code
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.input_twice = input_twice

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

        seq = [nn.Linear(self.hidden_dim + (input_dim if self.input_twice else 0), self.hidden_dim // 2),
               nn.ReLU(),
               nn.Linear(self.hidden_dim // 2, self.output_dim)]
        if logsoftmax:
            seq += [nn.LogSoftmax(dim=-1)]

        self.output_fc = nn.Sequential(*seq)

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)

        # Initialize the transformer model
        self.transformer_model = transformer_env_ARC(config={"input_dim": input_dim, "dropout": dropout, "hidden_dim": hidden_dim, "transformer_nheads": 2, "layers": 2})

    def forward(self, input_vec, game_vector, user_vector):
        # LSTM forward pass
        lstm_output = self.input_fc(input_vec)
        lstm_shape = lstm_output.shape
        shape = user_vector.shape
        assert game_vector.shape == shape
        if len(lstm_shape) != len(shape):
            lstm_output = lstm_output.reshape((1,) * (len(shape) - 1) + lstm_output.shape)
        user_vector = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        lstm_output, (game_vector, user_vector) = self.main_task(lstm_output.contiguous(),
                                                                 (game_vector.contiguous(),
                                                                  user_vector.contiguous()))
        user_vector = user_vector.reshape(shape)
        game_vector = game_vector.reshape(shape)

        # Transformer forward pass
        transformer_output = self.transformer_model({"x": input_vec})

        # Concatenate LSTM and transformer outputs
        combined_output = torch.cat([lstm_output, transformer_output["output"]], dim=-1)

        # Apply fusion layer
        fused_output = self.output_fc(combined_output)

        return fused_output

class HybridModel(nn.Module):
    def __init__(self, config, n_layers, input_dim, hidden_dim, output_dim, dropout, logsoftmax=True, input_twice=False):
        super(HybridModel, self).__init__()
        self.special_lstm_model = SpecialLSTM(n_layers, input_dim, hidden_dim, output_dim, dropout, logsoftmax, input_twice)
        self.transformer_model = transformer_env_ARC(config={"input_dim": input_dim, "dropout": dropout, "hidden_dim": hidden_dim, "transformer_nheads": 2, "layers": 2})
        self.fusion_layer = nn.Linear(output_dim * 2, output_dim)  # Update output_dim based on your requirement

    def forward(self, input_vec, game_vector, user_vector):
        lstm_output = self.special_lstm_model(input_vec, game_vector, user_vector)
        transformer_output = self.transformer_model({"x": input_vec})
        combined_output = torch.cat([lstm_output, transformer_output["output"]], dim=-1)
        fused_output = self.fusion_layer(combined_output)
        return fused_output
