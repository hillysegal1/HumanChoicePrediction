import torch
import torch.nn as nn
from utils.usersvectors import UsersVectors
from environments import environment
from SpecialLSTM import SpecialLSTM

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        attention_logits = self.attention_weights(lstm_outputs)
        attention_weights = torch.softmax(attention_logits, dim=1)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)
        return context_vector

class SpecialLSTMWithAttention(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dropout):
        super(SpecialLSTMWithAttention, self).__init__()
        self.special_lstm = SpecialLSTM(n_layers, input_dim, hidden_dim, output_dim, dropout)
        self.attention = Attention(hidden_dim)

    def forward(self, input_vec, game_vector, user_vector):
        lstm_output_dict = self.special_lstm.forward(input_vec, game_vector, user_vector)
        lstm_outputs = lstm_output_dict["output"]
        context_vector = self.attention.forward(lstm_outputs)
        return context_vector

class LSTM_env_ARC(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.special_lstm_with_attention = SpecialLSTMWithAttention(n_layers, input_dim, hidden_dim, output_dim,
                                                                    dropout)

    def forward(self, vectors, **kwargs):
        data = self.special_lstm_with_attention.forward(vectors["x"], vectors["game_vector"], vectors["user_vector"])
        return data

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        assert not update_vectors
        if vectors_in_input:
            output = self.model(data)  # Call the forward method of the model attribute
        else:
            output = self.model({**data, "user_vector": self.currentDM,
                                 "game_vector": self.currentGame})  # Call the forward method of the model attribute
        output["proba"] = torch.exp(output["output"].flatten())
        return output

class LSTM_env(environment.Environment):
    def init_model_arc(self, config):
        self.model = LSTM_env_ARC(n_layers=config['n_layers'], input_dim=config['input_dim'],
                                  hidden_dim=config['hidden_dim'],
                                  output_dim=config['output_dim'], dropout=config['dropout']).double()

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            output = self.model({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        if update_vectors:
            self.currentDM = output["user_vector"]
            self.currentGame = output["game_vector"]
        return output

    def init_user_vector(self):
        self.currentDM = self.model.special_lstm_with_attention.special_lstm.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.special_lstm_with_attention.special_lstm.init_game()

    def get_curr_vectors(self):
        return {"user_vector": 888}
