import joblib
import torch
import torch.nn as nn
import numpy as np
import fasttext as ft
import numpy as np

fasttext_model = None
enc_tag = None
device = None
model = None

class EntityModel(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim = 512, n_layers = 2, drop_prob=0.5, bidirectional=False):
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob,
                            batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, embeds, hidden):
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # пропустим через дропаут и линейный слой
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden
        
    def init_hidden(self, batch_size = 1):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_zeros = torch.zeros(self.n_layers * num_directions,
                              batch_size, self.hidden_dim,
                              dtype=torch.float32, device=device)
        c_zeros = torch.zeros(self.n_layers * num_directions,
                              batch_size, self.hidden_dim,
                              dtype=torch.float32, device=device)

        return (h_zeros, c_zeros)


def initialize():
    print('initializing entities model...')

    global fasttext_model
    fasttext_model = ft.load_model("./models/data/cc.en.300.bin") 
    meta_data = joblib.load("./models/data/meta.bin")

    global enc_tag
    enc_tag = meta_data["enc_tag"]

    global device
    device = torch.device("cuda")

    global model
    model = EntityModel(len(list(enc_tag.classes_)), 300)
    model.load_state_dict(torch.load("./models/data/entities.pt"))
    model.to(device)

    print('initialized')


def get_entities(words):
    inputs = torch.tensor(np.array([fasttext_model.get_word_vector(s) for s in words]), dtype=torch.float32)
    inputs = inputs.unsqueeze(0).to(device)
    h = model.init_hidden()
    tag, h = model(inputs, h)

    return enc_tag.inverse_transform(tag.argmax(-1).cpu().numpy().reshape(-1))
