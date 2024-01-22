import joblib
import torch

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection

import fasttext as ft

import numpy as np

MAX_LEN = 128  
TRAIN_BATCH_SIZE = 128 
VALID_BATCH_SIZE = 128 
EPOCHS = 1  
EMBED_DIM = 300 

MODEL_PATH = "./data/entities.pt"  # модель
TRAINING_FILE = "./data/ner_dataset.csv"    # предложения

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

fasttext_model = ft.load_model("./data/cc.ru.300.bin")

class EntityDataset:
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tags = np.array(self.tags[item])
        for w in text:
            if w == '' or w == np.nan:
                w = ' '
        ids = np.array([fasttext_model.get_word_vector(w) for w in text])

        ids_pad = np.zeros((MAX_LEN, EMBED_DIM))
        tags_pad = [] 
        
        for i in range(len(text)):
            padding_len = EMBED_DIM - ids[i].shape[0]
            ids_nppad = np.pad(ids[i], (0, padding_len), 'constant', constant_values=(0))
            ids_pad[i] = ids_nppad
            tags_pad.append(tags[i])

        padding_len = MAX_LEN - ids.shape[0]
        mask = np.ones(len(ids))
        mask = np.pad(mask, (0, padding_len), 'constant', constant_values=(0))
        tags_pad = np.pad(tags_pad, (0, padding_len), 'constant', constant_values=(0))

        # возвращаем матрицу с нулями, список тегов с нулями, и маску -- разметку, какие элементы 
        # являются словами, а какие -- пустые поля (нули). Маска нужна, чтобы правильно считать лосс
        # то есть, не учитывать в нем "пустые" части с нулями
        return (torch.tensor(ids_pad, dtype=torch.float32),
                torch.tensor(tags_pad, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long))

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

def acc_stat(pred, target, mask):
    mask = mask.bool()
    pred = torch.masked_select(pred, mask)
    target = torch.masked_select(target, mask)
    
    # угадано корректно
    correct = torch.sum((pred == target))
    # было всего, не считая "пустых" с нулями
    total = torch.tensor(len(pred))
    return correct, total

import torch.nn as nn

class EntityModel(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, bidirectional=False):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
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
        
    def init_hidden(self, batch_size):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_zeros = torch.zeros(self.n_layers * num_directions,
                              batch_size, self.hidden_dim,
                              dtype=torch.float32, device=device)
        c_zeros = torch.zeros(self.n_layers * num_directions,
                              batch_size, self.hidden_dim,
                              dtype=torch.float32, device=device)

        return (h_zeros, c_zeros)

def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].ffill()

    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, tag, enc_tag

sentences, tag, enc_tag = process_data(TRAINING_FILE)

meta_data = {
    "enc_tag": enc_tag
}

joblib.dump(meta_data, "./data/meta.bin")

num_tag = len(list(enc_tag.classes_))

# data split 
(
    train_sentences,
    test_sentences,
    train_tag,
    test_tag
) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.15)

train_dataset = EntityDataset(
    texts=train_sentences, tags=train_tag
)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=1,
    shuffle=True, drop_last=True
)

valid_dataset = EntityDataset(
    texts=test_sentences, tags=test_tag
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=1,
    shuffle=False, drop_last=True
)

def eval_model(model, valid_data_loader):
    h = model.init_hidden(VALID_BATCH_SIZE)
    losses = []
    
    correct_sum, total_sum = 0, 0
    
    for inputs, labels, mask in valid_data_loader:
        h = tuple([each.data for each in h])
        inputs = inputs.to(device)
        labels = labels.to(device)
        mask = mask.to(device)# отправим inputs, labels и mask на GPU
        model.zero_grad()
        output, h = model(inputs, h)
        loss = loss_fn(output, labels.flatten(), mask, num_tag)
        losses.append(loss.item())
        
        correct, total = acc_stat(torch.argmax(output, dim=-1).flatten(), labels.flatten(), mask.flatten())
        correct_sum += correct
        total_sum += total
    return losses, correct_sum / total_sum

hidden_dim = 512
n_layers = 2

model = EntityModel(num_tag, EMBED_DIM, hidden_dim, n_layers, drop_prob=0.5, bidirectional=False)
model.to(device)

lr=0.005
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


from torch.utils.tensorboard import SummaryWriter

counter = 0
print_every = 10
clip = 5
valid_loss_min = np.Inf
writer = SummaryWriter('logs')

model.train()
for i in range(EPOCHS):
    h = model.init_hidden(TRAIN_BATCH_SIZE)
    
    correct_sum, total_sum = 0, 0
    
    for inputs, labels, mask in train_data_loader:
        counter += 1
        h = tuple([e.data for e in h])

        inputs = inputs.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = loss_fn(output, labels.flatten(), mask, num_tag) # вызываем функцию для подсчета лосса
        loss.backward() # и делаем обратное распространение ошибки
        correct, total = acc_stat(torch.argmax(output, dim=-1).flatten(), labels.flatten(), mask.flatten()) # вызываем функцию acc_stat
        correct_sum += correct
        total_sum += total

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # градиентный спуск
        
        if counter % print_every == 0:
            model.eval()
            val_losses, val_acc = eval_model(model, valid_data_loader)
            model.train()
            
            val_loss = np.mean(val_losses)
            writer.add_scalar('train/loss', loss.item(), counter)
            writer.add_scalar('val/loss', val_loss, counter)
            writer.add_scalar('train/acc', correct_sum / total_sum, counter)
            writer.add_scalar('val/acc', val_acc, counter)

            print("Epoch: {}/{}...".format(i+1, EPOCHS),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(val_loss),  # loss - функция потерь
                  "Train Acc: {:.6f}".format(correct_sum / total_sum),
                  "Val Acc: {:.6f}".format(val_acc))
                
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), MODEL_PATH)
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)


