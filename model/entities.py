
import joblib
import torch
import torch.nn as nn
import transformers
#import nltk

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection

from gensim.models import FastText as ft
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

import numpy as np
#from nltk.tokenize import word_tokenize

from tqdm import tqdm

"""config"""

MAX_LEN = 128
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
EPOCHS = 5
EMBED_DIM = 300

MODEL_PATH = "./state_dict.pt"
TRAINING_FILE = "./data/ner_dataset.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

#import fasttext as ft
from gensim.models.fasttext import load_facebook_model
fasttext_model = load_facebook_model("cc.en.300.bin")
fasttext_model.wv["alabama"].size

class EntityDataset:
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tags = np.array(self.tags[item])
        ids = np.array([fasttext_model.wv[w] for w in text])

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
