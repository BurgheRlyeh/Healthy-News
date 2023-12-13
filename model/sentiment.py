import spacy
from tqdm.auto import tqdm
from spacy.tokens import DocBin
import ru_core_news_md
import re
import numpy as np
import pandas as pd
import random
nlp = spacy.load('ru_core_news_md')

df = pd.read_csv('dataset.csv', delimiter = '\t')
df = df[df['sentiment']!='neautral']
df['review'] = df['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['review'] = df['review'].str.lower()

def make_docs(data):
    docs = []
    a = tqdm(nlp.pipe(data, as_tuples=True), total = len(data))
    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total = len(data)):
        if label == 'negative':
            doc.cats["positive"] = 0
            doc.cats["negative"] = 1
        else:
            doc.cats["positive"] = 1
            doc.cats["negative"] = 0
        docs.append(doc)
    return docs

# convert df into a list of tuples(review, label)
data = [tuple(df.iloc[i, :]) for i in range(df.shape[0])]
random.shuffle(data)

print(len(data))
print("Example: ", data[0])
train = data[:50000]
valid = data[50000:]
print(f"Train size: {len(train)} Valid size: {len(valid)}")

# transform all the training data in spacy documents
train_docs = make_docs(train)
valid_docs = make_docs(valid)

doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk("train.spacy") # save it in a binary file to disc

doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk("valid.spacy")

