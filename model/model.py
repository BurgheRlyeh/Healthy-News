import pandas as pd
import re

df = pd.read_csv('IMDB Dataset.csv', delimiter = '\t')
df = df[df['sentiment']!='neautral']
df['review'] = df['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['review'] = df['review'].str.lower()

print(df)