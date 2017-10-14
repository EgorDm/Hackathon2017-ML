import pandas as pd
import tools.text_processing as text_proc

data = '../data/positives/dataset.csv'

df = pd.read_csv(data)

samples = []
for text in df['text']:
    samples.append(text_proc.clean_text(text))

dfs = pd.Series(samples)
print(dfs.describe())
print(dfs.head())