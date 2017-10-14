import tools.dataset as ds
import pandas as pd

positives = '../data/positives/dataset.csv'
negatives = '../data/positives/dataset.csv'
out = '../.tmp/test_ds_compiling.csv'


ds.combine_datasets(positives, negatives, out)

df = pd.read_csv(out)
print(df.head())
print(df.describe())
print(df.describe())
df = df.groupby('label')
print(df.count())