from tools.dataset import combine_data, combine_datasets
import pandas as pd

positives_dataset = 'data/positives/dataset.csv'
negatives_dataset = 'data/negatives/dataset.csv'
main_dataset = 'data/dataset.csv'

print('Combining data')
combine_data('data/positives/raw',positives_dataset)
combine_data('data/negatives/raw',negatives_dataset)

print('Combining datasets')
combine_datasets(positives_dataset, negatives_dataset, main_dataset)
print('All set')

df = pd.read_csv(main_dataset)
print(df.head())
print(df.describe())
df = df.groupby('label')
print(df.count())