import os

import pandas as pd
from sklearn.model_selection import train_test_split

import tools.text_processing as tp


def combine_data(in_path, out_filepath, min_len=20):
    if not os.path.exists(in_path): raise Exception('Invalid data location: ' + in_path)
    data = []
    print('Processing data')
    for filename in os.listdir(in_path):
        # if not filename.endswith(".txt"): continue
        try:
            with open(os.path.join(in_path, filename), 'r') as file:
                contents = file.readlines()
            contents = contents[1:]
            data.append(''.join(contents))
        except:
            print('Could not read file: {}'.format(os.path.join(in_path, filename)))

    df = pd.DataFrame(data=data, columns=['text'])
    df.drop_duplicates(subset=["text"], inplace=True)
    df.to_csv(out_filepath)
    print('Saved dataset')


def combine_datasets(positive_path, negative_path, out_filepath):
    print('Processing data')
    dp, lp = label_data(positive_path, 1)
    dn, ln = label_data(negative_path, 0)
    pl, nl = balance_dataset_len(len(dp), len(dn))
    col_data = {
        'label': lp[:pl] + ln[:nl],
        'data': dp[:pl] + dn[:nl]
    }

    df = pd.DataFrame(col_data)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(out_filepath)
    print('Saved dataset')


def label_data(path, label, min_len=50):
    if not os.path.exists(path): raise Exception('Invalid data location: ' + path)
    df = pd.read_csv(path)
    data = []
    labels = []
    for d in df['text']:
        try:
            text = tp.clean_text(d)
            if len(text) < min_len: continue
            data.append(text)
            labels.append(label)
        except:
            pass
    return data, labels


def balance_dataset_len(ds1_len, ds2_len, max_diff=0.1):
    lg = max(ds1_len, ds2_len)
    sm = min(ds1_len, ds2_len)
    if lg - sm <= lg * max_diff: return int(ds1_len), int(ds2_len)

    lg = sm + lg * max_diff
    sm, lg = int(sm), int(lg)
    return (lg, sm) if ds1_len > ds2_len else (sm, lg)


def load_dataset(path, valid_size=0.05):
    data = pd.read_csv(path)
    return train_test_split(data['data'].values, data['label'].values, test_size=valid_size,
                                                        random_state=42)
