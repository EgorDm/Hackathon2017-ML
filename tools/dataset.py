import os

import pandas as pd

import tools.text_processing as tp

data_columns = ['text']


def combine_data(in_path, out_filepath):
    if not os.path.exists(in_path): raise Exception('Invalid data location: ' + in_path)
    data = []
    print('Processing data')
    for filename in os.listdir(in_path):
        # if not filename.endswith(".txt"): continue
        try:
            with open(os.path.join(in_path, filename), 'r') as file:
                contents = file.readlines()
            contents = contents[1:]
            # todo clean
            data.append(''.join(contents))
        except:
            print('Could not read file: {}'.format(os.path.join(in_path, filename)))
    df = pd.DataFrame(data=data, columns=data_columns)
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


def label_data(path, label):
    if not os.path.exists(path): raise Exception('Invalid data location: ' + path)
    df = pd.read_csv(path)
    data = []
    labels = []
    for d in df['text']:
        data.append(tp.clean_text(d))
        labels.append(label)
    return data, labels


def balance_dataset_len(ds1_len, ds2_len, max_diff=0.1):
    lg = max(ds1_len, ds2_len)
    sm = min(ds1_len, ds2_len)
    if lg - sm <= lg * max_diff: return ds1_len, ds2_len

    lg = sm + lg * max_diff
    return (lg, sm) if ds1_len > ds2_len else (sm, lg)