import os
import pandas as pd

data_columns = ['text']


def combine_data(in_path, out_filepath):
    if not os.path.exists(in_path): raise Exception('Invalid data location: ' + in_path)
    data = []
    print('Processing data')
    for filename in os.listdir(in_path):
        #if not filename.endswith(".txt"): continue
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

