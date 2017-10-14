import argparse
import os

from models.BaseModel import BaseModel
from tools.ml import pick_model
from tools.text_processing import clean_text


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, default='data/validation_manual/text_0.txt', help='Data to classify')
    parser.add_argument('--name', type=str, default='my_model',
                        help='Name of the training session. Used to save the model.')
    run(parser.parse_args())


labels = {
    0: 'Is NOT about Corporate Social Responsibility',
    1: 'YES, it is about Corporate Social Responsibility',
}


def run(settings):
    save_file = 'saves/{}/save.model'.format(settings.name)
    model = BaseModel(save_file)

    if not os.path.exists(settings.input): raise Exception('Input file doesnt exist: {}'.format(settings.input))
    with open(settings.input, 'r') as f:
        text_input = ''.join(f.readlines())
    text_input = clean_text(text_input)

    result = model.predict([text_input])
    print('Input text is: {}'.format(labels[result[0]]))


if __name__ == '__main__':
    main()
