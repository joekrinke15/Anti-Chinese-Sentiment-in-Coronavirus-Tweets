import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from utils import clean_text
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from tqdm import tqdm



TOK_PATH = 'model_weights/tokenizer.pickle'
MODEL_PATH = 'model_weights/model.h5'
MAXLEN = 200  # more than 99th percentile

LABELS = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat']


def preprocess(text_list, tokenizer_path=TOK_PATH, maxlen=MAXLEN):
    print('cleaning input text....')
    cleaned_text = [clean_text(text) for text in tqdm(text_list)]
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    print('processing input ....')
    processed_input = tokenizer.texts_to_sequences(cleaned_text)
    processed_input = pad_sequences(processed_input, maxlen)
    return processed_input


def get_predicted_labels(model_input, model_path, batch_size=256):
    print('loading model....')
    model = load_model(model_path)
    print('predicting labels....')
    y_pred = model.predict(model_input, batch_size=batch_size, verbose=1)
    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default=None,
                        help="specify the input filename")
    parser.add_argument('-o', '--output_path',
                        default='./predicted_labels.csv', help="specify the output filename")
    parser.add_argument('--id', default='id', help="name of the id columns")
    parser.add_argument('--text', default='text',
                        help="name of the text columns")
    parser.add_argument('-tok', '--tokenizer_path',
                        default=TOK_PATH, help="path to tokenizer pickle")
    parser.add_argument('-model', '--model_path',
                        default=MODEL_PATH, help="path to saved model")
    parser.add_argument('-bs', '--batch_size', default=64,
                        help="prediction batch size", type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, usecols=[args.id, args.text])
    model_input = preprocess(df[args.text], args.tokenizer_path)
    predictions = get_predicted_labels(model_input, args.model_path, args.batch_size)
    for i, label in enumerate(LABELS):
        df[label] = predictions[:,i]
        pass

    df.to_csv(args.output_path)
