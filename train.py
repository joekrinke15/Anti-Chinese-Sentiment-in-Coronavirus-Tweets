import numpy as np
np.random.seed(42)

import pandas as pd
import pickle
from gensim.models import fasttext
from utils import clean_text

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers as nn
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adamax
from keras import backend as K


from tqdm import tqdm
tqdm.pandas()

TOK_PATH = 'model_weights/tokenizer.pickle'
MAXLEN = 200 # more than 99th percentile

relevant_cols = ['comment_text',
                 'target', 'severe_toxicity',
                 'identity_attack', 'insult',
                 'threat']


def get_fasttext_embeddings(word_index,
                            ft_path='model_weights/cc.en.300.bin.gz',
                            cached_path='model_weights/embedding_matrix.npy'):
    try:
        return np.load(cached_path)
    except FileNotFoundError:
        ft_model = fasttext.load_facebook_vectors(ft_path)
        vocab_size = len(word_index)
        embedding_matrix = np.zeros((vocab_size + 1, 300))
        for word, i in tqdm(word_index.items()):
            embedding_matrix[i] = ft_model[word]
            pass
        np.save(cached_path, embedding_matrix)
        return embedding_matrix


def make_model(embedding_matrix):
    inps = nn.Input(shape=(MAXLEN, ))
    x = nn.Embedding(*embedding_matrix.shape,
                     weights=[embedding_matrix],
                     trainable=False)(inps)
    x = nn.SpatialDropout1D(0.1)(x)
    x = nn.Bidirectional(nn.CuDNNGRU(256, return_sequences=True))(x)
    gap = nn.GlobalAveragePooling1D()(x)
    gmp = nn.GlobalMaxPooling1D()(x)
    x = nn.concatenate([gap, gmp])
    x = nn.Dropout(0.5)(x)
    x = nn.BatchNormalization()(x)
    x = nn.Dense(5, activation='sigmoid')(x)
    model = Model(inputs=inps, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adamax(decay=1e-6))
    return model


train_data = pd.read_csv('sentiment_data/train.csv', usecols=relevant_cols)
X_train = train_data.comment_text.progress_apply(lambda comment:
                                                 clean_text(comment)).astype(str)

try:
    with open(TOK_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
        pass
except FileNotFoundError:
    tokenizer = Tokenizer(lower=False, oov_token='_UNK_')
    tokenizer.fit_on_texts(X_train)
    with open(TOK_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, MAXLEN)
y_train = train_data[relevant_cols[1:]].astype(float).values


# %%
BATCH_SIZE = 64
EPOCHS = 50

embedding_matrix = get_fasttext_embeddings(tokenizer.word_index)
# %%

K.clear_session()
model = make_model(embedding_matrix)

es_c = EarlyStopping(monitor='val_loss', patience=3, mode='min')
mc_c = ModelCheckpoint(f'model_weights/model.h5',
                       monitor='val_loss',
                       save_best_only=True,
                       mode='min', verbose=1)


model.fit(X_train, y_train,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS, initial_epoch=2,
                 callbacks=[es_c, mc_c], 
                 validation_split=0.1, verbose=1)

# %%
