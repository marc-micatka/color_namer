import numpy as np
import random
import pandas as pd
import string
import os

import gensim

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, LSTM, Masking, TimeDistributed
from keras.models import Sequential
from matplotlib import pyplot as plt


## Utility Functions
def hex_to_rgb(value):
    if type(value) is str:
        value = value.lstrip()
        value = value.lstrip('#')
        value = tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    else:
        value = np.nan
    return value


def display_color(rgb):
    """Display a color from RGB values.
    :param rgb:
    :return:
    """
    print(rgb)
    rgb = rgb/255
    (h, w) = (64, 64)
    color_swatch = np.zeros((h, w, 3))
    for row in range(h):
        for col in range(w):
            color_swatch[row, col] = rgb

    plt.imshow(color_swatch)
    plt.show()


class ColorNamer:
    def __init__(self, training_loc='csv', verbose=True):
        self.verbose = verbose
        self.model = None
        self.training_folder_name = training_loc
        self.training_data = None
        self.training_data_file = 'all_names.csv'

        self.vector_file = "GoogleNews-vectors-negative300.bin"

        ## Load data into self.training_data
        if self.verbose:
            print("Generating Data from CSV Files...")
        self.generate_data()

        if self.verbose:
            print("Loading Data from CSV...")
        self.load_data()

    def generate_data(self):
        if not os.path.isfile(self.training_data_file):
            self.load_data()
            self.training_data.to_csv(self.training_data_file)

        # load all the datasets
        self.training_data = pd.read_csv(self.training_data_file)

    def load_data(self):
        all_dfs = []
        for dirpath, dirnames, all_files in os.walk(self.training_folder_name):
            for filename in all_files:
                temp_df = pd.read_csv(os.path.join(dirpath, filename),
                                      names=["Color", "Code", "Hex"])
                temp_df['Family'] = os.path.splitext(filename)[0]
                all_dfs.append(temp_df)
        merged_df = pd.concat(all_dfs, ignore_index=True)

        # Calculate RGB Values
        merged_df['RGB'] = merged_df.apply(lambda row: hex_to_rgb(row['Hex']), axis=1)
        merged_df = merged_df.replace('', np.nan)
        merged_df = merged_df.dropna()

        # Split RGB Values to Separate Columns
        new_col_list = ['R', 'G', 'B']
        for n, col in enumerate(new_col_list):
            merged_df[col] = merged_df['RGB'].apply(lambda rgb: rgb[n])

        self.training_data = merged_df

    def train_colors_to_words(self):
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(self.vector_file,
                                                                     binary=True)
        (self.model, self.max_tokens, self.dim) = self.words_to_color()

    def words_to_color(self):
        max_tokens = max([len(x.split()) for x in self.training_data['Color'].tolist()])

        (total_tokens, dim) = self.w2v.vectors.shape
        avg_vec = np.mean(self.w2v.vectors, axis=0)
        empty_vec = np.zeros(dim)

        (X_train, Y_train) = ([], [])

        for index in range(500):
            tokens = self.training_data['Color'][index].split()

            X = []
            i = 0

            for token in tokens:
                if token in self.w2v:
                    X.append(self.w2v[token])
                else:
                    X.append(avg_vec)

                i += 1

            while i < max_tokens:
                X.append(empty_vec)
                i += 1

            X_train.append(np.array(X))
            y = [self.training_data['R'][index],
                 self.training_data['G'][index],
                 self.training_data['B'][index]]
            Y_train.append(np.array(y))

        idxs = list(range(len(X_train)))
        random.shuffle(idxs)
        (final_X_train, final_Y_train) = ([], [])
        for idx in idxs:
            final_X_train.append(X_train[idx])
            final_Y_train.append(Y_train[idx])

        (final_X_train, final_Y_train) = (np.array(final_X_train), np.array(final_Y_train))

        self.model = self.build_colors2words_model()
        checkpoint = ModelCheckpoint("words2color_params.h5", monitor="val_loss", verbose=1, save_best_only=True)
        self.model.fit(final_X_train, final_Y_train, epochs=10, validation_split=0.1, callbacks=[checkpoint])
        self.model.load_weights("words2color_params.h5")
        return (self.model, max_tokens, dim)

    def generate_color(self, words, display=True):
        X_test = np.zeros((self.max_tokens, self.dim))
        for (i, word) in enumerate(words):
            X_test[i] = self.w2v[word]

        rgb = self.model.predict(np.array([X_test]))[0]
        if display:
            display_color(rgb)

    def build_colors2words_model(self):
        model = Sequential()
        model.add(Masking(mask_value=-1, input_shape=(self.max_tokens, 3)))
        model.add(LSTM(256, return_sequences=True))
        model.add(TimeDistributed(Dense(self.dim)))

        model.compile(loss="mse", optimizer="adam")
        return model


if __name__ == "__main__":
    color_namer = ColorNamer()
    color_namer.train_colors_to_words()
    color_namer.generate_color(["purple"])