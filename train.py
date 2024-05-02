import os
import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models


def load_training_data(data_folder="data/training"):
    x_list = []
    y_list = []
    for chord in os.listdir(data_folder):
        if not chord.startswith("."):
            chord_folder = os.path.join(data_folder, chord)
            if os.path.isdir(chord_folder):
                for wav_file in os.listdir(chord_folder):
                    file_path = os.path.join(chord_folder, wav_file)
                    audio_data, sample_data = preprocessing.load_and_normalize(file_path)

                    x_list.append(preprocessing.get_features(audio_data, sample_data))
                    y_list.append(chord)

    return np.array(x_list), np.array(y_list)


def train(x, y):
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    cnn_model = Sequential([
        Input(shape=(x.shape[1],)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(8, activation='softmax')
    ])

    cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

    cnn_model.fit(x,
                  y,
                  epochs=50,
                  batch_size=32,
                  validation_data=(x, y),
                  verbose=2,
                  callbacks=[early_stopping]
                  )

    return cnn_model