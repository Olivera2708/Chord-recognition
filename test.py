import os
import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_test_data(data_folder="data/test"):
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

    x = np.array(x_list)
    y = np.array(y_list)
    x = x.reshape(x.shape[0], -1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return x, y
