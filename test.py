import os
import preprocessing
import torch
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
    x = np.array([spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1]) for spectrogram in x])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return x, y


def test(model, x, y, batch_size=32):
    total = 0
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            inputs = x[i:i + batch_size]
            labels = y[i:i + batch_size]
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("Test Accuracy:", accuracy * 100, "%")