import os
import preprocessing
from model import ChordRecognitionModel
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
                    spectrogram = preprocessing.spectrogram(audio_data, sample_data)
                    x_list.append(spectrogram)
                    y_list.append(chord)

    return np.array(x_list), np.array(y_list)


def create_model(x, y):
    x = np.array([spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1]) for spectrogram in x])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    input_shape = x.shape[1:]
    num_classes = len(set(y))
    return ChordRecognitionModel(input_shape, num_classes), x, y


def train(model, x, y, batch_size=64, num_epochs=10, learning_rate=0.001): #dobro batch_size=32, num_epochs=10, learning_rate=0.001
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(x), batch_size):
            inputs = x[i:i + batch_size]
            labels = y[i:i + batch_size]
            
            optimizer.zero_grad()  # Nuliranje gradijenata
            outputs = model(inputs)  # Prolazak podataka kroz model
            loss = criterion(outputs, labels)  # Izračunavanje gubitka
            
            # Backward i ažuriranje optimizatora
            loss.backward() 
            optimizer.step()  
            
            # Računanje gubitka i tačnosti
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Dobijanje predikcija
            total += labels.size(0)  # Ukupno uzoraka
            correct += (predicted == labels).sum().item()  # Broj tačnih predikcija
        
        # Računanje srednjeg gubitka i tačnosti
        epoch_loss = running_loss / (len(x) / batch_size)  # Prosečan gubitak
        accuracy = correct / total  # Tačnost
        
        # Izveštavanje o rezultatima epohe
        print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    return model