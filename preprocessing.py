import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def pad_audio(audio_data, desired_seconds, sample_rate=22050):
    desired_length = desired_seconds * sample_rate
    current_length = len(audio_data)
    if current_length < desired_length:
        pad_width = desired_length - current_length
        return np.pad(audio_data, (0, pad_width), mode='constant', constant_values=0)
    elif current_length > desired_length:
        return audio_data[:desired_length]
    return audio_data

def normalize(audio_data):
    return audio_data / np.max(np.abs(audio_data))

def load_and_normalize(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    audio_data = normalize(audio_data)
    audio_data = pad_audio(audio_data, 5)
    return audio_data, sample_rate

def spectrogram(audio_data, sample_rate, n_mels=128, n_fft=2048, hop_length=128):
    return librosa.feature.melspectrogram(
        y=audio_data, 
        sr=sample_rate, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length
    )

def show_spectogram(spectrogram, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title("Spektrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def mfcc(audio_data, sample_rate, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return np.pad(mfcc, ((0, 0), (0, 862 - 216)), mode='constant')

def show_mfcc(mfccs, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.title("Mel-frekvencijski cepstralni koeficijenti")
    plt.colorbar()
    plt.show()

def get_features(audio_data, sample_data):
    features = []
    features.extend(spectrogram(audio_data, sample_data))
    features.extend(mfcc(audio_data, sample_data))
    return features