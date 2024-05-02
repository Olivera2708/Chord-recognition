import librosa
import librosa.display
import numpy as np

def normalize(audio_data):
    return audio_data / np.max(np.abs(audio_data))

def load_and_normalize(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    audio_data = normalize(audio_data)
    return audio_data, sample_rate

def mfcc(audio_data, sample_rate, n_mfcc=60):
    return np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)

def chroma(audio_data, sample_rate):
    return np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(audio_data)), sr=sample_rate).T, axis=0)

def tonnetz(audio_data, sample_rate):
    chroma = librosa.feature.chroma_stft(S=np.abs(librosa.stft(audio_data)), sr=sample_rate)
    return np.mean(librosa.feature.tonnetz(chroma=chroma).T, axis=0)

def get_features(audio_data, sample_data):
    features = []
    features.extend(mfcc(audio_data, sample_data))
    features.extend(chroma(audio_data, sample_data))
    features.extend(tonnetz(audio_data, sample_data))
    return features