import train
import test
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def create_cnn_model():
    x, y = train.load_training_data()
    x_reshaped = x.reshape(x.shape[0], -1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return train.train(x_reshaped, y_encoded)

def show_cnn_stats():
    if os.path.exists('model_cnn.keras'):
        cnn_model = load_model('model_cnn.keras')
    else:
        cnn_model = create_cnn_model()
        cnn_model.save('model_cnn.keras')

    x_test, y_test = test.load_test_data()
    cnn_score_test = cnn_model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(cnn_model.predict(x_test), axis=1)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    print("Testing Accuracy: ", cnn_score_test[1])
    print("Testing Precision:", precision_macro)
    print("Testing Recall:", recall_macro)


def show_plot(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    show_cnn_stats()