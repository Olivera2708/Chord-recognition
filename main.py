import train
import test
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics


def create_model():
    x, y = train.load_training_data()
    x_reshaped = x.reshape(x.shape[0], -1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return train.train(x_reshaped, y_encoded)

if __name__ == "__main__":
    if os.path.exists('model.keras'):
        cnn_model = load_model('model.keras')
    else:
        cnn_model = create_model()
        cnn_model.save('model.keras')

    cnn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    x_test, y_test = test.load_test_data()
    cnn_score_test = cnn_model.evaluate(x_test, y_test, verbose=0)
    print("\nTesting Accuracy: ", cnn_score_test[1])