import train
import test
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_training_data():
    x, y = train.load_training_data()
    x_reshaped = x.reshape(x.shape[0], -1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return x_reshaped, y_encoded

def show_cnn_stats(x_train, y_train, x_test, y_test):
    cnn_model = train.train_cnn(x_train, y_train)

    y_pred = np.argmax(cnn_model.predict(x_test), axis=1)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    print(f'Testing Accuracy {round(np.mean(y_pred==y_test)*100, 4)}%')
    print(f"Testing Precision: {round(precision_macro*100, 4)}%")
    print(f"Testing Recall: {round(recall_macro*100, 4)}%")

def show_knn_stats(x_train, y_train, x_test, y_test):
    knn_model = train.train_knn(x_train, y_train)

    y_pred = knn_model.predict(x_test)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    print(f'Testing Accuracy {round(np.mean(y_pred==y_test)*100, 4)}%')
    print(f"Testing Precision: {round(precision_macro*100, 4)}%")
    print(f"Testing Recall: {round(recall_macro*100, 4)}%")

def show_svm_stats(x_train, y_train, x_test, y_test):
    svm_model = train.train_svm(x_train, y_train)

    y_pred = svm_model.predict(x_test)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    print(f'Testing Accuracy {round(np.mean(y_pred==y_test)*100, 4)}%')
    print(f"Testing Precision: {round(precision_macro*100, 4)}%")
    print(f"Testing Recall: {round(recall_macro*100, 4)}%")

def show_rfc_stats(x_train, y_train, x_test, y_test):
    rfc_model = train.train_rfc(x_train, y_train)

    y_pred = rfc_model.predict(x_test)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    print(f'Testing Accuracy {round(np.mean(y_pred==y_test)*100, 4)}%')
    print(f"Testing Precision: {round(precision_macro*100, 4)}%")
    print(f"Testing Recall: {round(recall_macro*100, 4)}%")

def show_plot(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    x_train, y_train = load_training_data()
    x_test, y_test = test.load_test_data()

    print("----- Convolutional Neural Network -----") #Odlicno
    show_cnn_stats(x_train, y_train, x_test, y_test)
    print("\n--------- K-Nearest Neighbors ---------")
    show_knn_stats(x_train, y_train, x_test, y_test)
    print("\n------- Support Vector Machines -------") #Odlicno
    show_svm_stats(x_train, y_train, x_test, y_test)
    print("\n----- Random Forest Classifier -----") #Odlicno
    show_rfc_stats(x_train, y_train, x_test, y_test)