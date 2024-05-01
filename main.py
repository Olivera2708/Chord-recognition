import train
import test
import torch
import os


def create_model():
    x, y = train.load_training_data()
    model, x, y = train.create_model(x, y)
    return train.train(model, x, y)

if __name__ == "__main__":
    if os.path.exists('model.pth'):
        model = torch.load('model.pth')
    else:
        model = create_model()
        torch.save(model, 'model.pth')

    model.eval()
    x_test, y_test = test.load_test_data()
    test.test(model, x_test, y_test)