import train
import test
import torch
import os

if __name__ == "__main__":
    if os.path.exists('model.pth'):
        model = torch.load('model.pth')
    else:
        x, y = train.load_training_data()
        model, x, y = train.create_model(x, y)
        model = train.train(model, x, y)
        torch.save(model, 'model.pth')

    model.eval()
    x_test, y_test = test.load_test_data()
    test.test(model, x_test, y_test)