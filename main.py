import train

if __name__ == "__main__":
    x, y = train.load_training_data()
    model, x, y = train.create_model(x, y)
    train.train(model, x, y)