import torch 


def evaluate(model, batch_size=32):
    # Pretpostavljamo da imamo X_test i y_test
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model.eval()  # Model u modusu evaluacije
    correct = 0
    total = 0

    with torch.no_grad():  # Nema unazad propagacije
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i + batch_size]
            labels = y_test[i:i + batch_size]
            
            outputs = model(inputs)  # Dobijanje izlaza
            _, predicted = torch.max(outputs, 1)  # Predikcija klasa
            total += labels.size(0)  # Ukupno uzoraka
            correct += (predicted == labels).sum().item()  # Broj tačnih predikcija

    accuracy = correct / total  # Tačnost modela
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")  # Izračunavanje tačnosti
