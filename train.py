import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import NeuralNet
from load_data import load_cifar10

def preprocess_data(X):
    X = X.reshape(X.shape[0], -1)
    X = X / 255.0
    return X

def split_train_val(X, y, val_ratio=0.1):
    num_val = int(X.shape[0] * val_ratio)
    return X[num_val:], y[num_val:], X[:num_val], y[:num_val]

def train(model, X_train, y_train, X_val, y_val,
          epochs=20, batch_size=128, learning_rate=1e-2, reg=1e-3, lr_decay=0.95,
          save_path="best_model.pkl"):
    best_val_acc = 0


    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train = X_train[idx]
        y_train = y_train[idx]


        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            model.forward(X_batch)
            model.backward(y_batch, reg=reg)
            model.update(learning_rate)


        train_loss, train_acc = model.compute_loss_and_accuracy(X_train, y_train, reg)
        val_loss, val_acc = model.compute_loss_and_accuracy(X_val, y_val, reg)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}
            with open(save_path, 'wb') as f:
                pickle.dump(best_params, f)
            print(f"New best model saved with val_acc={val_acc:.4f}")


        learning_rate *= lr_decay


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Train Loss', marker='o')
    plt.plot(range(epochs), val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(range(epochs), val_accuracies, label='Val Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_loss_accuracy.png")


    print("训练完成！")

if __name__ == '__main__':
    cifar_dir = './cifar-10-batches-py'
    X_train_all, y_train_all, X_test, y_test = load_cifar10(cifar_dir)


    X_train_all = preprocess_data(X_train_all)
    X_test = preprocess_data(X_test)


    X_train, y_train, X_val, y_val = split_train_val(X_train_all, y_train_all)


    model = NeuralNet(input_size=3072, hidden_sizes=[256, 128], output_size=10, activation='relu')


    train(model, X_train, y_train, X_val, y_val,
          epochs=20, batch_size=128, learning_rate=1e-2, reg=1e-3)
