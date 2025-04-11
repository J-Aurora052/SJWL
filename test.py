import pickle
from model import NeuralNet
from load_data import load_cifar10

def preprocess_data(X):
    X = X.reshape(X.shape[0], -1)
    X = X / 255.0
    return X

def test_model(model_path="best_model.pkl"):
    cifar_dir = './cifar-10-batches-py'
    _, _, X_test, y_test = load_cifar10(cifar_dir)
    X_test = preprocess_data(X_test)


    model = NeuralNet(input_size=3072, hidden_sizes=[256, 128], output_size=10, activation='relu')


    with open(model_path, 'rb') as f:
        best_params = pickle.load(f)
        model.params = best_params


    loss, acc = model.compute_loss_and_accuracy(X_test, y_test, reg=0.0)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")

if __name__ == '__main__':
    test_model("best_model.pkl")
