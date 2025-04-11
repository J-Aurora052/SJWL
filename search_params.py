import itertools
from train import train, split_train_val
from model import NeuralNet
from load_data import load_cifar10


def preprocess_data(X):
    X = X.reshape(X.shape[0], -1)
    X = X / 255.0
    return X

def search():
    cifar_dir = './cifar-10-batches-py'
    X_train_all, y_train_all, _, _ = load_cifar10(cifar_dir)
    X_train_all = preprocess_data(X_train_all)
    X_train, y_train, X_val, y_val = split_train_val(X_train_all, y_train_all)


    learning_rates = [1e-2, 5e-3]
    hidden_layer_options = [[256, 128], [512, 256]]
    regs = [0.0, 1e-3, 1e-2]

    best_acc = 0
    best_setting = None

    for lr, hidden_sizes, reg in itertools.product(learning_rates, hidden_layer_options, regs):
        print(f"\n🔍 Trying: LR={lr}, hidden={hidden_sizes}, reg={reg}")
        model = NeuralNet(input_size=3072, hidden_sizes=hidden_sizes, output_size=10, activation='relu')
        train(model, X_train, y_train, X_val, y_val,
              epochs=5, batch_size=128, learning_rate=lr, reg=reg, save_path="temp.pkl")

        # 用训练完的模型评估验证集准确率
        _, acc = model.compute_loss_and_accuracy(X_val, y_val)
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_setting = (lr, hidden_sizes, reg)

    print(f"\n Best Hyperparameters → LR={best_setting[0]}, hidden={best_setting[1]}, reg={best_setting[2]}, Acc={best_acc:.4f}")

if __name__ == '__main__':
    search()