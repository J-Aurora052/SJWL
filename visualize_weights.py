import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def visualize_weights(best_model_path="best_model.pkl"):
    with open(best_model_path, 'rb') as f:
        best_params = pickle.load(f)


    W1 = best_params['W1']
    input_dim, hidden_dim = W1.shape

    if input_dim != 32 * 32 * 3:
        raise ValueError("输入维度与预期不符，请检查数据预处理。")


    num_display = min(hidden_dim, 16)
    fig, axes = plt.subplots(1, num_display, figsize=(num_display * 1.8, 2.5))
    for i in range(num_display):
        filt = W1[:, i].reshape(32, 32, 3)
        filt_min, filt_max = filt.min(), filt.max()
        filt_norm = (filt - filt_min) / (filt_max - filt_min + 1e-8)
        axes[i].imshow(filt_norm)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"F {i + 1}", fontsize=10)
    plt.suptitle("第一层滤波器可视化", fontsize=14)
    plt.tight_layout()
    plt.savefig("first_layer_filters.png", dpi=200)



    W2 = best_params['W2']
    plt.figure(figsize=(6, 4))
    plt.hist(W2.flatten(), bins=30, color="blue", alpha=0.7)
    plt.title("第二层权重分布")
    plt.xlabel("权重数值")
    plt.ylabel("频率")
    plt.savefig("second_layer_weights_distribution.png", dpi=200)



    W3 = best_params['W3']
    plt.figure(figsize=(6, 4))
    plt.hist(W3.flatten(), bins=30, color="green", alpha=0.7)
    plt.title("第三层权重分布")
    plt.xlabel("权重数值")
    plt.ylabel("频率")
    plt.savefig("third_layer_weights_distribution.png", dpi=200)



if __name__ == '__main__':
    visualize_weights()
