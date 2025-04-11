本次实验旨在手工构建并训练一个用于 CIFAR-10 数据集分类任务的三层卷积神经网络。

# 环境依赖
- Python 3.6+
- NumPy
- Matplotlib
- pickle




# 项目结构
```
├── cifar-10-batches-py/ # CIFAR-10 数据集文件夹（需手动下载）
├── load_data.py # 数据加载
├── model.py # 模型定义
├── train.py # 训练
├── test.py # 测试
├── search_params.py # 超参数搜索工具
├── visualize_weights.py # 权重可视化工具
├── temp.pkl # 训练临时缓存文件
```

# 流程

## 数据准备
1. 下载 [CIFAR-10 python 版本数据集](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. 解压到项目根目录的 `cifar-10-batches-py` 文件夹

## 使用默认参数训练
python train.py \
    --batch_size 128 \
    --epochs 20 \
    --learning_rate 0.01 \
    --save_path best_model.pkl

## 使用预训练权重测试
python test.py \
    --model_path best_model.pkl

## 预训练权重
模型权重: [网盘链接](https://pan.baidu.com/s/1OfJdZLPSb_VAEsTR8NNWYA?pwd=k5x8 提取码: k5x8)


## 可视化分析
生成可视化文件
python visualize_weights.py --output first_layer_filters.png

## 实验结果
- 验证集准确率：46.5%
- 测试集准确率：47.3%

# 改进方向
1. 添加 BatchNorm 和残差连接
2. 使用 Leaky ReLU 激活函数
3. 调整参数初始化策略
```
