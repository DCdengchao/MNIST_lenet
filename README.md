# MNIST_lenet

基于 LeNet-5 的 MNIST 手写数字识别项目。该项目使用 PyTorch 搭建 LeNet-5 卷积神经网络，对 MNIST 数据集中的 0–9 手写数字图像进行分类训练与测试。

## 项目简介

本项目实现了一个经典的 LeNet-5 网络结构，用于完成 MNIST 手写数字识别任务。主要流程包括：

1. 读取 MNIST 原始二进制数据文件；
2. 构建自定义 `Dataset` 数据集；
3. 使用 LeNet-5 网络进行模型训练；
4. 保存训练后的模型权重；
5. 加载模型并在测试集上进行识别与准确率评估；
6. 可选保存部分预测结果图像。

## 项目结构

```text
MNIST_lenet/
├── data/                 # MNIST 数据集文件目录
├── mnist_result/         # 测试结果图片保存目录
├── LeNet5.py             # LeNet-5 网络结构定义
├── myData.py             # 自定义 MNIST Dataset
├── readMnist.py          # MNIST 原始 idx 文件读取与解析
├── train.py              # 模型训练脚本
├── test.py               # 模型测试脚本
├── lenet5.pth            # 训练好的模型权重
└── README.md             # 项目说明文档
```

## 环境依赖

建议使用 Python 3.8 及以上版本。

主要依赖如下：

```bash
pip install torch torchvision numpy matplotlib opencv-python
```

## 数据集说明

本项目使用 MNIST 原始格式数据文件，包括：

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

默认数据路径在 `readMnist.py` 中设置为：

```python
fpath = '/home/dcc/Project/MNIST/data/'
```

如果你的数据集路径不同，需要修改 `readMnist.py` 中的 `fpath`：

```python
fpath = './data/'
```

并确保 `data/` 目录下包含 MNIST 的四个原始数据文件。

## 模型结构

项目中的 LeNet-5 网络主要由以下部分组成：

```text
输入图像：1 × 28 × 28

Conv1: 1 → 6, kernel_size=5
Sigmoid
MaxPool2d

Conv2: 6 → 16, kernel_size=5
Sigmoid
MaxPool2d

Flatten

FC1: 16 × 4 × 4 → 120
Sigmoid

FC2: 120 → 84
Sigmoid

FC3: 84 → 10
```

最终输出 10 个类别，对应数字 `0–9`。

## 训练模型

运行以下命令开始训练：

```bash
python train.py
```

训练脚本主要流程如下：

1. 读取 MNIST 训练集图像和标签；
2. 使用 `Mnist` 类构建训练数据集；
3. 使用 `DataLoader` 加载数据；
4. 初始化 LeNet-5 模型；
5. 使用交叉熵损失函数 `CrossEntropyLoss`；
6. 使用 Adam 优化器；
7. 训练 100 个 Epoch；
8. 将模型权重保存为 `lenet5.pth`。

训练完成后，会在项目根目录生成：

```text
lenet5.pth
```

## 测试模型

运行以下命令进行测试：

```bash
python test.py
```

测试脚本会加载已经训练好的模型权重：

```text
lenet5.pth
```

然后在 MNIST 测试集上进行预测，并输出模型准确率：

```text
准确率：0.xxxxxx
```

如果需要保存部分测试图片，可以在 `test.py` 中将：

```python
showimg = False
```

修改为：

```python
showimg = True
```

程序会将前 20 张测试图片保存到：

```text
mnist_result/
```

保存的图片文件名中包含预测类别和真实类别，例如：

```text
img_0_pred_7_true_7.png
```

## 核心文件说明

### `LeNet5.py`

定义 LeNet-5 卷积神经网络结构，包括两个卷积层、两个池化层和三个全连接层。

### `readMnist.py`

用于解析 MNIST 原始二进制数据文件，包括图像文件和标签文件。

主要函数包括：

```python
load_train_images()
load_train_labels()
load_test_images()
load_test_labels()
```

### `myData.py`

自定义 PyTorch 数据集类 `Mnist`，用于将 MNIST 图像和标签封装成可被 `DataLoader` 调用的数据格式。

同时对图像进行如下预处理：

```python
transforms.ToTensor()
transforms.Normalize(mean=[0.5], std=[0.5])
```

### `train.py`

模型训练脚本，负责加载训练数据、训练 LeNet-5 模型并保存权重文件。

### `test.py`

模型测试脚本，负责加载训练好的模型权重，并在测试集上计算识别准确率。

## 使用流程

完整运行流程如下：

```bash
# 1. 安装依赖
pip install torch torchvision numpy matplotlib opencv-python

# 2. 准备 MNIST 数据集
# 将四个 MNIST 原始文件放入 data/ 目录

# 3. 修改 readMnist.py 中的数据路径
fpath = './data/'

# 4. 训练模型
python train.py

# 5. 测试模型
python test.py
```

## 注意事项

1. 运行前需要确认 MNIST 数据集路径是否正确。
2. 如果没有 GPU，需要将 `train.py` 中的 `.cuda()` 相关代码修改为 CPU 版本。
3. `train.py` 中当前 batch size 为 1，训练速度可能较慢，可以根据显存情况适当调大。
4. `test.py` 默认在 CPU 上加载模型进行测试。
5. 如果修改模型结构，需要重新训练并生成新的 `lenet5.pth` 权重文件。

## 项目特点

* 使用 PyTorch 实现 LeNet-5；
* 支持 MNIST 原始 idx 文件解析；
* 包含完整的训练和测试流程；
* 支持保存部分预测结果图像；
* 代码结构简单，适合深度学习入门学习。

## 后续改进方向

可以进一步优化以下内容：

1. 增加 `requirements.txt`，方便快速安装环境；
2. 增加命令行参数，例如 batch size、epoch、learning rate；
3. 支持 CPU/GPU 自动切换；
4. 增加训练过程中的 loss 曲线和准确率曲线可视化；
5. 增加模型评估指标，如混淆矩阵、分类报告等；
6. 将数据路径改为相对路径，提升项目可移植性。

## License

本项目仅用于学习与实验。
