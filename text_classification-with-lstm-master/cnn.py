import pandas as pd
import torch
from torch import nn
import jieba
from gensim.models import Word2Vec
import numpy as np
from data_set import load_tsv
from torch.utils.data import DataLoader, TensorDataset


# 数据读取
def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [[line.strip()] for line in f.readlines()]
        return data

train_x = load_txt('train.txt')
test_x = load_txt('test.txt')
train = train_x + test_x
X_all = [i for x in train for i in x]

_, train_y = load_tsv("./data/train.tsv")
_, test_y = load_tsv("./data/test.tsv")
# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences=X_all, vector_size=100, window=5, min_count=1, workers=4)


# 将文本转换为Word2Vec向量表示
def text_to_vector(text):
    vector = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
    return sum(vector) / len(vector) if vector else [0] * word2vec_model.vector_size


X_train_w2v = [[text_to_vector(text)] for line in train_x for text in line]
X_test_w2v = [[text_to_vector(text)] for line in test_x for text in line]

# 将词向量转换为PyTorch张量
X_train_array = np.array(X_train_w2v, dtype=np.float32)
X_train_tensor = torch.Tensor(X_train_array)
X_test_array = np.array(X_test_w2v, dtype=np.float32)
X_test_tensor = torch.Tensor(X_test_array)
# 使用DataLoader打包文件
train_dataset = TensorDataset(X_train_tensor, torch.LongTensor(train_y))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, torch.LongTensor(test_y))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# 定义cnn模型
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)  # 第一个一维卷积层
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # 第二个一维卷积层
        self.fc = nn.Linear(64, output_size)  # 全连接层

    def forward(self, x):
        x = x.permute(0, 2, 1)  # # Conv1d期望输入格式为(batch_size, channels, sequence_length)
        x = torch.relu(self.conv1(x))  # 第一个卷积层的激活函数
        x = torch.relu(self.conv2(x))  # 第二个卷积层的激活函数
        x = torch.max_pool1d(x, kernel_size=x.size(2))  # 全局最大池化
        x = x.squeeze(2)  # 移除最后一个维度
        x = self.fc(x)  # 全连接层
        return x


# 定义CNN模型、损失函数和优化器
input_size = word2vec_model.vector_size  # 输入大小为 Word2Vec 向量大小
output_size = 2  # 输出大小
cnn_model = CNNModel(input_size, output_size)  # 创建 CNN 模型对象
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0002)  # Adam 优化器

if __name__ == "__main__":
    # 训练和评估
    num_epochs = 100  # 迭代次数
    log_interval = 100  # 日志打印间隔
    loss_min = 100  # 最小损失值
    for epoch in range(num_epochs):
        cnn_model.train()  # 设置模型为训练模式
        for batch_idx, (data, target) in enumerate(train_loader):
            outputs = cnn_model(data)  # 模型前向传播
            loss = criterion(outputs, target)  # 计算损失

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if batch_idx % log_interval == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, batch_idx, len(train_loader), loss.item()))
            if loss.item() < loss_min:
                loss_min = loss.item()
                torch.save(cnn_model, 'cnn_model.pth')

    # 评估
    with torch.no_grad():
        cnn_model.eval()
        correct = 0
        total = 0
        for data, target in test_loader:
            outputs = cnn_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        accuracy = correct / total
        print('测试准确率（CNN模型）：{:.2%}'.format(accuracy))