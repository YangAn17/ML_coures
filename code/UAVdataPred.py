import torch
import pandas as pd
import torch.nn as nn			    #帮助我们创建和训练神经网络
import torch.utils.data as Data	    #导入数据集
import matplotlib.pyplot as plt

# 针对通讯模式和交通模式融合数据集训练
dataFram1_test = pd.read_csv('data\\UVAdataset_csv\\dataset_mult_te.CSV')
dataFram1_train = pd.read_csv('data\\UVAdataset_csv\\dataset_mult_tr.CSV')
data1_value_test = dataFram1_test.values
data1_value_train = dataFram1_train.values

# 划分数据中输入与输出
X1_value_test = data1_value_test[:,:-1].astype(float)       # 前n-1列都是输入X： 最后一列是输出：Y
Y1_value_test = data1_value_test[:,-1:].astype(int)
X1_value_train = data1_value_train[:,:-1].astype(float)
Y1_value_train = data1_value_train[:,-1:].astype(int)

# 转换将csv为可供训练的tensor格式
X1_test, Y1_test = torch.FloatTensor(X1_value_test),torch.IntTensor(Y1_value_test)         # 输入：tensor浮点数，输出：tensor整数
X1_train, Y1_train = torch.FloatTensor(X1_value_train),torch.IntTensor(Y1_value_train)

# 整理为可供批量训练的格式
test_data1 = Data.TensorDataset(X1_test, Y1_test)	        # 将测试集转化为张量后，使用TensorDataset将x y整理到一块
train_data1 = Data.TensorDataset(X1_train, Y1_train)

# 设置训练超参数 
num_epochs = 5               # 迭代轮数
batch_size = 96              # 一次训练抓取数据样本数量
weight_decay = 0             # 权值衰减
learning_rate = 0.001        # 学习率
hidden_channel_size_1 = 32
hidden_channel_size_2 = 32

# 定义一个数据加载器，将数据集进行批量处理
train_loader1 = Data.DataLoader(
    dataset = train_data1,               # 使用的数据集
    batch_size = batch_size,             # 批处理样本大小
    shuffle = True,                      # 随机打乱数据
    num_workers = 0,            
)
test_loader1 = Data.DataLoader(
    dataset = test_data1,
    batch_size = batch_size,
    shuffle = True, 
    num_workers = 0,
)

# 输出维度
for X, y in train_loader1:
    print(f"Shape of X : {X.shape}")
    print(f"Shape of y: {y.shape}")
    break

# CUDA GPU加速训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 使用nn.Module类定义全连接神经网络
class MLPmodel(nn.Module):
    def __init__(self,hidden_channel_size_1,hidden_channel_size_2):
        super(MLPmodel, self).__init__()
        # 定义隐藏层
        self.hidden1 = nn.Linear(
            in_features = 54,                       # 隐藏层的输入，数据的特征数
            out_features = hidden_channel_size_1,   # 隐藏层的输出，神经元的数量
            bias=True,                              # 默认有偏置
        )
        # 定义激活函数
        self.active1 = nn.ReLU()
        self.hidden2 = nn.Linear(
            in_features = hidden_channel_size_1,
            out_features = hidden_channel_size_2,
        )
        self.active2 = nn.ReLU()
        # 定义预测回归层
        self.regression = nn.Linear(
            in_features = hidden_channel_size_2, 
            out_features = 1,
       )
    # 定义网络的前向传播路径
    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        output = self.regression(x)
        return torch.sigmoid(output)                   # sigmoid处理结果为单—二元标签

# 建立模型实例
mlp1 = MLPmodel(hidden_channel_size_1,hidden_channel_size_2).to(device)

# 定义损失和优化器
# criterion = nn.MSELoss()
criterion = nn.BCELoss()                                # 交叉熵损失函数
optimizer = torch.optim.AdamW(                          # AdamW优化器降低对学习率的要求
    mlp1.parameters(),
    lr = learning_rate,
    weight_decay = weight_decay
)
train_loss_all = []                                     # 输出每个批次训练的损失函数

# 进行训练,并输出每次迭代的损失函数
total_step = len(train_loader1)
for epoch in range(num_epochs):
    # 对训练数据的加载器进行迭代计算
    for step, (tr_x, tr_y) in enumerate(train_loader1):
        tr_x = tr_x.to(torch.float32)
        tr_y = tr_y.to(torch.float32)
        tr_x = tr_x.to(device)                          # 数据载入cuda
        tr_y = tr_y.to(device)

        output = mlp1(tr_x)                             # MLP在训练batch上的输出
        train_loss = criterion(output, tr_y)            # 二元分类使用交叉熵损失函数

        optimizer.zero_grad()                           # 每个迭代步的梯度初始化为0
        train_loss.backward()                           # 损失的后向传播
        optimizer.step()                                # 使用梯度进行优化

        train_loss_all.append(train_loss.item())        # 记录每次迭代损失
        
        if (step+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, step+1, total_step, train_loss.item()))

# 测试
mlp1.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for te_x, te_y in test_loader1:
        te_x = te_x.to(device)
        te_y = te_y.to(device)
        outputs = mlp1(te_x)
        # 0-1处理、数据类型处理
        predicted = torch.where(
            outputs > 0.9,
            torch.ones_like(outputs),
            torch.zeros_like(outputs)
        )
        predicted = predicted.to(torch.int64)
        # 计算准确率
        total += te_y.size(0)
        correct += (predicted == te_y).sum().item()

    print('Test Accuracy of the model on test_data1 is: {} %'.format(100 * correct / total))

# 可视化输出损失
plt.figure()                                # 创建新画板
plt.plot(train_loss_all, "r-")	            # 折线图输出
plt.title("Train loss per iteration")       # 表头
plt.show()
