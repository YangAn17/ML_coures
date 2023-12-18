import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# MNIST dataset
training_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# Find your own hyper-parameters 
num_epochs = 5          # 迭代轮数
batch_size = 100        # 一次训练抓取数据样本数量
weight_decay = 0        # 权值衰减
learning_rate = 0.001   # 学习率
hidden_channel_size_1 = 64
hidden_channel_size_2 = 64

# Data loader
train_dataloader = torch.utils.data.DataLoader(
    training_data, 
    batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# CNN
class CNeuralNet(nn.Module):
    def __init__(self,hidden_channel_size_1,hidden_channel_size_2,num_classes=10):
        super(CNeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, hidden_channel_size_1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(hidden_channel_size_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_channel_size_1, hidden_channel_size_2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(hidden_channel_size_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*hidden_channel_size_2, num_classes)
         # originally 28*28, after two MaxPool2d(2), becomes 7*7
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = CNeuralNet(hidden_channel_size_1,hidden_channel_size_2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

# Train the model
total_step = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
            
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
            
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print(total)
        correct += (predicted == labels).sum().item()
        print(correct)

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))