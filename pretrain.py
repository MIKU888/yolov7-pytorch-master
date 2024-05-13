import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from nets.yolo_unet import YoloUnet  

# 设定超参数
num_epochs = 10
learning_rate = 0.001
batch_size = 4


def getMask(x, maskRatio):
    tempx = torch.torch.flatten(x, start_dim=1, end_dim=-1)  # 变成[batch,640*640*3]
    b, l = tempx.shape
    noise = torch.randn(b, l, device=x.device)
    ids_sort = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_sort, dim=1)
    mask = torch.zeros(b, l).to(x.device)
    len_keep = l - int(l * maskRatio)
    mask[:, :len_keep] = 1
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # maskedX = torch.mul(tempx,mask).unsqueeze(dim=1)
    mask = mask.reshape(batch_size, 3, 640, -1)
    return mask


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载训练集
train_dataset = datasets.ImageFolder(root='../pretrain', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = len(train_dataset.classes)  # 根据数据集确定类别数
model = YoloUnet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs = data[0].to(device)

        mask = getMask(inputs, 0.3)

        optimizer.zero_grad()  # 梯度清零
        outputs = model(mask)  # 前向传播
        loss = criterion(outputs * (1 - mask), inputs * (1 - mask))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item() * inputs.size(0)
        print(i)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'unet_model.pth')
