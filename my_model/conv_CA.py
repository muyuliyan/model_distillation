import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    def forward(self, x):
        out = F.relu(self.conv(x))
        return out

class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=5, padding=2)
    def forward(self, x):
        out = F.relu(self.conv(x))
        return out

class ModelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        out = self.pool(F.relu(self.conv(x)))
        out = F.interpolate(out, size=x.shape[2:])
        return out
    
class ModelD(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=2, dilation=2)
    def forward(self, x):
        out = F.relu(self.conv(x))
        return out

class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 10, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x

class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_models = nn.ModuleList([ModelA(), ModelB(), ModelC(), ModelD()])
        self.big_model = BigModel()
    
    def forward(self, x):
        features = [m(x) for m in self.sub_models]
        combined = torch.cat(features, dim=1)
        out = self.big_model(combined)
        return out

class HybridAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 全局分支 (类似SENet)
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        # 局部分支 (类似CBAM)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(channels*2, channels, 1)
        self.residual = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        global_w = self.global_att(x)
        local_w = self.local_att(x)
        
        global_w = global_w.expand_as(local_w)
        
        combined = torch.cat([global_w, local_w], dim=1)
        fused_w = self.fusion(combined)
        fused_w = fused_w.sigmoid()  # 确保输出在0-1之间
        return x * (fused_w + self.residual) + x

class AttentionEnsembleModel(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()
        self.sub_models = nn.ModuleList([ModelA(), ModelB(), ModelC(), ModelD()])
        
        self.attention = HybridAttention(64, reduction=reduction)
        
        self.big_model = BigModel()
    
    def forward(self, x):
        features = [m(x) for m in self.sub_models]
        combined = torch.cat(features, dim=1)
        
        # 应用融合注意力
        attended = self.attention(combined)
        
        out = self.big_model(attended)
        return out
    
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    save_dir = './saved'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((32, 32)),
    ])

    # 使用CIFAR10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = EnsembleModel().to(device)
    #  model = HybridAttentionEnsembleModel(reduction=8).to(device)   # 如需使用注意力模型请取消注释

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    time = 200  # 训练轮数

    for epoch in range(time):
        model.train()
        train_loss = 0
        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            output = model(batch_data)
            loss = criterion(output, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_data.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证集评估
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                output = model(batch_data)
                loss = criterion(output, batch_target)
                val_loss += loss.item() * batch_data.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == batch_target).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(f'Epoch {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, 'best_model.pth')  # 修正了拼写错误
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with val_acc: {val_acc:.4f} at {model_path}")

    # 加载最佳模型进行测试
    model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded best model from {model_path}")
    else:
        print("Warning: Best model not found, using last model for testing")

    # 测试集评估
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_data, batch_target in test_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            output = model(batch_data)
            loss = criterion(output, batch_target)
            test_loss += loss.item() * batch_data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == batch_target).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print(f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')


