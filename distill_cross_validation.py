#!/usr/bin/env python
# distill_dla_mobilenetv2.py
"""
知识蒸馏实现：DLA(教师) → MobileNetV2(学生)
基于 pytorch-cifar 仓库
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from models import *
from utils import progress_bar
from sklearn.model_selection import KFold

# 配置参数
parser = argparse.ArgumentParser(description='知识蒸馏：DLA->MobileNetV2')
parser.add_argument('--lr', default=0.05, type=float, help='学习率')
parser.add_argument('--resume', '-r', action='store_true', help='从检查点恢复')
parser.add_argument('--epochs', default=200, type=int, help='训练轮数')
parser.add_argument('--alpha', default=0.7, type=float, help='硬损失权重')
parser.add_argument('--temp', default=5.0, type=float, help='蒸馏温度')
parser.add_argument('--batch_size', default=128, type=int, help='批大小')
args = parser.parse_args()

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # 最佳测试准确率
start_epoch = 0  # 起始轮数

# 数据准备
print('==> 准备数据..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# 初始化模型
print('==> 构建模型..')

# 教师模型：DLA (Deep Layer Aggregation)
teacher = DLA()
teacher = teacher.to(device)
if device == 'cuda':
    teacher = torch.nn.DataParallel(teacher)
    cudnn.benchmark = True

# 加载预训练教师模型
teacher_ckpt = './checkpoint/dla.pth'
if os.path.exists(teacher_ckpt):
    print(f'==> 加载教师模型 {teacher_ckpt}..')
    teacher.load_state_dict(torch.load(teacher_ckpt))
else:
    raise FileNotFoundError(f"教师模型权重不存在: {teacher_ckpt}")

teacher.eval()  # 固定教师模型
for param in teacher.parameters():
    param.requires_grad = False

# 学生模型：MobileNetV2
student = MobileNetV2()
student = student.to(device)
if device == 'cuda':
    student = torch.nn.DataParallel(student)

# 蒸馏损失函数
class DistillationLoss(nn.Module):
    def __init__(self, alpha, temp):
        super().__init__()
        self.alpha = alpha
        self.temp = temp
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels):
        # 硬损失（标准交叉熵）
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 软损失（KL散度）
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits/self.temp, dim=1),
            F.softmax(teacher_logits/self.temp, dim=1)
        ) * (self.temp ** 2)  # 温度缩放补偿
        
        # 组合损失
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# 初始化蒸馏损失
distill_loss = DistillationLoss(alpha=args.alpha, temp=args.temp).to(device)

# 优化器和调度器
optimizer = optim.SGD(student.parameters(), lr=args.lr, 
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# 从检查点恢复
if args.resume:
    print('==> 从检查点恢复..')
    assert os.path.isdir('checkpoint'), '错误：无检查点目录!'
    checkpoint = torch.load('./checkpoint/mobilenetv2_distilled.pth')
    student.load_state_dict(checkpoint['student'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# 测试函数
def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 
                         f'Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')

    acc = 100.*correct/total
    return acc

# 训练函数
def distill_train(epoch):
    print(f'\nEpoch: {epoch}')
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 动态温度调整（后期增加温度）
    current_temp = args.temp * min(1.0, epoch/args.epochs + 0.2)
    distill_loss.temp = current_temp
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 教师预测
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        # 学生预测
        student_logits = student(inputs)
        
        # 计算蒸馏损失
        loss = distill_loss(student_logits, teacher_logits, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计信息
        train_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 显示进度
        progress_bar(batch_idx, len(trainloader), 
                    f'Loss: {train_loss/(batch_idx+1):.3f} | '
                    f'Acc: {100.*correct/total:.3f}% | '
                    f'Temp: {current_temp:.1f}')

# 交叉验证参数
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_results = []

# 获取全部训练数据和标签
all_data = trainset.data
all_targets = np.array(trainset.targets)

for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
    print(f'==> Fold {fold+1}/{num_folds}')
    # 构建本折的训练集和验证集
    train_data = all_data[train_idx]
    train_targets = all_targets[train_idx]
    val_data = all_data[val_idx]
    val_targets = all_targets[val_idx]

    # 构建Dataset对象
    train_foldset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transform_train)
    val_foldset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transform_test)
    train_foldset.data = train_data
    train_foldset.targets = train_targets.tolist()
    val_foldset.data = val_data
    val_foldset.targets = val_targets.tolist()

    trainloader = torch.utils.data.DataLoader(
        train_foldset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(
        val_foldset, batch_size=100, shuffle=False, num_workers=2)

    # 初始化学生模型
    student = MobileNetV2().to(device)
    if device == 'cuda':
        student = torch.nn.DataParallel(student)
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    distill_loss = DistillationLoss(alpha=args.alpha, temp=args.temp).to(device)

    best_val_acc = 0
    for epoch in range(args.epochs):
        distill_train(epoch)  # 需修改为用本折的trainloader
        val_acc = test(student, valloader)  # 需修改为用本折的valloader
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存本折最佳模型
            torch.save(student.state_dict(), f'./checkpoint/mobilenetv2_fold{fold+1}.pth')
    fold_results.append(best_val_acc)
    print(f'Fold {fold+1} best val acc: {best_val_acc:.2f}%')

print(f'交叉验证平均准确率: {np.mean(fold_results):.2f}%')

# 训练循环
for epoch in range(start_epoch, start_epoch+args.epochs):
    distill_train(epoch)
    test_acc = test(student)
    scheduler.step()
    
    # 保存最佳模型
    if test_acc > best_acc:
        print(f'==> 保存最佳学生模型 (准确率: {test_acc:.2f}%)...')
        state = {
            'student': student.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
            'teacher': 'DLA',
            'temp': args.temp,
            'alpha': args.alpha
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/mobilenetv2_distilled.pth')
        best_acc = test_acc

    # 保存最新检查点
    torch.save({
        'student': student.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'acc': test_acc,
        'epoch': epoch,
    }, './checkpoint/mobilenetv2_latest.pth')

print(f'最佳准确率: {best_acc:.2f}%')