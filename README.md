# model_distillation
This is a knowledge distillation model based on pytorch-cifar repository of kuangliu

Based on pytorch-cifar repository's classic model collection, conduct knowledge distillation learning on classic models, analyze the CIFAR10 classification learning effect of the model, and evaluate the memory advantage of knowledge distillation

## Prerequisites

- Python 3.6+
- PyTorch 1.0+

## Training

```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy

| Model                                                | Acc.   |
| ---------------------------------------------------- | ------ |
| [VGG16](https://arxiv.org/abs/1409.1556)             | 92.64% |
| [ResNet18](https://arxiv.org/abs/1512.03385)         | 93.02% |
| [ResNet50](https://arxiv.org/abs/1512.03385)         | 93.62% |
| [ResNet101](https://arxiv.org/abs/1512.03385)        | 93.75% |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)    | 94.24% |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)    | 94.29% |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)      | 94.43% |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431) | 94.73% |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431) | 94.82% |
| [SimpleDLA](https://arxiv.org/abs/1707.064)          | 94.89% |
| [DenseNet121](https://arxiv.org/abs/1608.06993)      | 95.04% |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)   | 95.11% |
| [DPN92](https://arxiv.org/abs/1707.01629)            | 95.16% |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)          | 95.47% |

## Knowledge distillation script usage

This project supports DLA → MobileNetV2 knowledge distillation training, and the script file is `distill_dla_mobilenetv2.py`。

### Basic commands

```bash
python distill_dla_mobilenetv2.py
```

### Optional parameter description

- `--lr` 学习率（默认 0.05）
- `--epochs` 训练轮数（默认 200）
- `--alpha` 硬损失权重（默认 0.7）
- `--temp` 蒸馏温度（默认 5.0）
- `--batch_size` 批大小（默认 128）
- `--resume` 从最新检查点恢复训练

for example：

```bash
python distill_dla_mobilenetv2.py --lr 0.01 --epochs 100 --alpha 0.5 --temp 4.0 --batch_size 64
```

### Training process:

1. Auto-load `./checkpoint/dla.pth` as the teacher model weight.
2. The best student model is saved to `./checkpoint/mobilenetv2_distilled.pth` during training.
3. The latest checkpoints will be saved to `./checkpoint/mobilenetv2_latest.pth` in each round, and the `--resume` parameter can be used to resume the training.

### View the results

After the training is over, the terminal outputs the best accuracy. Model weights can be found under the 'checkpoint' folder.

---

If you need to customize the data path or model structure, please refer to the parameter settings section of the script to modify it.

---

模型蒸馏 (model_distillation)
本项目基于 kuangliu 的 pytorch-cifar 仓库实现知识蒸馏模型。通过该仓库的经典模型集合，对经典模型进行知识蒸馏学习，分析模型在 CIFAR10 分类任务上的学习效果，并评估知识蒸馏带来的内存优势。

环境要求
Python 3.6+

PyTorch 1.0+

训练方法
text
# 启动训练：
python main.py

# 手动恢复训练（可调整学习率）：
python main.py --resume --lr=0.01
准确率对比
模型	准确率
VGG16	92.64%
ResNet18	93.02%
ResNet50	93.62%
ResNet101	93.75%
RegNetX_200MF	94.24%
RegNetY_400MF	94.29%
MobileNetV2	94.43%
ResNeXt29(32x4d)	94.73%
ResNeXt29(2x64d)	94.82%
SimpleDLA	94.89%
DenseNet121	95.04%
PreActResNet18	95.11%
DPN92	95.16%
DLA	95.47%
知识蒸馏脚本使用说明
本项目支持 DLA → MobileNetV2 的知识蒸馏训练，脚本文件为 distill_dla_mobilenetv2.py。

基础命令
bash
python distill_dla_mobilenetv2.py
可选参数说明
--lr 学习率（默认 0.05）

--epochs 训练轮数（默认 200）

--alpha 硬损失权重（默认 0.7）

--temp 蒸馏温度（默认 5.0）

--batch_size 批大小（默认 128）

--resume 从最新检查点恢复训练

使用示例：

bash
python distill_dla_mobilenetv2.py --lr 0.01 --epochs 100 --alpha 0.5 --temp 4.0 --batch_size 64
训练流程
自动加载 ./checkpoint/dla.pth 作为教师模型权重

训练过程中最佳学生模型将保存至 ./checkpoint/mobilenetv2_distilled.pth

每轮训练的最新检查点将保存至 ./checkpoint/mobilenetv2_latest.pth，可使用 --resume 参数恢复训练

查看结果
训练结束后，终端将输出最佳准确率。所有模型权重可在 'checkpoint' 文件夹中找到。
