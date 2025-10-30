非常好 👍 你问的这个是 PyTorch 训练中最常用的可视化工具之一。
我们一步步讲清楚——如何用 **TensorBoard**（官方支持）来可视化训练过程。

---

## 🧩 一、安装

```bash
pip install tensorboard
```

如果你想兼容旧项目（比如以前用 `tensorboardX`），可以额外装上：

```bash
pip install tensorboardX
```

---

## 🧠 二、在 PyTorch 中使用（推荐官方接口）

官方接口在 `torch.utils.tensorboard`，API 与 `tensorboardX` 完全一致。

### ✅ 基本示例

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

# 1️⃣ 创建日志目录（所有记录都会写到这里）
writer = SummaryWriter(log_dir='./runs/experiment1')

# 2️⃣ 添加标量（最常用：loss, acc）
for step in range(100):
    writer.add_scalar('Loss/train', np.random.random(), step)
    writer.add_scalar('Accuracy/train', np.random.random(), step)

# 3️⃣ 添加图像
images = torch.rand(4, 3, 28, 28)
writer.add_images('InputImages', images)

# 4️⃣ 添加权重直方图
weights = torch.randn(100)
writer.add_histogram('Layer1/weights', weights, 0)

# 5️⃣ 添加模型计算图
import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(1, 16, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(26*26*16, 10)
)
x = torch.randn(1, 1, 28, 28)
writer.add_graph(model, x)

writer.close()
```

---

## 🚀 三、启动 TensorBoard

运行命令：

```bash
tensorboard --logdir=./runs
```

打开浏览器访问：

> [http://localhost:6006](http://localhost:6006)

---

## 📊 四、TensorBoard 常见可视化内容

| 类型   | API                                                                     | 用途                  |
| ---- | ----------------------------------------------------------------------- | ------------------- |
| 标量   | `add_scalar(tag, value, step)`                                          | loss、accuracy、lr    |
| 多个标量 | `add_scalars(main_tag, {'train':v1, 'val':v2}, step)`                   | 对比 train vs val     |
| 图像   | `add_image(tag, img_tensor, step)` / `add_images(tag, img_batch, step)` | 输入样本或结果             |
| 直方图  | `add_histogram(tag, values, step)`                                      | 参数分布变化              |
| 计算图  | `add_graph(model, input_tensor)`                                        | 可视化模型结构             |
| 文本   | `add_text(tag, text_string, step)`                                      | 训练日志或备注             |
| 嵌入   | `add_embedding(features, metadata, label_img)`                          | t-SNE/embedding 可视化 |

---

## 🧩 五、集成到训练循环中（完整示例）

下面是一个典型的 MNIST 训练 + TensorBoard 记录流程：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1️⃣ 数据
transform = transforms.ToTensor()
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# 2️⃣ 模型
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*14*14, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3️⃣ TensorBoard
writer = SummaryWriter('./runs/mnist_demo')

# 4️⃣ 训练循环
for epoch in range(5):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(trainloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# 5️⃣ 保存模型结构
sample_input = torch.randn(1, 1, 28, 28)
writer.add_graph(model, sample_input)

writer.close()
```

运行后再执行：

```bash
tensorboard --logdir=./runs
```

---

## 💡 六、实用技巧

* 🧮 可以用不同的 `log_dir` 区分实验（如 `runs/exp1_lr_0.01`, `runs/exp2_lr_0.001`）
* 🔍 支持多个实验曲线同时对比
* 🧠 支持记录学习率变化曲线（适配 Scheduler）

---

是否希望我帮你把 **TensorBoard 可视化** 整合进 “深度可分离膨胀卷积 MNIST” 的代码版本？那样你可以实时看到 loss、accuracy、feature map 的图像。
