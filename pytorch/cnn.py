import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.softmax(x, dim=1) ## 这里不要用softmax,因为损失计算的时候使用的CrossEntropyLosCrossEntropyLoCrossEntropyLoss内部已经包含softmax
        return x

    def predict_probs(self, x):
        probs = F.softmax(self.forward(x))
        return probs

    def predict(self, x):
        probs = self.predict_probs(x)
        return torch.argmax(probs, dim=1)

    def get_loss(self, y_pred, y):
        loss = nn.CrossEntropyLoss()(y_pred, y)
        return loss

    def start_train(self, train_loader, epoch=100):
        loss_history = []
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epoch):
            cnt = 0
            loss_sum = 0.0
            for i, (images, labels) in enumerate(train_loader):
                loss_sum += self.train_batch(images, labels, optimizer)
                cnt += 1
            loss_history.append(loss_sum / cnt)
            print(f"epoch={epoch}, loss={loss_sum / cnt}")

    def train_batch(self, images, labels, optimizer):
        self.train()
        optimizer.zero_grad()
        outputs = self.forward(images)
        loss = self.get_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(train_dataset.data.shape, test_dataset.data.shape)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnn = SimpleCNN()
    cnn.start_train(train_loader, epoch=10)