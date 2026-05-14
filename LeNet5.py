
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=120),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid()
        )

        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, img):
        img = self.conv1.forward(img)
        img = self.conv2.forward(img)

        img = img.view(img.size()[0], -1)

        img = self.fc1.forward(img)
        img = self.fc2.forward(img)
        img = self.fc3.forward(img)

        return img
