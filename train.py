import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from readMnist import *
from myData import Mnist
from LeNet5 import LeNet5

train_images = load_train_images()
train_labels = load_train_labels()

trainData = Mnist(train_images, train_labels)
train_data = DataLoader(dataset=trainData, batch_size=1, shuffle=True)

lenet5 = LeNet5()
lenet5.cuda()

lossFun = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=lenet5.parameters(), lr=1e-4)

Epochs = 100
L = len(train_data)

for epoch in range(Epochs):
    for i, (img, id) in enumerate(train_data):

        img = img.float()
        id = id.float()

        img = img.cuda()
        id = id.cuda()

        img = Variable(img, requires_grad=True)
        id = Variable(id, requires_grad=True)

        Output = lenet5.forward(img)
        loss = lossFun(Output, id.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter = epoch * L + i + 1
        if iter % 100 == 0:
            print('epoch：{}，iter:{},loss:{:.6f}'.format(epoch + 1, iter, loss))

    torch.save(lenet5.state_dict(), 'lenet5.pth')
