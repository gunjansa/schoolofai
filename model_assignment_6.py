# File to create Model Definition
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 10, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)

class Net2(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        # x = F.relu(F.max_pool2d(x, 2))
        # x = F.relu(F.max_pool2d(x, 2))
        x = x.view(len(x), 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Net3(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.batchnorm7 = nn.BatchNorm2d(32)
        # self.conv8 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.batchnorm8 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.1)
        
        # CONV and GAP
        self.conv9 = nn.Conv2d(32, 10, kernel_size=1)

        # GAP and FC1
        self.average_pooling =  nn.AvgPool2d(3) 
        self.fc1 = nn.Linear(32, 10)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.dropout(self.pool1(self.batchnorm2(F.relu(self.conv2(self.batchnorm1(F.relu(self.conv1(x))))))))
        x = self.dropout(self.pool2(self.batchnorm4(F.relu(self.conv4(self.batchnorm3(F.relu(self.conv3(x))))))))
        x = self.dropout(self.pool3(self.batchnorm6(F.relu(self.conv6(self.batchnorm5(F.relu(self.conv5(x))))))))
        x = self.dropout(self.batchnorm7(F.relu(self.conv7(x))))
        
        x = F.relu(self.conv9(x))

        x = F.relu(self.average_pooling(x))
        x = x.view(len(x), 10)
        # x = self.fc1(x)
        
        return F.log_softmax(x, dim=1)
