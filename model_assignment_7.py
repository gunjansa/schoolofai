class Net1(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net1, self).__init__()
        # RF = RF + (jin * (kernel-1))
        # jin = jin * stride
        # For MaxPool, kernel = 2 and stride = 2
        dropout_value = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26, RF = 3, jin=1

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24, RF = 5, jin=1

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 6, jin = 2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 10, RF = 10, jin=2

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 5, RF = 12, jin = 2

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 3, RF = 20, jin=4

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 1, RF = 28, jin=4
        
        # CONV and GAP
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 1, RF = 28

        # OUTPUT BLOCK
        output_size_after_averaging = 1
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size_after_averaging)
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        # x = self.convblock3(x)
        # x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        # x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)

        x = self.convblock9(x)
        x = self.gap(x)

        x = x.view(len(x), 10)
        
        return F.log_softmax(x, dim=1)
