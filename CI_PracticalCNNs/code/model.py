# model.py
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_blocks, initial_filters=32, nhl_activation='relu', ol_activation=None):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_channels = 1
        fs = [initial_filters * (2 ** i) for i in range(num_blocks)]
        
        for filters in fs:
            self.layers.append(nn.Conv2d(prev_channels, filters, kernel_size=3, padding=1))
            if nhl_activation == 'relu':
                self.layers.append(nn.ReLU())
            elif nhl_activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            self.layers.append(nn.MaxPool2d(2, 2))
            prev_channels = filters
        
        final_dim = 28 // (2 ** num_blocks)
        if final_dim <= 0:
            raise ValueError("The number of blocks reduces the feature map to non-positive dimensions.")
        self.fc_input_size = fs[-1] * final_dim * final_dim
        self.fc = nn.Linear(self.fc_input_size, 101)
        
        if ol_activation == 'softmax':
            self.ol_activation = nn.Softmax(dim=1)
        elif ol_activation == 'log_softmax':
            self.ol_activation = nn.LogSoftmax(dim=1)
        else:
            self.ol_activation = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)
        if self.ol_activation:
            x = self.ol_activation(x)
        return x
