import torch
import torch.nn as nn
from torchvision import models

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
        self.layer1 = nn.Linear(3072, 256)
        self.layer2 = nn.Linear(256, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.log_softmax(x)       
        
        return out


class ResNet34CustomInput(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(ResNet34CustomInput, self).__init__()

        # Load the pre-trained ResNet-34 model without the final classification layer
        self.resnet = models.resnet34(pretrained=True)
        # Remove the original fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Modify the first convolution layer to accept your specific input channel size
        if input_channels != 3:
            self.resnet[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Adaptive average pooling layer to adapt to different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Dropout layer
        self.dropout = nn.Dropout(0)

        # New fully connected layer for your specific number of classes
        self.fc = nn.Linear(512, num_classes)  # 512 is the output features of ResNet34

        #Xavier initialisation
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x