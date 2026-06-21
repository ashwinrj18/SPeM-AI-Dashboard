import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetPI(nn.Module):
    def __init__(self):
        super(ResNetPI, self).__init__()
        # Initialize an untrained ResNet-18
        resnet = resnet18(weights=None)
        
        # Modify the first layer for 1-channel grayscale pressure inputs
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Keep the first three residual blocks (drop layer4 for a lighter model)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        # Regularization and pooling
        self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.dropout(x)
        x = self.avgpool(x)
        
        # Flatten into a 1D feature vector per image
        x = torch.flatten(x, 1)
        return x

class SPeM_CRNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SPeM_CRNN, self).__init__()
        
        # 1. Spatial block
        self.cnn = ResNetPI()
        
        # 2. Temporal block (LSTM expects the 256 output channels from ResNetPI's layer3)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        
        # 3. Classifier block
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [Batch, Frames, Channels, Height, Width]
        batch_size, frames, c, h, w = x.size()
        
        # Collapse batch and frames to process through CNN at once
        x_reshaped = x.view(batch_size * frames, c, h, w)
        cnn_features = self.cnn(x_reshaped)
        
        # Reshape back to sequence format for the LSTM
        sequence_features = cnn_features.view(batch_size, frames, -1)
        
        # Process sequence and grab the final hidden state
        lstm_out, (hn, cn) = self.lstm(sequence_features)
        final_hidden_state = hn[-1] 
        
        # Final classification
        output = self.fc(final_hidden_state)
        return output