import torch.nn as nn
import torchvision.models as models
    
class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        base_model = models.resnet18(weights=None)

        # fix the first layer with input 1
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        base_model.fc = nn.Linear(512, 4 * 26)

        self.resnet = base_model

    def forward(self, x):
        x = self.resnet(x)
        return x.view(-1, 4, 26)  # (batch_size, 4, 26)