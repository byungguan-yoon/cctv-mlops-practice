import timm
import torch.nn.functional as F
from torch import nn

class FaceModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = timm.create_model('mobilenetv2_100', pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1]).eval()
        
    def normalize(self, feat):
        return F.normalize(feat, p=2, dim=1)

    def forward(self, x):
        output = self.model(x)
        output = self.normalize(output)
        return output