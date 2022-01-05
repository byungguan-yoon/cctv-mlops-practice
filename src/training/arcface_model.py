from conf import *

import torch 
import torch.nn as nn
import torchvision.models as models
import timm
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

    
class Backbone(nn.Module):

    
    def __init__(self, name='resnet18', pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)
        
        if 'regnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'csp' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'res' in name: #works also for resnest
            self.out_features = self.net.fc.in_features
        elif 'efficientnet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'densenet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'senet' in name:
            self.out_features = self.net.fc.in_features
        elif 'inception' in name:
            self.out_features = self.net.last_linear.in_features
        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x

    
class Net(nn.Module):
    def __init__(self, args, pretrained=True):
        super(Net, self).__init__()
        
        self.backbone = Backbone('tf_efficientnet_b0_ns', pretrained=pretrained)
        
        if args.pool == "gem":
            self.global_pool = GeM(p_trainable=True)
        elif args.pool == "identity":
            self.global_pool = torch.nn.Identity()
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.embedding_size = args.embedding_size
        
        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if args.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif args.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )
            
        self.head = ArcMarginProduct(self.embedding_size, args.n_classes)
        
        if args.pretrained_weights is not None:
            self.load_state_dict(torch.load(args.pretrained_weights, map_location='cpu'), strict=False)
            print('weights loaded from',args.pretrained_weights)

    def forward(self, img, get_embeddings=False, get_attentions=False):

        x = img
        x = self.backbone(x)
        
        x = self.global_pool(x)
        x = x[:,:,0,0]
        
        x = self.neck(x)

        logits = self.head(x)

        return logits

# sigmoid = torch.nn.Sigmoid()
# class Swish(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * sigmoid(i)
#         ctx.save_for_backward(i)
#         return result
#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_variables[0]
#         sigmoid_i = sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
# swish = Swish.apply

# class Swish_module(nn.Module):
#     def forward(self, x):
#         return swish(x)
# swish_layer = Swish_module()

class timm_models(nn.Module):
    def __init__(self, args):
        super().__init__()
        model = timm.create_model(args.encoder_name, pretrained=args.pretrained, num_classes=args.n_classes)
        self.model = model

    def forward(self, img):
        output = self.model(img)
        return output

class Resnet50(nn.Module):
    def __init__(self, num_classes, dropout=False, pretrained=False):
        super().__init__()
        model = models.resnet50(pretrained=pretrained)
        model = list(model.children())[:-1]
        if dropout:
            model.append(nn.Dropout(0.2))
        model.append(nn.Conv2d(2048, num_classes, 1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)

class CNN_Base(nn.Module):
    def __init__(self, ):
        super(CNN_Base, self).__init__()  

        self.cnn_layer = nn.Sequential(            
            nn.Conv2d(3,6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(6,12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(12,15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential( 
            nn.Linear(735, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), 
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def build_model(args, device):
        
    if args.arch == 'base': 
        model = CNN_Base().to(device)
    elif args.arch == 'arcface':
        model = Net(args)
    else:
        model = timm_models(args)

    if device: 
        model = model.to(device)

    return model

if __name__ == "__main__":
    model = build_model(args, 'cpu')
    sample_tensor = torch.rand(3, 3, 224, 224)
    output = model(sample_tensor)
    print(output.shape)