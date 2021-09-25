import torch
import torch.nn as nn
import torchvision.models as models


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class VGGStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19(pretrained=True).features.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self.model.children():
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        self.conv_num = 5

    def __call__(self, x):
        ys = []
        for layer in self.model.children():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                ys.append(gram_matrix(x))
                if len(ys) >= self.conv_num:
                    break
        return ys
