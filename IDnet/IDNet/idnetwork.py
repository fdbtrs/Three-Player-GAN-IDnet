from torch import nn
from torchvision import transforms

from IDNet.backbones.iresnet import iresnet50
from IDNet.loss import CosFace


class IDNet(nn.Module):
    def __init__(self, embedding_size, c_dim):
        super(IDNet, self).__init__()
        self.backbone = iresnet50(dropout=0.4, num_features=embedding_size)
        self.cosface = CosFace(embedding_size, c_dim)
        self.resize = nn.Sequential(transforms.Resize(112))

    def forward(self, images, labels):
        resized_images = self.resize(images)
        embeddings = self.backbone(resized_images)
        ret = self.cosface(embeddings, labels)
        return ret
