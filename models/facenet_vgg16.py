from torchvision import models
from torch import nn
import torch.nn.functional as F


class FaceNetVGG16(nn.Module):
    def __init__(self, embed_layer_size, dropout):
        super().__init__()
        self.feature_extractor = models.vgg16(pretrained=True).features
        self.feature_extractor.requires_grad_(False)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, embed_layer_size)

    def one_iter(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def forward(self, anchor, positive, negative):
        anchor_embed = self.one_iter(anchor)
        pos_embed = self.one_iter(positive)
        neg_embed = self.one_iter(negative)
        return anchor_embed, pos_embed, neg_embed
