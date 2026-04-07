import os
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
import open_clip

class OpenCLIP_original(nn.Module):
    def __init__(self, args):
        super(OpenCLIP_original, self).__init__()
        self.backbone = self.init_backbone(args.openclip_model, args.openclip_pretrained)

    def forward(self, image):
        feat = self.backbone(image)
            
        return feat

    def init_backbone(self, model_name, pretrained):
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

        return model.visual