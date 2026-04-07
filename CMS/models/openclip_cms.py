import os
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
import open_clip
from models import vision_transformer as vits

class OpenCLIPCMS(nn.Module):
    def __init__(self, args):
        super(OpenCLIPCMS, self).__init__()
        self.k = args.k
        self.backbone = self.init_backbone(args.openclip_model, args.openclip_pretrained)
        self.img_projection_head = vits.__dict__['DINOHead'](in_dim=512, out_dim=args.feat_dim, nlayers=args.num_mlp_layers)

    def forward(self, image):
        feat = self.backbone(image)
        feat = self.img_projection_head(feat)
        feat = F.normalize(feat, dim=-1)
            
        return feat

    def init_backbone(self, model_name, pretrained):
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        # 首先冻结所有参数
        for param in model.visual.parameters():
            param.requires_grad = False

        # 然后只解冻最后一个transformer块和最后的层归一化
        for name, param in model.visual.named_parameters():
            if 'resblocks.11' in name or 'ln_post' in name:
                param.requires_grad = True

        return model.visual