import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

import open_clip
from open_clip.tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()



CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "Pokemon89_39": "a photo of {}, a type of pokemon.",
    "Animals92_42": "a photo of {}, a type of animal.",
    "Architecture50_23": "a photo of {}, a type of architecture.",
    "Attire54_28": "a photo of {}, a type of attire.",
    "FolkArt59_27": "a photo of {}, a type of folk art.",
    "Food53_27": "a photo of {}, a type of food.",
    "Insects_Spiders106_52": "a photo of {}, a type of insect or spider.",
    "Landmark59_30": "a photo of {}, a type of landmark.",
    "Plants_Fugi113_56": "a photo of {}, a type of plant or fungus."
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)

    # try:
    #     # loading JIT archive
    #     model = torch.jit.load(model_path, map_location="cpu").eval()
    #     state_dict = None

    # except RuntimeError:
    #     state_dict = torch.load(model_path, map_location="cpu")

    # model = clip.build_model(state_dict or model.state_dict())
    model, _, preprocess = open_clip.create_model_and_transforms(backbone_name, precision=cfg.TRAINER.COOP.PREC, pretrained='laion400m_e32')
    model.eval()

    return model


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        # Modified for OpenCLIP compatibility
        self.attn_mask = clip_model.attn_mask
        self.dtype = clip_model.visual.conv1.weight.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([open_clip.tokenizer.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.adapter = Adapter(512, 4).to(self.dtype)

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)
        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features
            
        text_features = self.text_encoder()
        # x = self.adapter(text_features)
        # ratio = 0.2
        # text_features = ratio * x + (1 - ratio) * text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIP_Adapter(TrainerX):
    """ CLIP-Adapter """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        
        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        

        self.register_model('adapter', self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)
        
        # CSV logging template for experiment metadata
        default_log_template = dict()
        default_log_template["Method"] = str(self.cfg.TRAINER.NAME)
        default_log_template["PREC"] = self.cfg.TRAINER.COOP.PREC
        default_log_template["MAX_EPOCH"] = self.cfg.OPTIM.MAX_EPOCH
        # Experiment settings
        default_log_template["SHOTS"] = self.cfg.DATASET.NUM_SHOTS
        default_log_template["LR"] = self.cfg.OPTIM.LR
        default_log_template["BATCH_SIZE"] = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        default_log_template["Datasets"] = self.cfg.DATASET.NAME
        default_log_template["SUB_CLASSES"] = self.cfg.DATASET.SUB_CLASSES
        self.note = "EXP"
        self.default_log_template = default_log_template
        
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            try:
                model_path = osp.join(directory, name, model_file)
                if not osp.exists(model_path):
                    raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            except:
                epoch = self.cfg.OPTIM.MAX_EPOCH
                model_file = "model.pth.tar-" + str(epoch)
                model_path = osp.join(directory, name, model_file)
                if not osp.exists(model_path):
                    raise FileNotFoundError('Model not found at "{}"'.format(model_path))
                else:
                    print("Use Model found at: ", model_path)

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            
            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
