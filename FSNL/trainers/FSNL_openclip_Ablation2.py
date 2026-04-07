import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from typing import Union, List
import re
import copy
import random

import open_clip
from open_clip.tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# _tokenizer = _Tokenizer()


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
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone_name, precision=cfg.TRAINER.FSNL.PREC, pretrained='laion400m_e32')
    model.eval()

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        # Modified for OpenCLIP compatibility
        self.attn_mask = clip_model.attn_mask
        self.dtype = clip_model.visual.conv1.weight.dtype
        print("clip_model.visual.conv1.weight.dtype: ", self.dtype)

    def forward(self, prompts, tokenized_prompts):  # second arg: token embeddings, third arg: BPE token ids

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Modified for OpenCLIP compatibility
        x = self.transformer(x, attn_mask=self.attn_mask.to(x.device))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection

        return x

CUSTOM_TRAINING_HINT_PROMPT = {
    "Pokemon89_39": "pokemon",
    "Animals92_42": "animal",
    "Architecture50_23": "architecture",
    "Attire54_28": "attire",
    "FolkArt59_27": "folk art",
    "Food53_27": "food",
    "Insects_Spiders106_52": "insect or spider",
    "Landmark59_30": "landmark",
    "Plants_Fugi113_56": "plant or fungus",
}


class NameLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        
        HINT = CUSTOM_TRAINING_HINT_PROMPT[cfg.DATASET.NAME]
        self.set_type = HINT
        print("Hint: ", self.set_type)

        # # Convert all classnames to lowercase
        # lower_classnames = [name.replace("_", " ").lower()
        #                     for name in classnames]
        # self.classnames = lower_classnames
        self.classnames = classnames

        # Whether to use fixed-length name embeddings (True=fixed, False=dynamic)
        fixed_length_embedding = cfg.TRAINER.FSNL.FLE

        # Number of tokens when using fixed-length embedding
        n_ctx = cfg.TRAINER.FSNL.N_CTX

        # When dynamic: controls initialization. True=init from CLIP token embeddings, False=random init
        cne_init_from_clip = cfg.TRAINER.FSNL.CIFC

        # Get model's dtype and embedding dimension
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]

        # Create a dictionary to store embeddings for each classname
        with torch.no_grad():
            self.name_embeddings = nn.ParameterDict().type(self.dtype)
            for name in self.classnames:
                if fixed_length_embedding:
                    embed_length = n_ctx
                    embedding_tensor = torch.empty(
                        embed_length, self.ctx_dim, dtype=self.dtype)
                    nn.init.normal_(embedding_tensor, std=0.02)
                else:
                    bpe_tokens = _tokenizer.encode(name)
                    embed_length = len(bpe_tokens)
                    if cne_init_from_clip:
                        embedding_tensor = clip_model.token_embedding(
                            torch.tensor(bpe_tokens)).type(self.dtype)
                    else:
                        embedding_tensor = torch.empty(
                            embed_length, self.ctx_dim, dtype=self.dtype)
                        nn.init.normal_(embedding_tensor, std=0.02)
                self.name_embeddings[name] = nn.Parameter(embedding_tensor)

        print('fixed_length_embedding status: ' + str(fixed_length_embedding))
        if fixed_length_embedding:
            print(
                'Static embedding lenght, Number of tokens for each classname: ' + str(n_ctx))
        else:
            print('Dynamic embedding length, embedding tensor init_from_clip status: ' +
                  str(cne_init_from_clip))

        self.clip_model_token_embedding = clip_model.token_embedding
        self.fixed_length_embedding = fixed_length_embedding
        self.cne_init_from_clip = cne_init_from_clip
        self.n_ctx = n_ctx
        self.all_tokenized_prompts = None
        self.all_tokenized_prompts_inorder = None
        
    def forward(self, caption=[]):
        all_embeddings = []
        all_tokenized_prompts = []
        # count = 0
        # Build prompt string for each classname
        for name in self.classnames:
            prompt = f"a photo of {name}, a type of {self.set_type}."
            # prompt = f"a photo of {name} {self.set_type}."
            # prompt = f"a photo of {name}."
            random.shuffle(caption)
            if len(caption) >0:
                for cap in caption:
                    if name in cap.lower():
                        # cap_clean = cap.lower().replace(name, "")
                        prompt = cap
                        # count += 1
                        # print("Change caption: ", count, ": ",name, ": ",prompt)
                        # print(name, " changed into: ", prompt)
            embedding, tokenized_prompt = self.text2embedding(prompt)
            all_embeddings.append(embedding)
            all_tokenized_prompts.append(tokenized_prompt)

        # Stack all embeddings and tokenized_prompts
        # (n_cls, token_length, ctx_dim)
        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_tokenized_prompts = torch.stack(
            all_tokenized_prompts, dim=0)  # (n_cls, token_length)

        self.all_tokenized_prompts = all_tokenized_prompts

        return all_embeddings

    def forward_inorder(self, caption=[]):
        all_embeddings = []
        all_tokenized_prompts = []
        for cap in caption:
            prompt = cap
            embedding, tokenized_prompt = self.text2embedding(prompt)

            all_embeddings.append(embedding)
            all_tokenized_prompts.append(tokenized_prompt)

        # Stack all embeddings and tokenized_prompts
        # (n_cls, token_length, ctx_dim)
        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_tokenized_prompts = torch.stack(
            all_tokenized_prompts, dim=0)  # (n_cls, token_length)

        self.all_tokenized_prompts_inorder = all_tokenized_prompts

        return all_embeddings
    
    def text2embedding(self, text):

        # Sync to the same device as token embeddings
        device = self.clip_model_token_embedding.weight.device
        self.name_embeddings = self.name_embeddings.to(device)

        # 1. Match classnames in the text
        # Convert text to lowercase
        text = text.lower()
        matched_names = [name for name in self.classnames if name in text]

        # 2. Split the text based on matched classnames while preserving order and classnames using re.split
        # Use capturing groups
        pattern = "|".join(
            "(" + re.escape(name) + ")" for name in matched_names)
        segments = re.split(pattern, text)

        # Removing empty strings if any
        segments = [segment for segment in segments if segment]

        # 3. Convert each segment to BPE tokens and count the number of tokens for each segment
        bpe_tokens_segments = [_tokenizer.encode(
            segment) for segment in segments]
        token_lengths = [len(tokens) for tokens in bpe_tokens_segments]

        # 4. Truncate the bpe tokens, fake_token_lengths is used for foreseeing the length of embedding after replacing.
        max_bpe_length = 75  # 77 - 2 (for SOT and EOT tokens)

        if self.fixed_length_embedding:
            fake_token_lengths = []
            for segment, token_length in zip(segments, token_lengths):
                if segment in matched_names:
                    fake_token_lengths.append(self.n_ctx)
                else:
                    fake_token_lengths.append(token_length)

        else:
            fake_token_lengths = copy.deepcopy(token_lengths)

        # Compute the current total length of BPE tokens
        current_length = sum(fake_token_lengths)

        # Truncate bpe tokens if necessary
        while current_length > max_bpe_length:
            # Remove the last segment
            last_segment = segments.pop()  # list of text segments
            last_bpe_tokens_segment = bpe_tokens_segments.pop()  # list of BPE token sequences per segment
            last_segment_length = fake_token_lengths.pop()  # list of token lengths per segment
            _ = token_lengths.pop()

            # If the removed segment is a classname, simply reduce the current_length
            if last_segment in matched_names:
                current_length -= last_segment_length
            else:
                # Calculate the available space for this segment after truncation
                available_space = max_bpe_length - \
                    (current_length - last_segment_length)

                # If there's still enough space for part of the segment
                if available_space > 0:
                    truncated_bpe_tokens = last_bpe_tokens_segment[
                        :available_space]

                    # Update segments, bpe_tokens_segments and fake_token_lengths
                    segments.append(last_segment)
                    bpe_tokens_segments.append(truncated_bpe_tokens)
                    fake_token_lengths.append(len(truncated_bpe_tokens))
                    token_lengths.append(len(truncated_bpe_tokens))

                    current_length = sum(fake_token_lengths)
                else:
                    current_length -= last_segment_length
                # Update the current total length of bpe tokens

        # 5. Concatenate BPE tokens with SOT, EOT and compute embeddings

        sot_token = _tokenizer.encoder["<start_of_text>"]
        eot_token = _tokenizer.encoder["<end_of_text>"]
        all_tokens = [sot_token] + sum(bpe_tokens_segments, []) + [eot_token]

        with torch.no_grad():
            embeddings = self.clip_model_token_embedding(
                torch.tensor(all_tokens).to(device)).type(self.dtype)

            if self.fixed_length_embedding:
                pseudo_bpe_tokens_segments = []
                pseudo_tokens = _tokenizer.encode(" ".join(["X"] * self.n_ctx))
                for segment, bpe_tokens_segment in zip(segments, bpe_tokens_segments):
                    if segment in matched_names:
                        pseudo_bpe_tokens_segments.append(pseudo_tokens)
                    else:
                        pseudo_bpe_tokens_segments.append(bpe_tokens_segment)
                pseudo_all_tokens = [sot_token] + \
                    sum(pseudo_bpe_tokens_segments, []) + [eot_token]

                tokenized_prompts = torch.tensor(pseudo_all_tokens).to(device)
            else:
                tokenized_prompts = torch.tensor(all_tokens).to(device)

        # 6. Replace embeddings for matched classnames
        start_idx = 1  # Starting after SOT token
        for i, (segment, length) in enumerate(zip(segments, token_lengths)):
            if segment in matched_names:
                # Split the embeddings into three parts: before, during, and after the matched segment
                before = embeddings[:start_idx]
                middle = self.name_embeddings[segment]
                after = embeddings[start_idx+length:]

                # Concatenate the three parts
                embeddings = torch.cat([before, middle, after], dim=0)

                # Adjust the start index
                start_idx += len(middle)
            else:
                start_idx += length

        # 7. Padding for embedding and tokenized_prompts
        embedding_length = embeddings.shape[0]
        tokenized_prompts_length = tokenized_prompts.shape[0]
        assert embedding_length == tokenized_prompts_length, f"embedding_length ({embedding_length}) must equal to tokenized_prompts_length ({tokenized_prompts_length})"

        if embedding_length < 77:
            with torch.no_grad():
                padding_length = 77 - embedding_length
                padding_tokens = torch.zeros(
                    padding_length, dtype=torch.long).to(device)
                padding_embeddings = self.clip_model_token_embedding(
                    padding_tokens).type(self.dtype)

                tokenized_prompts = torch.cat(
                    [tokenized_prompts, padding_tokens], dim=0)
            embeddings = torch.cat([embeddings, padding_embeddings], dim=0)

        # (token_length=77, ctx_dim), (token_length=77)
        return embeddings, tokenized_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # Convert all classnames to lowercase
        self.classnames = [name.replace("_", " ").lower()
                            for name in classnames]
        self.classnames = [name.replace(".", " ").lower()
                            for name in classnames]
        self.name_learner = NameLearner(cfg, self.classnames, clip_model)
        # _ = self.name_learner()  # used to update self.all_tokenized_prompts
        # self.tokenized_prompts = self.name_learner.all_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        # Modified for OpenCLIP compatibility
        self.dtype = clip_model.visual.conv1.weight.dtype

    def forward(self, image, caption=[]):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.name_learner(caption)
        tokenized_prompts = self.name_learner.all_tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def extract_classname(self, description):
        description = description.lower()
        
        for classname in self.classnames:
            if classname in description:
                template = description.replace(classname, '{}')
                return template, classname
        return None, None

    def swap_context(self, captions):
        templates = []
        caption_classnames = []
        
        # Extract template and classname from each caption
        for description in captions:
            template, classname = self.extract_classname(description)
            templates.append(template)
            caption_classnames.append(classname)

        reorganized_captions = []
        for idx, classname in enumerate(caption_classnames):
            # Pick a template from a different class
            swapped_template = random.choice([temp for i, temp in enumerate(templates) if caption_classnames[i] != classname])
            # Use replace to handle templates with multiple {} placeholders
            reorganized_captions.append(swapped_template.replace("{}", classname))
            # reorganized_captions.append(swapped_template.format(classname))

        return reorganized_captions


    def forward_modified_clip_loss(self, image, caption, label):
        caption = self.swap_context(caption)

        # image_features = self.image_encoder(image.type(self.dtype))
        image_features = image
        prompts = self.name_learner.forward_inorder(caption)
        tokenized_prompts = self.name_learner.all_tokenized_prompts_inorder
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
    
        image_loss = self.modified_contrastive_loss(logits_per_image, label)
        caption_loss = self.modified_contrastive_loss(logits_per_text, label)

        return (image_loss + caption_loss) / 2.0

    def modified_contrastive_loss(self, logits, label):
        # Build a binary matrix: entry (i,j)=True if samples i and j share the same label
        same_label_matrix = label.unsqueeze(0) == label.unsqueeze(1)

        # Apply softmax normalization to logits
        normalized_logits = F.softmax(logits, dim=1)
        
        # Scores for positive pairs
        positive_probs = normalized_logits * same_label_matrix
        positive_probs_sum = positive_probs.sum(dim=1)
        
        # Compute loss via negative log-likelihood
        loss = -torch.log(positive_probs_sum + 1e-10).mean()

        return loss


    def forward_Base_Double(self, image, label):

        # image_features = self.image_encoder(image.type(self.dtype))
        image_features = image
        
        # Compute text features from name embeddings
        prompts = self.name_learner()
        tokenized_prompts = self.name_learner.all_tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # Create label indices for the text side
        num_classname = text_features.size(0)
        label_for_text = torch.arange(num_classname, device=label.device)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        image_loss = self.modified_contrastive_loss2(logits_per_image, label, label_for_text)
        caption_loss = self.modified_contrastive_loss2(logits_per_text, label_for_text, label)

        return (image_loss + caption_loss) / 2.0
        # return image_loss

    def modified_contrastive_loss2(self, logits, label1, label2):
        # Build a binary matrix indicating which samples share the same label
        # num_rows, num_cols = logits.shape
        
        same_label_matrix = label2.unsqueeze(0) == label1.unsqueeze(1)
        # same_label_matrix = same_label_matrix.float()
        
        # # Ensure same_label_matrix dimensions match logits
        # assert same_label_matrix.shape[0] == num_rows and same_label_matrix.shape[1] == num_cols

        # Normalize logits
        normalized_logits = F.softmax(logits, dim=-1)

        # Compute positive class probability for each sample
        positive_probs = normalized_logits * same_label_matrix
        positive_probs_sum = positive_probs.sum(dim=1)

        # Compute loss
        loss = -torch.log(positive_probs_sum + 1e-10).mean()
        return loss



@TRAINER_REGISTRY.register()
class FSNL_openclip_Ablation2(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FSNL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
        #     # CLIP's default precision is fp16
        #     clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "name_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(
                self.model.name_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give name_learner to the optimizer
        self.optim = build_optimizer(self.model.name_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "name_learner", self.model.name_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FSNL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(
                f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        # print(sum(p.numel() for p in self.model.name_learner.parameters() if p.requires_grad))
        # raise ValueError("Testing")
        
        # self.use_random_caption = not cfg.TRAINER.LTNa.MIX
        print("PREC: ", self.cfg.TRAINER.FSNL.PREC)
        
        self.FSNL_USE_CAPTION = cfg.TRAINER.FSNL.USE_CAPTION
        print("FSNL_USE_CAPTION: ", self.FSNL_USE_CAPTION)
        
        # CSV logging template for experiment metadata
        default_log_template = dict()
        # Model info
        default_log_template["Method"] = str(self.cfg.TRAINER.NAME)
        default_log_template["PREC"] = self.cfg.TRAINER.FSNL.PREC
        default_log_template["MAX_EPOCH"] = self.cfg.OPTIM.MAX_EPOCH
        # Experiment settings
        default_log_template["SHOTS"] = self.cfg.DATASET.NUM_SHOTS
        default_log_template["LR"] = self.cfg.OPTIM.LR
        default_log_template["BATCH_SIZE"] = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        default_log_template["Datasets"] = self.cfg.DATASET.NAME
        default_log_template["SUB_CLASSES"] = self.cfg.DATASET.SUB_CLASSES
        self.note = ""
        self.default_log_template = default_log_template
        
    def forward_backward(self, batch):
        image, label, caption = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FSNL.PREC
        if prec == "amp":
            with autocast():
                if self.FSNL_USE_CAPTION:
                    # Base Double + Caption
                    image_features = self.model.image_encoder(image.type(self.model.dtype))
                    # Ablation
                    # loss1 = self.model.forward_modified_clip_loss(image_features, caption, label)
                    loss2 = self.model.forward_Base_Double(image_features, label)
                    # loss = (loss1 + loss2)/ 2.0   
                    loss = loss2
                else:
                    # Base Double
                    image_features = self.model.image_encoder(image.type(self.model.dtype))
                    loss2 = self.model.forward_Base_Double(image_features, label)
                    loss =loss2


            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if self.FSNL_USE_CAPTION:
                # Base Double + Caption
                image_features = self.model.image_encoder(image.type(self.model.dtype))
                # Ablation
                # loss1 = self.model.forward_modified_clip_loss(image_features, caption, label)
                loss2 = self.model.forward_Base_Double(image_features, label)
                # loss = (loss1 + loss2)/ 2.0   
                loss = loss2
            else:
                # Base Double
                image_features = self.model.image_encoder(image.type(self.model.dtype))
                loss2 = self.model.forward_Base_Double(image_features, label)
                loss = loss2

            self.model_backward_and_update(loss)
            

        loss_summary = {
            "loss": loss.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        caption = batch["caption"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, caption


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # # Ignore fixed token vectors
            # if "token_prefix" in state_dict:
            #     del state_dict["token_prefix"]

            # if "token_suffix" in state_dict:
            #     del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(
                name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
