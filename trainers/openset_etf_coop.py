# CoOp w/ DePT
import os
import os.path as osp
import math
import torch
import torch.nn.functional as F
from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_lr_scheduler
from plotnine import *
from torch.cuda.amp import GradScaler, autocast
from torch import nn
import numpy as np

from .coop import CoOp, load_clip_to_cpu
from .coop import CustomCLIP as CustomCLIP_
from .coop import PromptLearner
from .elp_maple import FiLM
from .optim import build_optimizer


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    if feat_in >= num_classes:
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-6), \
            "The max irregular value is : {}".format(
                torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


class DRLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, reg_lambda=0.):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(self, feat, target, h_norm2=None, m_norm2=None, avg_factor=None):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)
        return loss * self.loss_weight


class OpenSetPromptLearner(PromptLearner):
    def __init__(self, cfg, classnames, clip_model, n_past_ood=0):
        super().__init__(cfg, classnames, clip_model)

        # build an initial clsname_vector from two class name embeddings
        n_cls = self.n_cls
        name_lens = self.name_lens
        suffix = self.token_suffix

        # if n_cls >= 2:
        #     idx_a, idx_b = 0, 1
        # else:
        #     idx_a = idx_b = 0

        if n_cls >= 2:
            base_seed = 1
            stage = int(getattr(getattr(cfg, "DATASET", object()), "CURRENT_STAGE", 0))
            g = torch.Generator()
            g.manual_seed(base_seed + stage)
            perm = torch.randperm(n_cls, generator=g)
            idx_a = int(perm[0].item())
            idx_b = int(perm[1].item())
        else:
            idx_a = idx_b = 0
 
        # use the same class as the template when building OOD prompts
        self.ood_template_idx = int(idx_a)
        # Persist template indices across stages.
        # The buffer is ordered as: [stage1_template, stage2_template, ..., current_stage_template].
        # It will be expanded during checkpoint loading when moving to the next stage.
        init_len = max(1, int(n_past_ood) + 1)
        self.register_buffer(
            "ood_template_idxs",
            torch.tensor([int(idx_a)] * init_len, dtype=torch.long),
        )
        
        name_len_a = name_lens[idx_a]
        name_len_b = name_lens[idx_b]

        class_a_tokens = suffix[idx_a : idx_a + 1, :name_len_a, :]
        class_b_tokens = suffix[idx_b : idx_b + 1, :name_len_b, :]

        class_a_tokens = class_a_tokens.to(dtype=self.ctx.dtype, device=self.ctx.device)
        class_b_tokens = class_b_tokens.to(dtype=self.ctx.dtype, device=self.ctx.device)

        # use the average number of tokens across all class names as L_ood
        # L_ood = max(1, int(round(sum(name_lens) / len(name_lens))))
        L_ood = 3
        print(L_ood)
        print(name_len_a)

        len_a = class_a_tokens.shape[1]
        if len_a >= L_ood:
            class_a_seq = class_a_tokens[:, :L_ood, :]
        else:
            pad_len_a = L_ood - len_a
            pad_a = class_a_tokens[:, -1:, :].expand(1, pad_len_a, -1)
            class_a_seq = torch.cat([class_a_tokens, pad_a], dim=1)

        len_b = class_b_tokens.shape[1]
        if len_b >= L_ood:
            class_b_seq = class_b_tokens[:, :L_ood, :]
        else:
            pad_len_b = L_ood - len_b
            pad_b = class_b_tokens[:, -1:, :].expand(1, pad_len_b, -1)
            class_b_seq = torch.cat([class_b_tokens, pad_b], dim=1)

        clsname_seq = 0.5 * (class_a_seq + class_b_seq)
        clsname_seq = clsname_seq.to(dtype=self.ctx.dtype, device=self.ctx.device)
        print(clsname_seq)
        # shape (1, 1, dim) so it can be used as a single token embedding
        self.clsname_seq = nn.Parameter(clsname_seq)

        # Past OOD sequences (Frozen)
        self.n_past_ood = n_past_ood
        if n_past_ood > 0:
            # Initialize with same shape/values as placeholder, will be overwritten by load_model
            self.past_clsname_seqs = nn.ParameterList([
                nn.Parameter(clsname_seq.clone()) for _ in range(n_past_ood)
            ])
            for p in self.past_clsname_seqs:
                p.requires_grad_(False)

        # flag that can be toggled externally to control OOD behavior
        self.is_ood = True

    def _build_ood_prompt(self, clsname_seq, template_idx=None):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # use the chosen template class for building a new prompt
        if template_idx is None:
            template_idx = int(getattr(self, "ood_template_idx", 0))
        else:
            template_idx = int(template_idx)
        if template_idx < 0 or template_idx >= self.n_cls:
            template_idx = 0
        prefix_k = prefix[template_idx : template_idx + 1, :, :]
        suffix_k = suffix[template_idx : template_idx + 1, :, :].clone()
        ctx_k = ctx[template_idx : template_idx + 1, :, :]

        # replace the entire class-name token segment in suffix_k
        name_lenk = self.name_lens[template_idx]
        ood_seq = clsname_seq.to(suffix_k.device, dtype=suffix_k.dtype)
        L_ood = ood_seq.shape[1]
        if L_ood >= name_lenk:
            ood_used = ood_seq[:, :name_lenk, :]
        else:
            pad_len = name_lenk - L_ood
            pad = ood_seq[:, -1:, :].expand(1, pad_len, -1)
            ood_used = torch.cat([ood_seq, pad], dim=1)
        suffix_k[:, :name_lenk, :] = ood_used

        new_prompt = torch.cat([
            prefix_k,
            ctx_k,
            suffix_k,
        ], dim=1)
        
        return new_prompt

    def forward(self):
        # first build the standard prompts from the base class
        prompts = super().forward()

        # 只在 "end" setting 下修改行为
        if self.class_token_position != "end":
            return prompts

        extra_prompts = []
        
        # 1. Append Past OOD classes
        if self.n_past_ood > 0:
            # past_clsname_seqs[i] uses ood_template_idxs[i]
            if hasattr(self, "ood_template_idxs") and torch.is_tensor(self.ood_template_idxs):
                for i, seq in enumerate(self.past_clsname_seqs):
                    if i < int(self.ood_template_idxs.numel()):
                        t = int(self.ood_template_idxs[i].item())
                    else:
                        t = int(getattr(self, "ood_template_idx", 0))
                    extra_prompts.append(self._build_ood_prompt(seq, template_idx=t))
            else:
                for seq in self.past_clsname_seqs:
                    extra_prompts.append(self._build_ood_prompt(seq, template_idx=int(getattr(self, "ood_template_idx", 0))))

        # 2. Append Current OOD class
        if self.is_ood:
            # clsname_seq uses the last template idx
            if hasattr(self, "ood_template_idxs") and torch.is_tensor(self.ood_template_idxs) and self.ood_template_idxs.numel() > 0:
                t = int(self.ood_template_idxs[-1].item())
            else:
                t = int(getattr(self, "ood_template_idx", 0))
            extra_prompts.append(self._build_ood_prompt(self.clsname_seq, template_idx=t))

        # Combine
        if len(extra_prompts) > 0:
            prompts = torch.cat([prompts] + extra_prompts, dim=0)
            # print(f"[OpenSetPromptLearner] num prompts after concat: {prompts.size(0)}")
        return prompts

class ETFhead(nn.Module):
    """Fixed ETF Classifier Head for NC-FSCIL."""
    def __init__(self, in_features, num_classes, eval_classes=None, with_len=False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.eval_classes = eval_classes if eval_classes is not None else num_classes
        self.with_len = with_len

        # Generate ETF
        orth_vec = generate_random_orthogonal_matrix(in_features, num_classes)
        i_nc_nc = torch.eye(num_classes)
        one_nc_nc = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
        # ETF formula: W = U * (I - 1/K) * sqrt(K/(K-1))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(num_classes / (num_classes - 1)))
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('etf_vec', etf_vec)
        
        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
        self.register_buffer('etf_rect', etf_rect)

    def ensure_output_dim(self, target_out_features):
        # ETF is fixed pre-assigned structure.
        # We just update the number of classes we evaluate on.
        if target_out_features > self.num_classes:
            print(f"Warning: Requested {target_out_features} classes but ETF only has {self.num_classes}.")
        self.eval_classes = min(target_out_features, self.num_classes)

    def pre_logits(self, x):
        # Normalize features
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def forward(self, x, return_feat=False):
        # x: (B, D)
        feat = self.pre_logits(x)
        logits = feat @ self.etf_vec.to(feat.dtype)

        # Slicing logic based on eval_classes
        if self.eval_classes < logits.size(1):
            logits = logits[:, :self.eval_classes]
        
        if return_feat:
            return logits, feat
        return logits

class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

        # Infer n_past_ood from CURRENT_STAGE when available.
        # Mapping rule: at stage S, there are max(0, S-1) past OOD stages.
        current_stage = getattr(getattr(cfg, "DATASET", object()), "CURRENT_STAGE", None)
        n_past_ood = max(0, int(current_stage) - 1)
        
        # replace the original prompt learner with the OpenSetPromptLearner
        self.prompt_learner = OpenSetPromptLearner(cfg, classnames, clip_model, n_past_ood=n_past_ood)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        self.dataset = cfg.DATASET.NAME
        self.etf_cfg = cfg.TRAINER.ETFhead
        self.film_cfg = cfg.TRAINER.FILM

        clip_dim = clip_model.text_projection.size(1)
        
        film_cfg = self.film_cfg

        if film_cfg.ETF:
            # cwT module
            self.film_etf_img = FiLM(clip_dim)
            self.film_etf_text = FiLM(clip_dim)
        
        # Initialize Loss
        self.dr_loss = DRLoss(loss_weight=1.0)

        # for base to new, base classes will be 'base'
        # for cross dataset, classes from ImageNet will be 'base'
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'openset') or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            assert self.etf_cfg.TYPE in ['similarity', 'etf']

            # linear classifier
            if self.etf_cfg.TYPE == 'similarity':
                self.etf_proj = nn.Identity()
            elif self.etf_cfg.TYPE == 'etf':
                out_dim = len(classnames) + current_stage
                # max_classes = len(classnames) + 10
                # Assume a large enough number for ETF total classes, e.g. 200
                max_classes = getattr(cfg.DATASET, "TOTAL_NUM_CLASSES", 200)
                # if max_classes < out_dim: max_classes = out_dim + 100
                
                self.etf_proj = ETFhead(clip_dim, max_classes, eval_classes=out_dim)
        else:
            self.etf_proj = nn.Identity()
        
    def forward(self, img, labels=None):
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'openset') or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            return self._forward_base(img, labels)
        else:
            return self._forward_new(img)

    def _forward_base(self, img, labels=None):
        """ forward function for base classes """
        text_feats, img_feats = self._forward_feats(img)
        
        # forward similartiy and linear logits
        logits = self._forward_logits_similarity(text_feats, img_feats)
        logits_etf, labels_etf, features_etf = self._forward_logits_etf(text_feats, img_feats, labels)
        
        if self.prompt_learner.training:
            # while training, return loss of both logits
            return self._loss(logits, labels, logits_etf, labels_etf, features_etf)
        
        if not self.etf_cfg.TEST_TIME_FUSION:
            return logits_etf

        # while inference, fusion both logits and return
        etf_weight = self.etf_cfg.WEIGHT
        logits = (1 - etf_weight) * logits + etf_weight * logits_etf
        return logits
    
    def _forward_new(self, img):
        """ forward function for new classes """
        assert not self.prompt_learner.training
        
        # for new classes, only forward similarity logits
        text_feats, img_feats = self._forward_feats(img)
        logits = self._forward_logits_similarity(text_feats, img_feats)
        return logits
    
    def _forward_feats(self, img):
        prompts = self.prompt_learner()

        tokenized_prompts = self.tokenized_prompts

        # Ensure tokenized_prompts has the same batch size as prompts.
        # OpenSetPromptLearner may append multiple OOD prompts
        # (past OODs + current OOD). We infer how many extra prompts
        # were added by comparing the number of prompts with the
        # base number of classes, and replicate the first row of
        # tokenized_prompts accordingly.
        if self.prompt_learner.class_token_position == "end":
            base_num_classes = self.prompt_learner.n_cls
            total_num_prompts = prompts.size(0)
            extra_prompts = total_num_prompts - base_num_classes

            if extra_prompts > 0:
                tokenized_list = [tokenized_prompts]

                # Append one tokenized prompt row per OOD prompt, using the aligned ood_template_idxs
                if hasattr(self.prompt_learner, "ood_template_idxs") and torch.is_tensor(self.prompt_learner.ood_template_idxs):
                    tids = self.prompt_learner.ood_template_idxs
                    # Expect len(tids) == n_past_ood + 1 when is_ood=True
                    for i in range(int(tids.numel())):
                        t = int(tids[i].item())
                        if t < 0 or t >= tokenized_prompts.size(0):
                            t = 0
                        tokenized_list.append(tokenized_prompts[t : t + 1, :])
                else:
                    # Fallback to replicating current template row
                    t = int(getattr(self.prompt_learner, "ood_template_idx", 0))
                    if t < 0 or t >= tokenized_prompts.size(0):
                        t = 0
                    for _ in range(int(extra_prompts)):
                        tokenized_list.append(tokenized_prompts[t : t + 1, :])

                tokenized_prompts = torch.cat(tokenized_list, dim=0)

        text_feats = self.text_encoder(prompts, tokenized_prompts)
        img_feats = self.image_encoder(img.type(self.dtype))

        return text_feats, img_feats
    
    def _forward_logits_similarity(self, text_feats, img_feats):
        # normalize and calcute cosine similarity
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feats.t()
        return logits
    
    def _forward_logits_etf(self, text_feats, img_feats, labels):
        # cwT module
        if self.film_cfg.ETF:
            text_feats = self.film_etf_text(text_feats)
            img_feats = self.film_etf_img(img_feats)

        # while new head is similarity head, use similarity forward function
        if self.etf_cfg.TYPE == 'similarity':
            return self._forward_logits_similarity(text_feats, img_feats), labels, None

        if hasattr(self.etf_proj, 'ensure_output_dim'):
            current_num_classes = text_feats.size(0)  # Consistent with the number of prompts (base [+ 1 if OOD])
            self.etf_proj.ensure_output_dim(int(current_num_classes))

        if labels is None:
            # while inference, forward image features only
            all_feats = img_feats
            all_labels = labels
        else:
            # while training, image features and text features will be concated to train classifier
            text_feats = text_feats[labels]
            all_feats = torch.cat([text_feats, img_feats])
            all_labels = torch.cat([labels, labels])

        all_feats_out = None
        if isinstance(self.etf_proj, ETFhead):
            all_logits, all_feats_out = self.etf_proj(all_feats, return_feat=True)
        else:
            all_logits = self.etf_proj(all_feats)
            
        return all_logits, all_labels, all_feats_out
    
    def _loss(self, logits, labels, logits_etf, labels_etf, features_etf=None):
        # calculate similarity loss and linear loss
        loss_cls = F.cross_entropy(logits, labels)
        
        if isinstance(self.etf_proj, ETFhead) and features_etf is not None:
            # Use DR Loss
            etf_vec = self.etf_proj.etf_vec
            # labels_etf are indices. target = etf_vec[:, labels_etf].t()
            target = etf_vec[:, labels_etf].t()
            loss_cls_etf = self.dr_loss(features_etf, target)
        else:
            loss_cls_etf = F.cross_entropy(logits_etf, labels_etf)

        etf_weight = self.etf_cfg.WEIGHT
        loss = (1 - etf_weight) * loss_cls + etf_weight * loss_cls_etf
        return loss
    

@TRAINER_REGISTRY.register()
class OpenSetETFCoOp(CoOp):
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = cfg.TRAINER.NAMES_TO_UPDATE

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
                
            param.requires_grad_(update)

        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        print(f"Parameters to be updated: {list(sorted(enabled))}")

        eval_only = getattr(getattr(cfg, "TRAIN", object()), "EVAL_ONLY", False)
        if cfg.MODEL.INIT_WEIGHTS and not eval_only:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim, infos = build_optimizer(self.model, cfg.OPTIM)

        if infos is not None:
            print('Learning rate of parameters:')
            for info in infos:
                print('lr: {}, layers: {}'.format(info['lr'], info['layers']))
        
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("PromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

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
            if epoch < 0:
                all_model_files = os.listdir(osp.join(directory, name))
                all_model_files = [file_ for file_ in all_model_files if file_ != 'checkpoint']
                model_epochs = [int(file_.split('-')[-1]) for file_ in all_model_files]
                last_epoch = max(model_epochs)
                model_file = 'model.pth.tar-' + str(last_epoch)

            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            # for some dataset in domain generalization, number of target classes is different from number of source classes
            # thus a mapping must be created to preserve the required class weights

            # if self.cfg.DATASET.NAME in ['ImageNetA', 'ImageNetR']:
            #     from datasets.imagenet import ImageNet
            #     from dassl.utils import listdir_nohidden

            #     # read classes from source dataset
            #     dataset = self.dm.dataset
            #     text_file = osp.join(dataset.dataset_dir, "classnames.txt")
            #     all_folders = ImageNet.read_classnames(text_file).keys()

            #     # read classes from target dataset
            #     TO_BE_IGNORED = ["README.txt"]
            #     folders = listdir_nohidden(dataset.image_dir, sort=True)
            #     folders = [f for f in folders if f not in TO_BE_IGNORED]

            #     # find that which class from target dataset is in source dataset
            #     is_reserves = [f in folders for f in all_folders]

            #     # only reserve required class weights
            #     print(f'State dict is CLIPPED to match the shape of target dataset {self.cfg.DATASET.NAME}!')
            #     state_dict['linear_probe_proj.weight'] = state_dict['linear_probe_proj.weight'][is_reserves]
            #     state_dict['linear_probe_proj.bias'] = state_dict['linear_probe_proj.bias'][is_reserves]
                
            # set strict=False
            # --- Incremental Learning Logic: Migrate Weights ---
            # Check if we need to migrate the previous 'clsname_seq' to 'past_clsname_seqs'
            if hasattr(self._models[name], "prompt_learner") and hasattr(self._models[name].prompt_learner, "n_past_ood"):
                n_past_model = self._models[name].prompt_learner.n_past_ood
                
                # Count past seqs in checkpoint
                past_keys_in_ckpt = [k for k in state_dict.keys() if "prompt_learner.past_clsname_seqs" in k]
                n_past_ckpt = len(past_keys_in_ckpt)
                
                # Case: Moving to next stage (e.g. from 1 OOD to 2 OODs)
                # Checkpoint has N past, Model has N+1 past.
                # We need to move ckpt['clsname_seq'] -> state_dict['past_clsname_seqs.{N}']
                if n_past_model == n_past_ckpt + 1:
                    print(f"Incremental Stage Detected: Migrating previous 'clsname_seq' to 'past_clsname_seqs.{n_past_ckpt}'")
                    
                    src_key = "prompt_learner.clsname_seq"
                    tgt_key = f"prompt_learner.past_clsname_seqs.{n_past_ckpt}"
                    
                    if src_key in state_dict:
                        state_dict[tgt_key] = state_dict[src_key].clone()
                        # Remove src_key so that the current model's clsname_seq remains freshly initialized (random/mixed)
                        del state_dict[src_key]
                    else:
                        print("Warning: 'clsname_seq' not found in checkpoint during migration.")

                    # Expand ood_template_idxs by appending the current stage template idx.
                    # This keeps alignment: past_clsname_seqs[i] <-> ood_template_idxs[i], clsname_seq <-> ood_template_idxs[-1].
                    tids_key = "prompt_learner.ood_template_idxs"
                    old_tids = None
                    if tids_key in state_dict and torch.is_tensor(state_dict[tids_key]):
                        old_tids = state_dict[tids_key].to(dtype=torch.long).view(-1)
                    else:
                        # Backward compatible: older ckpt only has a single int attribute; treat as one stage with template 0.
                        old_tids = torch.zeros((n_past_ckpt + 1,), dtype=torch.long)

                    cur_t = int(getattr(self._models[name].prompt_learner, "ood_template_idx", 0))
                    dev = old_tids.device
                    new_tids = torch.cat(
                        [old_tids, torch.tensor([cur_t], dtype=torch.long, device=dev)], dim=0
                    )
                    state_dict[tids_key] = new_tids

            self._models[name].load_state_dict(state_dict, strict=False)
