# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import importlib
import timm
from timm.models.layers import trunc_normal_
from .vision_transformer import _create_vision_transformer
from .input_prompt import Input_prompt

class Model(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    
    Reference:
	Gong, Yuan, Yu-An Chung, and James Glass.
	"Ast: Audio spectrogram transformer." Interspeech. 2021.
    """
    def __init__(self, args, label_dim, fstride, tstride, input_fdim, input_tdim, imagenet_pretrain, audioset_pretrain, model_size, verbose=True):

        super(Model, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        self.args = args
        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = _create_vision_transformer('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain, distilled=True)
            elif model_size == 'small224':
                self.v = _create_vision_transformer('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain, distilled=True)
            elif model_size == 'base224':
                self.v = _create_vision_transformer('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain, distilled=True)
            elif model_size == 'base384':
                self.v = _create_vision_transformer('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain, distilled=True, args=args)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches # 2D Image to Patch Embedding (in ViT)
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            # TODO can use sinusoidal positional embedding instead
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            #sd = torch.load('../pretrained_models/audioset_10_10_0.4593.pth', map_location=device) #audioset_pretrain
            audio_model = Model(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', args= self.args, verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            msg = audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches)) #300

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

            
            #frozen AST original parameters
            if self.args.input_prompt or self.args.embedding_prompt or self.args.adapter:
                for name, p in audio_model.named_parameters():
                    if name in msg.missing_keys:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            
            #define input prompt
            if self.args.input_prompt:
                self.input_prompter = Input_prompt(
                    prompt_size = args.input_prompt_num,
                    input_tdim = args.frame_length,
                    input_fdim = args.nfilts
                )

            #define embedding prompt
            if self.args.embedding_prompt:
                assert self.args.embedding_prompt_num > 0, self.args.embedding_prompt_num
                for i in range(12):
                    setattr(self, 'embedding_prompt{}'.format(i), nn.Parameter(torch.empty(1, self.args.embedding_prompt_num, 768)))
                    torch.nn.init.xavier_uniform_(getattr(self, 'embedding_prompt{}'.format(i)).data)


    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        if self.args.input_prompt:
            x = self.input_prompter(x)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)         # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.num_tokens > 0 else None
        dist_token = self.v.dist_token.expand(B, -1, -1)        # self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        x = torch.cat((cls_tokens, dist_token, x), dim=1) 
        x = x + self.v.pos_embed                                # self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, self.embed_dim))
        x = self.v.pos_drop(x)                                  # dropout
        for idx, blk in enumerate(self.v.blocks):                               # Transform blocks
            if self.args.embedding_prompt:
                eee = getattr(self, 'embedding_prompt{}'.format(idx)).expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            x = blk(x)
            if self.args.embedding_prompt:
                x = x[:, self.args.embedding_prompt_num:, :]
        x = self.v.norm(x)                                      # self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        x = (x[:, 0] + x[:, 1]) / 2
        
        return x

