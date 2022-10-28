import math
from random import random
import torch
import torch.utils.data as td
import torchaudio as ta
import numpy as np
import warnings
import torch.nn.functional
import utils.util as util
import random 

class Kaldi_Fbank():
    """Extract Kaldi_fbank from raw x using torchaudio.
    """
    def __init__(self, args):
        super(Kaldi_Fbank, self).__init__()
        
        self.args = args

    def __call__(self, x, sr, x2 = None):
        # mixup
        if x2 is not None:
            if x.shape[1] != x2.shape[1]:
                if x.shape[1] > x2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, x.shape[1])
                    temp_wav[0, 0:x2.shape[1]] = x2
                    x2 = temp_wav
                else:
                    # cutting
                    x2 = x2[0, 0:x.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_x = mix_lambda * x + (1 - mix_lambda) * x2.to(self.args.device)
            x = mix_x - mix_x.mean()

        fbank = ta.compliance.kaldi.fbank(x, frame_length=self.args.winlen, frame_shift=self.args.winstep, htk_compat=True, sample_frequency=sr,
                                                  window_type=self.args.winfunc, num_mel_bins=self.args.nfilts, dither=0.0)

        n_frames = fbank.shape[0]

        p = self.args.frame_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.args.frame_length, :]

        if x2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

def get_loaders(args, dataset):
    train_set = TrainSet(args, dataset)
    train_set_sampler = td.DistributedSampler(train_set, shuffle=True)
    
    validation_set = TestSet(args, dataset.validation_set, dataset.classes_labels)
    validation_set_sampler = td.DistributedSampler(validation_set, shuffle=False)

    evaluationt_set = TestSet(args, dataset.evaluation_set, dataset.classes_labels)
    evaluationt_set_sampler = td.DistributedSampler(evaluationt_set, shuffle=False)

    train_loader = td.DataLoader(
        train_set,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=train_set_sampler
    )

    validation_loader = td.DataLoader(
        validation_set,
        batch_size=args.batch_size * 2,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=validation_set_sampler
    )

    evaluation_loader = td.DataLoader(
        evaluationt_set,
        batch_size=args.batch_size * 2,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=evaluationt_set_sampler
    )
    
    return train_set, train_set_sampler, train_loader, validation_set, validation_loader, evaluationt_set, evaluation_loader

class TrainSet(td.Dataset):

    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.items = dataset.train_set
        self.fbank = Kaldi_Fbank(args)
        # set label
        count = 0
        
        
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        
        # read wav
        x, sr = ta.load(item.path)
        x = x - x.mean()
        x = x.to(self.args.device)
        mix_x = None

        if random.random() < self.args.mixup:
            mix_sample_idx = random.randint(0, len(self.items)-1)
            mix_item = self.items[mix_sample_idx]

            mix_x, sr = ta.load(mix_item.path)
            mix_x = mix_x - mix_x.mean()
            mix_x = mix_x.to(self.args.device)

        fbank, mix_lambda = self.fbank(x, sr, mix_x)
        # SpecAug, not do for eval set
        freqm = ta.transforms.FrequencyMasking(self.args.freqm)
        timem = ta.transforms.TimeMasking(self.args.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        
        fbank = fbank.unsqueeze(0)
        if self.args.freqm != 0:
            fbank = freqm(fbank)
        if self.args.timem != 0:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.cpu()
        # normalize the input for both training and test
        if not self.args.skip_norm:
            fbank = (fbank - self.args.norm_mean) / (self.args.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.args.add_noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        if mix_lambda == 0:
            label_indices = np.zeros(self.args.num_classes)
            
            for label_str in item.label.split(','):    #one_hot for multi label classification (speech, audioset)
                label_indices[int(self.dataset.classes_labels[label_str])] = 1.0    

            label_indices = torch.FloatTensor(label_indices) 
        else: 
            label_indices = np.zeros(self.args.num_classes)
            # add sample 1 labels
            for label_str in item.label.split(','):
                label_indices[int(self.dataset.classes_labels[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_item.label.split(','):
                label_indices[int(self.dataset.classes_labels[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
            
        return fbank, label_indices


class TestSet(td.Dataset):
    
    def __init__(self, args, dataset, classes_labels):
        self.args = args
        self.items = dataset
        self.classes_labels = classes_labels
        self.fbank = Kaldi_Fbank(args)

    def __len__(self):
        return len(self.items)


    def __getitem__(self, index):
        item = self.items[index]

        # read wav
        x, sr = ta.load(item.path)
        x = x - x.mean()
        x = x.to(self.args.device)

        fbank, mix_lambda = self.fbank(x, sr)
        fbank = fbank.cpu()
        if not self.args.skip_norm:
            fbank = (fbank - self.args.norm_mean) / (self.args.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        label_indices = np.zeros(self.args.num_classes)
        
        for label_str in item.label.split(','):    #one_hot for multi label classification (speech, audioset)
            label_indices[int(self.classes_labels[label_str])] = 1.0    

        label_indices = torch.FloatTensor(label_indices) 

        return fbank, label_indices, item.path
