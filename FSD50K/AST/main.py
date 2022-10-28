import os
import random
import importlib
import numpy as np

import torch
import torch.nn as nn

import arguments
import trainers.traintest as traintest
import data.data_loader as data
from data.dataset import Datasets
from utils.util import init_weights
from log.controller import LogModuleController

def set_experiment_environment(args):
    # reproducible
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.backends.cudnn.deterministic = args.flag_reproduciable
    torch.backends.cudnn.benchmark = not args.flag_reproduciable

    # DDP env
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4021'
    args.rank = args.process_id
    args.device = 'cuda:' + args.gpu_ids[args.process_id]
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.distributed.init_process_group(
            backend='nccl', world_size=args.world_size, rank=args.rank)

def run(process_id, args):
    # check parent process
    args.process_id = process_id
    args.flag_parent = process_id == 0
    
    # experiment environment
    set_experiment_environment(args)
    trainer = traintest.ModelTrainer()
    trainer.args = args
    
    # logger
    if args.flag_parent:
        logger = LogModuleController.Builder(args.name, args.project
        ).tags(args.tags
        ).description(args.description
        ).save_source_files(os.path.dirname(os.path.realpath(__file__))
        ).use_local(args.path_logging)
        logger = logger.build()
        trainer.logger = logger

    # dataset
    modelue = importlib.import_module("models.{}".format(args.backend_module_name)).__getattribute__('Model')

    trainer.dataset = Datasets(args)
    
    args.num_classes = len(trainer.dataset.classes_labels)
    
    # data loader
    loaders = data.get_loaders(args, trainer.dataset)
    trainer.train_set, trainer.train_set_sampler, trainer.train_loader, trainer.validation_set, trainer.validation_loader, trainer.evaluation_set, trainer.evaluation_loader = loaders[0], loaders[1], loaders[2], loaders[3], loaders[4], loaders[5], loaders[6]

    
    #backbone pm
    modelue = importlib.import_module("models.{}".format(args.main_module_name)).__getattribute__('Model')
    model = modelue(
        args = args,
        label_dim = args.num_classes, 
        fstride = args.fstride, 
        tstride = args.tstride, 
        input_fdim = args.nfilts, 
        input_tdim = args.frame_length, 
        imagenet_pretrain = args.imagenet_pretrain, 
        audioset_pretrain = args.audioset_pretrain, 
        model_size = args.model_size
        ).to(args.device)

    #head
    modelue = importlib.import_module("models.{}".format(args.backend_module_name)).__getattribute__('Model')
    classification_head = modelue(
        embd_dim = model.v.pos_embed.shape[2],
        label_dim = args.num_classes
    ).to(args.device)
    classification_head.apply(init_weights)

    if args.flag_parent:
        nb_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        nb_mlp_params = sum(p.numel() for p in classification_head.parameters() if p.requires_grad)
        nb_all_params = nb_model_params + nb_mlp_params
        trainer.logger.log_text('nb_params', str(nb_all_params))
        args.nb_all_params = str(nb_all_params)
        args.nb_model_params = str(nb_model_params)
        args.nb_mlp_params = str(nb_mlp_params)
        print('Model parameter #: ',args.nb_model_params)
        print('Classification MLP parameter #: ',args.nb_mlp_params)
        print('All parameter #: ',args.nb_all_params)
        trainer.logger.log_parameter(vars(args))
    
        
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)
    classification_head = nn.SyncBatchNorm.convert_sync_batchnorm(classification_head)
    classification_head = nn.parallel.DistributedDataParallel(classification_head, device_ids=[args.device], find_unused_parameters=False)
    trainer.model = model
    trainer.classification_head = classification_head
    
    # criterion
    criterion = {}
    
    classification_loss_function = importlib.import_module('loss.'+ args.classification_loss).__getattribute__('LossFunction')
    criterion['classification_loss'] = classification_loss_function()
    
    trainer.criterion = criterion
    
    # optimizer
    trainer.optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classification_head.parameters()), 
        lr=args.lr, 
        weight_decay=args.weigth_decay,
        amsgrad = args.amsgrad,
        betas = (0.95, 0.999)
    )


    args.number_iteration = len(trainer.train_loader)
    

    #lr scheduler
    trainer.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    trainer.optimizer, list(range(args.lr_decay_start_epoch, args.lr_decay_end_epoch)), gamma=args.gamma)
    
    trainer.run()
    
    if args.flag_parent:
        trainer.logger.finish()


if __name__ == '__main__':
    # get arguments
    args = arguments.get_args()
    
    # set reproducible
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    # set gpu device
    if args.usable_gpu is None: 
        args.gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        args.gpu_ids = args.usable_gpu.split(',')
    
    if len(args.gpu_ids) == 0:
        raise Exception('Only GPU env are supported')

    # set DDP
    args.world_size = len(args.gpu_ids)
    args.batch_size = args.batch_size // (args.world_size)
    args.num_workers = args.num_workers // args.world_size
    
    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run, 
        nprocs=args.world_size, 
        args=(args,)
    )
    
