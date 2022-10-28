import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser=argparse.ArgumentParser()

    # expeirment info
    parser.add_argument('-project', type=str, default='IPET')
    parser.add_argument('-name', type=str, required=True)
    parser.add_argument('-tags', type=str, required=True)
    parser.add_argument('-description', type=str, default='')

    # dir
    parser.add_argument('-path_logging', type=str, default='/source')
    parser.add_argument('-path_training_dataset', type=str, default='../datafiles/speechcommand_train_data.json')
    parser.add_argument('-path_validation_dataset', type=str, default='../datafiles/speechcommand_valid_data.json')
    parser.add_argument('-path_evaluation_dataset', type=str, default='../datafiles/speechcommand_eval_data.json')
    parser.add_argument('-path_data_label', type=str, default='../datafiles/class_labels_indices.csv')

    # device
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-usable_gpu', type=str, default='0,1,2,3')

    # hyper-parameters
    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-lr', type = float, default = 1e-3) 
    parser.add_argument('-lr_decay_start_epoch', type=int, default=5)
    parser.add_argument('-lr_decay_end_epoch', type=int, default=26)
    parser.add_argument('-gamma', type = float, default = 0.85) 
    parser.add_argument('-weigth_decay', type = float, default = 5e-7) 
    parser.add_argument('-classification_loss', type=str, default='bce')

    # training setting
    parser.add_argument('-number_iteration_for_log', type=int, default=10)
    parser.add_argument('-rand_seed', type=int, default=1234)
    parser.add_argument('-flag_reproduciable', type = str2bool, nargs='?', const=True, default = True)
    
    #model architectures
    parser.add_argument('-backend_module_name', type=str, default='head')
    parser.add_argument('-model_type', type=str, default='facebook/wav2vec2-base')
    parser.add_argument('-finetuning_task', type=str, default='audio-classification')

    #parameter-efficient transfer learning methods
    parser.add_argument('-input_prompt', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-input_prompt_num', type=int, default=10)
    parser.add_argument('-embedding_prompt', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-embedding_prompt_num', type=int, default=64)
    parser.add_argument('-adapter', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-adapter_scalar', type = float, default = 0.1) 
    parser.add_argument('-adapter_hidden_dim', type = int, default = 64) 

    #data processing
    parser.add_argument('-frame_length', type = int, default = 1024) 
    
    #data augmentation
    parser.add_argument('-mixup', type = float, default = 0) 
    
    args=parser.parse_args()

    return args