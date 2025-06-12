import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from src.utils_data import *
from src.transformation_utils import *
from src.utils_all import *

from src.MAE.utils_mae import *
from src.MAE.trainer_mae import *
from src.MAE.MAE_1D import *
from src.MAE.multi_trait import *
from src.MAE.trainer_trait import *

import glob
import gc
from datetime import datetime

import torch
import torch.nn as nn
import wandb 

#######################
# Global Configuration
#######################
import argparse

# Create the parser
my_parser = argparse.ArgumentParser(description='Training semi-supervised method')

# Add the arguments
my_parser.add_argument('--seed',
                       metavar='seed',
                       type=int,default=155,
                       help='SEED')


my_parser.add_argument('--directory_path',
                       metavar='directory_path',
                       default=str(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),  'Splits')),
                       type=str,
                       help='Path the unlabeled splits')

my_parser.add_argument('--path_save',
                       metavar='path_save',
                       default=str(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),  '/mae')),
                       type=str,
                       help='the path to checkpoint and metadata')


my_parser.add_argument('--input_shape',
                       metavar='input_shape',
                       type=int,default=1720,
                       help='input_shape MAE')

my_parser.add_argument('--type_s',
                       metavar='type_s',
                       type=str,default='full',
                       help='Type of the sensor: full OR half range')


my_parser.add_argument('--n_epochs',
                       metavar='n_epochs',
                       type=int,default=500,
                       help='Number of epochs')

my_parser.add_argument('--batch_size',
                       metavar='batch_size',
                       type=int,default=128,
                       help='batch_size')


my_parser.add_argument('--lr',
                       metavar='lr',
                       type=float,default=5e-4,
                       help='Learning rate')

my_parser.add_argument('--weight_decay',
                       metavar='weight_decay',
                       type=float,default=1e-4,
                       help='weight_decay')

my_parser.add_argument('--mask_ratio',
                       metavar='mask_ratio',
                       type=float,default=0.75,
                       help='mask_ratio')

my_parser.add_argument('--augmentation',
                       metavar='augmentation',
                       type=bool, default=True,
                       help='augmentation')

my_parser.add_argument('--scale',
                       metavar='scale',
                       type=bool, default=False,
                       help='scale')

my_parser.add_argument('--name_experiment',
                       metavar='name_experiment',
                       type=str,default='',
                       help='name_experiment')


my_parser.add_argument('--project_wandb',
                       metavar='project_wandb',
                       type=str,default='MAE_wandb_test',
                       help='project_wandb')


# Execute the parse_args() method
args = my_parser.parse_args()

path_save = args.path_save ##path_save
project = args.project_wandb ## experiment name 

directory_path = args.directory_path

seed = args.seed

batch_size = args.batch_size 
n_epochs = args.n_epochs

lr = args.lr
weight_decay = args.weight_decay
mask_ratio = args.mask_ratio
augmentation = args.augmentation
scale = args.scale

name_experiment = args.name_experiment ## experiment name 

input_shape = args.input_shape
type_s = args.type_s
###########################

device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")

ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
percentage_tr = 1

if __name__ == "__main__":

    # Summarize GPU memory usage
    print(torch.cuda.memory_summary())

    file_paths = glob.glob(os.path.join(directory_path, "*.csv"))
    file_paths = file_paths[:int(percentage_tr*len(file_paths))]

    run_mae = 'MAE_{}_Training_{}_{}UNlabels_{}'.format(formatted_datetime, name_experiment, 100*percentage_tr, seed)

    checkpoint_dir_mae = os.path.join(path_save, "checkpoints_{}".format(run_mae))

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    if not os.path.exists(checkpoint_dir_mae):
        os.mkdir(checkpoint_dir_mae)
    
    #####
    settings_dict_mae = {
        'seed': seed,
        'epochs': n_epochs,
        'batch_size': batch_size,
        'augmentation': augmentation,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'load_model_path': None, #load_model_path
        'file_paths': file_paths,
        
        'w_loss': 1,
        'mask_ratio': mask_ratio,
        'n_bands': input_shape,
        'type':type_s,

        'seq_size': 20,
        'in_chans': 1,
        'embed_dim': 128,
        'depth': 10, #6,
        'num_heads': 16, #4,
        'decoder_embed_dim': 128,
        'decoder_depth': 6, #4
        'decoder_num_heads': 4,
        'mlp_ratio': 4.0,
        'norm_layer': nn.LayerNorm,
        'cls_token': False,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
        'checkpoint_dir': checkpoint_dir_mae,
        'early_stop': True,
        'patience': 10,
    }
    
    
    sets = Settings()
    sets.update_from_dict(settings_dict_mae)
    
    # wandb.init(
    # # Set the project where this run will be logged
    # project=project,
    # # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    # name=f"experiment_{run_mae}",
    # # Track hyperparameters and run metadata
    # config=settings_dict_mae,
    # dir = checkpoint_dir_mae
    # )
    
    test = Trainer(sets)
    # test.settings.logger = wandb
    test.train()

    if (test.settings.logger is not None):
        test.settings.logger.finish()

    # Clean up after training
    del test
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Completed training model {percentage_tr}. GPU memory cleared.\n")