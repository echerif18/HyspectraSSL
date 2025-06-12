# Import all necessary modules and functions from local utilities and MAE packages.
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
from datetime import datetime

import numpy as np
import pandas as pd
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

my_parser.add_argument('--path_data_lb',
                       metavar='path_data_lb',
                       default=str(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),  'Datasets/50_all_traits.csv')),
                       type=str,
                       help='Path the labeled data')

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
path_data_lb = args.path_data_lb

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

##############
# Set the device to GPU if available, otherwise CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format (used to tag experiment runs)
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")


ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]

#######################
# Grid Search Setup
#######################
# # Define parameter grid for grid search over transformer depth and number of heads.
depth = 10
num_heads = 16
param_grid = list([1, 0.1, 0.01, 0.001,  0.])


if __name__ == "__main__":
    # Initialize containers to store results from grid search.
    results = []
    results_DS = []

    # # Loop over each combination of depth and number of heads.
    for w_loss in param_grid:
        # Create a unique run name using the formatted datetime, depth, num_heads, and seed.
        run_mae = 'MAE_{}_Training_w_loss{}_{}UNlabels_{}'.format(formatted_datetime, w_loss, name_experiment, seed)
        
        
        # Define a checkpoint directory for this experiment run.
        checkpoint_dir_mae = os.path.join(path_save, "checkpoints_{}".format(run_mae))

        if not os.path.exists(path_save):
            os.mkdir(path_save)

        if not os.path.exists(checkpoint_dir_mae):
            os.mkdir(checkpoint_dir_mae)

        file_paths = glob.glob(os.path.join(directory_path, "*.csv"))
        
        # Prepare a dictionary with all the training settings for the MAE.
        settings_dict_mae = {
            'seed': seed,  # Seed for reproducibility.
            'epochs': n_epochs,
            'batch_size': batch_size,
            'augmentation': augmentation,
            'learning_rate': lr,
            'weight_decay': weight_decay,

            'file_paths': file_paths,
            'mask_ratio': mask_ratio,
            'w_loss': w_loss,
            
            'n_bands': input_shape,
            'type':type_s,
            'seq_size': 20,
            'in_chans': 1,
            'embed_dim': 128,
            'depth': depth,  # Using current depth from grid.
            'num_heads': num_heads,  # Using current number of heads.
            'decoder_embed_dim': 128,
            'decoder_depth': 6,  # Fixed decoder depth.
            'decoder_num_heads': 4,
            'mlp_ratio': 4.0,
            'norm_layer': nn.LayerNorm,
            'cls_token': False,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        
            'checkpoint_dir': checkpoint_dir_mae,
            'early_stop': True,
            'patience': 10,
        }
        
        # Instantiate a settings object and update it with the above dictionary.
        sets = Settings()
        sets.update_from_dict(settings_dict_mae)
        
        # # Initialize a new wandb run for logging experiment details.
        # wandb.init(
        #     project=project,            # Project name in wandb.
        #     name=f"experiment_{run_mae}",  # Unique run name.
        #     config=settings_dict_mae,      # Log the configuration parameters.
        #     dir=checkpoint_dir_mae
        # )
        
        # Instantiate the MAE trainer with the current settings.
        test = Trainer(sets)
        # test.settings.logger = wandb  # Attach wandb logger to the trainer.
        test.train()  # Start training the model.
        
        # Finish the wandb run after training is complete.
        if (test.settings.logger is not None):
            test.settings.logger.finish()

        # Save the final validation loss for this run.
        final_val_loss = test.valid_loss_list
        results.append((w_loss, final_val_loss[-1]))
        
        # Run downstream evaluation on the trained model.
        # val_r2, val_nrmse = downstream(test.model)
        # val_r2, val_nrmse = run_consistent_experiment(checkpoint_dir_mae, path_data_lb, save=True, n_epochs=1)
        mean_metrics, std_metrics= run_consistent_experiment(checkpoint_dir_mae, path_data_lb, save=True)

        val_r2, val_nrmse = mean_metrics.values[:, 0], mean_metrics.values[:, 2]
        # Save selected evaluation metrics using quantiles (70th percentile).
        results_DS.append((w_loss, np.mean(val_r2), np.mean(val_nrmse)))


    # Compile the results from grid search into pandas DataFrames.
    df_results = pd.DataFrame(results, columns=["w_loss", "Final_Val_Loss"])
    df_DS_results = pd.DataFrame(results_DS, columns=["w_loss", "Final_Val_r2", "Final_Val_nrmse"])

    # Save the results to CSV files in the checkpoint directory.
    df_results.to_csv(os.path.join(checkpoint_dir_mae, 'experimentMAE_{}_{}_ValLossLast.csv'.format(formatted_datetime, seed)))
    df_DS_results.to_csv(os.path.join(checkpoint_dir_mae, 'experimentMAE_{}_{}_ValDSLast.csv'.format(formatted_datetime, seed)))