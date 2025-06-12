import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.transformation_utils import *
from src.utils_all import *
from src.utils_data import *

from src.MAE.utils_mae import *
from src.MAE.trainer_mae import *
from src.MAE.MAE_1D import *
from src.MAE.multi_trait import *
from src.MAE.trainer_trait import *


############
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

my_parser.add_argument('--path_save',
                       metavar='path_save',
                       type=str,
                       help='the path to checkpoint and metadata')

my_parser.add_argument('--input_shape',
                       metavar='input_shape',
                       type=int,default=1721,
                       help='input_shape AE_RTM')

my_parser.add_argument('--type_s',
                       metavar='type_s',
                       type=str,default='full',
                       help='Type of the sensor: full OR half range')


# Execute the parse_args() method
args = my_parser.parse_args()

seed = args.seed

path_save = args.path_save ##path_save
path_data_lb = args.path_data_lb

input_shape = args.input_shape
type_s = args.type_s


if __name__ == "__main__":

    seed_all(seed=seed) ###155

    for percentage_tr in [1, 0.8, 0.6, 0.4, 0.2]:
        
        # Summarize GPU memory usage
        print(torch.cuda.memory_summary())
        ##################

        # ######### Validation ###
        # mean_metrics, std_metrics = run_consistent_experiment(path_save, path_data_lb, [155, 381, 187], fine_tune=False, n_epochs = 200, percentage_tr=percentage_tr, type_sp=type_s, n_bands=input_shape,  save=True, name='LP_{}'.format(percentage_tr))
        # mean_metrics, std_metrics = run_consistent_experiment(path_save, path_data_lb, [155, 381, 187], fine_tune=True, n_epochs = 80, percentage_tr=percentage_tr, type_sp=type_s, n_bands=input_shape,  save=True, name='FT_{}'.format(percentage_tr))
        
      ######### Validation ###
        mean_metrics, std_metrics = run_consistent_experiment(path_data_lb, [155, 381, 187], fine_tune=False, n_epochs = 200, percentage_tr=percentage_tr, type_sp=type_s, n_bands=input_shape,  save=True, name='LP_{}'.format(percentage_tr), checkpoint_dir_mae=path_save)
        mean_metrics, std_metrics = run_consistent_experiment(path_save, path_data_lb, [155, 381, 187], fine_tune=True, n_epochs = 80, percentage_tr=percentage_tr, type_sp=type_s, n_bands=input_shape,  save=True, name='FT_{}'.format(percentage_tr), checkpoint_dir_mae=path_save)
