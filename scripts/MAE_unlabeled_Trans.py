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

path_save = args.path_save
path_data_lb = args.path_data_lb

input_shape = args.input_shape
type_s = args.type_s

##################

############## TRansferabiloty ###
eval_scores, ext_scores = run_consistent_experimentCV(path_save, [155], fine_tune=False, n_epochs = 200, percentage_tr=1, type_sp=type_s, n_bands=input_shape,  save=True,  start=0, end=None)
eval_scores, ext_scores = run_consistent_experimentCV(path_save, [155], fine_tune=True, n_epochs = 80, percentage_tr=1, type_sp=type_s, n_bands=input_shape,  save=True,  start=0, end=None)
