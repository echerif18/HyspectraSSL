import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from src.utils_data import *
from src.transformation_utils import *
from src.utils_all import *
from src.Multi_trait.multi_model import *
from src.Multi_trait.trainer_multi import *

import pandas as pd
from datetime import datetime

import gc
import wandb 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from itertools import islice


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
                       default=str(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),  '/multi')),
                       type=str,
                       help='the path to checkpoint and metadata')


my_parser.add_argument('--input_shape',
                       metavar='input_shape',
                       type=int,default=1720,
                       help='input_shape')

my_parser.add_argument('--type_s',
                       metavar='type_s',
                       type=str,default='full',
                       help='Type of the sensor: full OR half range')


my_parser.add_argument('--n_epochs',
                       metavar='n_epochs',
                       type=int,default=150,
                       help='Number of epochs')

my_parser.add_argument('--batch_size',
                       metavar='batch_size',
                       type=int,default=256,
                       help='batch_size')


my_parser.add_argument('--lr',
                       metavar='lr',
                       type=float,default=5e-4,
                       help='Learning rate')

my_parser.add_argument('--weight_decay',
                       metavar='weight_decay',
                       type=float,default=1e-4,
                       help='weight_decay')

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
                       type=str,default='multitrait_withScaler',
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
augmentation = args.augmentation
scale = args.scale

name_experiment = args.name_experiment ## experiment name 

input_shape = args.input_shape
type_s = args.type_s
###########################

# Check if GPU is available
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print("GPU is available. Using GPU for computation.")
else:
    # If GPU is not available, fall back to CPU
    device = torch.device("cpu")
    print("GPU is not available. Using CPU for computation.")


# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")

ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
percentage_tr = 1


######## Data ########
file_paths = glob.glob(os.path.join(directory_path, "*.csv"))

################ Lbeled ###############
db_lb = pd.read_csv(path_data_lb, low_memory=False).drop(['Unnamed: 0'], axis=1)


eval_scores = {}
ext_scores = {}


for gp, (db_lb_all, samples_val_ext, test_ids) in islice(enumerate(sliding_custom_cv(db_lb, seed=42)), 0, None):
# for gp, (db_lb_all, samples_val_ext, test_ids) in islice(enumerate(sliding_custom_dataset(db_lb, seed=42)), 0, None):
    run = 'multitrait_{}_{}labels_{}_Trans'.format(formatted_datetime, gp, seed)
    checkpoint_dir = os.path.join(path_save, "checkpoints_{}".format(run)) #'./checkpoints'
    print(run)

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ################ Data ###############
    ext_all = samples_val_ext.copy()
    
    ext_val_x = feature_preparation(ext_all.loc[:, '400':'2500']).loc[:, 400:2450]
    ext_val_y = ext_all[ls_tr]

    x_p_val = torch.tensor(ext_val_x.values, dtype=torch.float)
    lb_p_val = torch.tensor(ext_val_y.values, dtype=torch.float)

    X_labeled, y_labeled = data_prep_db(db_lb_all, ls_tr)
    metadata = db_lb_all.iloc[:, :1]  # The metadata (dataset of origin)
    
    red_ed = X_labeled.loc[:,750]
    red_end = X_labeled.loc[:,1300]
    red1000_ = X_labeled.loc[:,1000]
    
    idx = X_labeled[(red_end>red1000_) & (red_ed>red1000_)].index
    
    if(len(idx)>0):
        # X_labeled.loc[idx,:].T.plot(legend=False)
        X_labeled.drop(idx, inplace=True)
        y_labeled.drop(idx, inplace=True)
        metadata.drop(idx, inplace=True)
    
    
    # Split labeled data into train (80%), validation (20%)
    fr_sup, X_val= train_test_split(X_labeled, test_size=0.2, stratify=metadata.dataset, random_state=300)
    
    y_sup = y_labeled.loc[fr_sup.index,:]
    y_val = y_labeled.loc[X_val.index,:]
    
    meta_train = metadata.loc[fr_sup.index,:]
    meta_val = metadata.loc[X_val.index,:]
    
    if(percentage_tr<1):
        fr_sup, _= train_test_split(fr_sup, test_size=1-percentage_tr, stratify=meta_train.dataset, random_state=300)
        
        y_sup = y_sup.loc[fr_sup.index,:]
        meta_train = meta_train.loc[fr_sup.index,:]
    
    
    # db_tr = balanceData(pd.concat([fr_sup, y_sup], axis=1), meta_train, ls_tr, random_state=300,percentage=1)
    db_tr = pd.concat([meta_train, fr_sup, y_sup], axis=1)
    
    fr_sup = db_tr.loc[:, 400:400+input_shape] 

    y_sup = db_tr.loc[:,'cab':]
    meta_train = db_tr.iloc[:,:1]
    
    
    # Create the dataset
    train_dataset = SpectraDataset(fr_sup, y_sup, meta_train, augmentation=True, aug_prob=0.7) ### FR: aug_prob=0.7
    # Define DataLoader with the custom collate function for fair upsampling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = SpectraDataset(X_train=X_val, y_train=y_val, meta_train=meta_val, augmentation=False)
    # Create DataLoader for the test dataset
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ext_dataset = TensorDataset(x_p_val, lb_p_val)
    ext_loader = DataLoader(ext_dataset, batch_size=batch_size, shuffle=False)
    
    ########## Scaler ###
    # scaler_list = None
    scaler_model = save_scaler(y_sup, standardize=True, scale=True)

    
    # Example usage:
    settings_dict = {
        'epochs': n_epochs,
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'checkpoint_dir': checkpoint_dir,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': 1e-4,
        'early_stopping':True,
        'early_stop': True,
        'patience': 10,
        'scaler_model': scaler_model,
        #'logger':wandb
    }

    # wandb.init(
    #     # Set the project where this run will be logged
    #     project=project,
    #     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #     name=f"experiment_{run}",
    #     # Track hyperparameters and run metadata
    #     config=settings_dict,
    #     dir = checkpoint_dir
    #     )
    
    
    sets = Settings()
    # sets.pretrained_model = test_mae.model
    sets.update_from_dict(settings_dict)
    
    test_reg = Trainer_MultiTraits(sets)
    # test_reg.settings.logger = wandb  # Attach wandb logger to the trainer.
    test_reg.train()
    
    if (test_reg.settings.logger is not None):
        test_reg.settings.logger.finish() 

    def evaluate(loader, save=False, path=None):
        outputs, labels = [], []
        for batch in tqdm(loader):
            x, y = batch[:2]
            x = x.to(sets.device).unsqueeze(dim=1)[:, :]
            y = y.to(sets.device)
            with torch.no_grad():
                pred = test_reg.model(x.float())
                if test_reg.transformation_layer_inv:
                    pred = test_reg.transformation_layer_inv(pred)
                outputs.append(pred.cpu().numpy())
                labels.append(y.cpu().numpy())
        pred_df = pd.DataFrame(np.concatenate(outputs), columns=ls_tr)
        obs_df = pd.DataFrame(np.concatenate(labels), columns=ls_tr)
        
        if(save):
            pred_df.to_csv(path + '_Preds.csv')
            obs_df.to_csv(path + '_Obs.csv')
        return eval_metrics(obs_df, pred_df)
    
    # path_val = os.path.join(checkpoint_dir, 'DS{}_val_gp{}_Trans'.format(seed, gp))
    eval_scores[gp] = evaluate(valid_loader)
    
    path_val = os.path.join(checkpoint_dir, 'DS{}_ext_gp{}_Trans'.format(seed, gp))
    ext_scores[gp] = evaluate(ext_loader, save=True, path=path_val)
    ext_scores[gp].to_csv(path_val + '_metrics.csv')

    del test_reg
    gc.collect()
    torch.cuda.empty_cache()