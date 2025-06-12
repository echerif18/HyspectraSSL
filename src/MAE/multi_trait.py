# Consistent Downstream MAE Regression Script
import os
import sys

# Add project root to path for module resolution
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from src.transformation_utils import *
from src.utils_all import *
from src.utils_data import *
from src.MAE.utils_mae import *
from src.MAE.trainer_mae import *
from src.MAE.MAE_1D import *
from src.MAE.trainer_trait import *


import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



#########################################
# Helper Function: half_latent
#########################################
def half_latent(x, model):
    """
    Extract latent features using a half masking strategy.

    Steps:
        1. Embed patches using the model's sequence embedding.
        2. Add positional embeddings (excluding the cls token if present).
        3. Apply half masking to the embedded patches.
        4. Pass through the Transformer blocks.
        5. Normalize the resulting features.
    
    Args:
        x (torch.Tensor): Input spectral data.
        model (nn.Module): MAE model containing the embedding, masking, and Transformer blocks.
    
    Returns:
        torch.Tensor: The latent representation.
    """
    # Embed patches from the input
    x = model.seq_embed(x)
    
    # Add positional embeddings without the cls token
    x = x + model.pos_embed[:, np.sum(model.is_cls_token):, :]
    
    # Apply half masking to the embedded patches
    x, mask, ids_restore = model.half_masking(x)
    
    # Pass through the Transformer blocks sequentially
    for blk in model.blocks:
        x = blk(x)
    # Normalize the output latent features
    z = model.norm(x)
    return z

#########################################
# LatentRegressionModel Definition
#########################################
class LatentRegressionModel(nn.Module):
    """
    Regression model that leverages a pretrained MAE encoder to extract latent features,
    then applies an aggregation strategy and a regression head to predict target values.

    Args:
        pretrained_encoder (nn.Module): Pretrained MAE encoder.
        latent_dim (int): Dimension of the latent vector.
        output_dim (int): Dimension of the output (e.g., number of regression targets).
        input_dim (int, optional): Dimension of the input spectra. Default is 1720.
        type_sp (str, optional): Strategy type ('full' for full extraction or otherwise for half masking). Default is 'full'.
        hidden_dims (list of int, optional): Sizes of hidden layers for dense regression head. If None, use a single layer.
        aggregation (str, optional): Aggregation strategy ('none', 'mean', or 'custom').
        normalize_latent (bool, optional): Whether to apply LayerNorm to the latent space.
        freeze_encoder (bool, optional): Whether to freeze the encoder during training.
    """
    def __init__(self, pretrained_encoder, latent_dim, output_dim, input_dim=1720, type_sp='full',
                 hidden_dims=None, aggregation="none", normalize_latent=False, freeze_encoder=True):
        super().__init__()
        self.encoder = pretrained_encoder
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.type_sp = type_sp
        self.aggregation = aggregation
        self.normalize_latent = normalize_latent

        # Optional normalization layer for latent features.
        if self.normalize_latent:
            self.normalization = nn.LayerNorm(self.latent_dim)
        else:
            self.normalization = None

        # Calculate the number of sequences (patches) from the input.
        seq_size = (self.input_dim // self.encoder.seq_embed.seq_size)
        
        # Determine the input dimension for the regression head based on the aggregation method.
        input_dim = {
            "none": seq_size * latent_dim,
            "mean": latent_dim,
            "custom": 3 * latent_dim
        }.get(aggregation, seq_size * latent_dim)

        # Build the regression head with optional hidden layers.
        self.regression_head = self._build_regression_head(input_dim, hidden_dims, output_dim)

    def _build_regression_head(self, input_dim, hidden_dims, output_dim):
        """
        Constructs a regression head composed of dense layers and activation functions.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int or None): List of hidden layer sizes.
            output_dim (int): Dimension of the output layer.
        
        Returns:
            nn.Sequential: A sequential model representing the regression head.
        """
        layers = []
        if hidden_dims is not None:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dim))
                input_dim = hidden_dim
        # Final output layer
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass to extract latent representations and perform regression.

        Depending on the type_sp parameter, either the full encoder or half masking strategy is applied.
        Then, an aggregation strategy is applied to the latent features before passing them
        through the regression head.

        Args:
            x (torch.Tensor): Input spectral data.
        
        Returns:
            torch.Tensor: Regression predictions.
        """
        # Extract latent representations using full or half masking strategy.
        if self.type_sp == 'full':
            latent, _, _ = self.encoder.forward_encoder(x, mask_ratio=0)  # No masking during regression
        else:
            latent = half_latent(x, self.encoder)
        
        # Normalize the latent features if enabled.
        if self.normalize_latent:
            latent = self.normalization(latent)

        # Apply aggregation strategy:
        if self.aggregation == "none":  # Use full sequence flattening
            latent = latent.flatten(start_dim=1)
        elif self.aggregation == "mean":  # Mean pooling across the sequence dimension
            latent = latent.mean(dim=1)
        elif self.aggregation == "custom":  # Custom aggregation (e.g., VIS/NIR/SWIR concatenation)
            # Example for custom aggregation with three segments:
            vis = torch.mean(latent[:, :10, :], dim=1)
            nir = torch.mean(latent[:, 10:48, :], dim=1)
            swir = torch.mean(latent[:, 48:, :], dim=1)
            
            latent = torch.cat([vis, nir, swir], dim=1)
            latent = latent.flatten(start_dim=1)

        # Pass the aggregated latent features through the regression head.
        return self.regression_head(latent)


#########################################
# Downstream Regression Pipeline
#########################################
# def load_model(checkpoint_dir_mae, type_sp='full', n_bands=1720, seq_size=20,d=10, h=16,mask_ratio=0.75 ):
#     # Load pretrained MAE model
#     path_model_mae = os.path.join(checkpoint_dir_mae, 'best_model.h5')

#     settings_dict_mae = {
#         'learning_rate': 1e-3,
#         'weight_decay': 0,
#         'load_model_path': path_model_mae, #path_model_mae load_model_path
#         'should_save_models': True,
#         'skip_completed_experiment': True,
#         'number_of_data_workers': 0,
#         'w_loss' : 1,
        
#         'mask_ratio':mask_ratio,
#         'n_bands': n_bands,
#         'type':type_sp,
#         'seq_size': seq_size,
#         'in_chans': 1,
#         'embed_dim': 128,
#         'depth': d,
#         'num_heads': h,
#         'decoder_embed_dim': 128,
#         'decoder_depth': 6,
#         'decoder_num_heads': 4,
#         'mlp_ratio': 4.0,
#         'norm_layer': nn.LayerNorm,
#         'cls_token': False,
#         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     }

#     sets = Settings()
#     sets.update_from_dict(settings_dict_mae)
#     trainer = Trainer(sets)
#     trainer.model_setup()
#     trainer.model = torch.load(path_model_mae)
#     return trainer

def load_model(type_sp='full', n_bands=1720, seq_size=20,d=10, h=16,mask_ratio=0.75, checkpoint_dir_mae=None, HF=False, repo_id=None, model_id=None):
    if(HF):
        path_model_mae = hf_hub_download(
            repo_id=repo_id,
            filename=os.path.join(model_id, 'best_model.h5'))
    else:
        # Load pretrained MAE model
        path_model_mae = os.path.join(checkpoint_dir_mae, 'best_model.h5')

    settings_dict_mae = {
        'learning_rate': 1e-3,
        'weight_decay': 0,
        'load_model_path': path_model_mae, #path_model_mae load_model_path
        'should_save_models': True,
        'skip_completed_experiment': True,
        'number_of_data_workers': 0,
        'w_loss' : 1,
        
        'mask_ratio':mask_ratio,
        'n_bands': n_bands,
        'type':type_sp,
        'seq_size': seq_size,
        'in_chans': 1,
        'embed_dim': 128,
        'depth': d,
        'num_heads': h,
        'decoder_embed_dim': 128,
        'decoder_depth': 6,
        'decoder_num_heads': 4,
        'mlp_ratio': 4.0,
        'norm_layer': nn.LayerNorm,
        'cls_token': False,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    sets = Settings()
    sets.update_from_dict(settings_dict_mae)
    trainer = Trainer(sets)
    trainer.model_setup()
    trainer.model = torch.load(path_model_mae)
    return trainer


# def run_consistent_experiment(checkpoint_dir_mae, path_data_lb, seeds=[155, 381, 187], fine_tune=False, n_epochs = 200, percentage_tr=1, type_sp='full', n_bands=1720,  save=False, name=''):
#     ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
#     batch_size = 256

#     # # Load pretrained MAE model
#     trainer = load_model(checkpoint_dir_mae, type_sp='full', n_bands=1720, seq_size=20,d=10, h=16,mask_ratio=0.75)
#     pretrained_model = trainer.model

def run_consistent_experiment(path_data_lb, seeds=[155, 381, 187], fine_tune=False, n_epochs = 200, percentage_tr=1, type_sp='full', n_bands=1720,  save=False, name='',checkpoint_dir_mae=None, HF=False, repo_id=None, model_id=None):
    ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
    batch_size = 256
    
    if(HF):
        trainer = load_model(type_sp='full', n_bands=1720, seq_size=20,d=10, h=16,mask_ratio=0.75, HF=True, repo_id=repo_id, model_id=model_id)
    else:
        # # Load pretrained MAE model
        trainer = load_model(type_sp='full', n_bands=1720, seq_size=20,d=10, h=16,mask_ratio=0.75, checkpoint_dir_mae=checkpoint_dir_mae)
    pretrained_model = trainer.model

    eval_scores = {}

    for SEED in seeds:
        set_seed(SEED)

        db_lb_all = pd.read_csv(path_data_lb).drop(['Unnamed: 0'], axis=1)

        groups = db_lb_all.groupby('dataset')
        val_ext_idx = list(groups.get_group(32).index) + list(groups.get_group(3).index) + list(groups.get_group(50).index)
        db_lb_all.drop(val_ext_idx, inplace=True)

        X_labeled, y_labeled = data_prep_db(db_lb_all, ls_tr)
        metadata = db_lb_all.iloc[:, :1]

        red_ed = X_labeled.loc[:,750]
        red_end = X_labeled.loc[:,1300]
        red1000_ = X_labeled.loc[:,1000]
        
        idx = X_labeled[(red_end>red1000_) & (red_ed>red1000_)].index
        
        if(len(idx)>0):
            X_labeled.drop(idx, inplace=True)
            y_labeled.drop(idx, inplace=True)
            metadata.drop(idx, inplace=True)

        fr_sup, val_x = train_test_split(X_labeled, test_size=0.2, stratify=metadata.dataset, random_state=300)

        if(n_bands != 1720):
            fr_sup = fr_sup.loc[:,400:900] ##### half range
            val_x = val_x.loc[:,400:900] ##### half range

        y_sup = y_labeled.loc[fr_sup.index, :]
        val_y = y_labeled.loc[val_x.index, :]
            
        meta_train = metadata.loc[fr_sup.index, :]
        meta_val = metadata.loc[val_x.index, :]

        if percentage_tr < 1:
            fr_sup, _ = train_test_split(fr_sup, test_size=1 - percentage_tr, stratify=meta_train.dataset, random_state=300)
            y_sup = y_sup.loc[fr_sup.index, :]
            meta_train = meta_train.loc[fr_sup.index, :]

        scaler_model = save_scaler(y_sup, standardize=True, scale=True)

        train_dataset = SpectraDataset(fr_sup, y_sup, meta_train, augmentation=True, aug_prob=0.6)
        test_dataset = SpectraDataset(val_x, val_y, meta_val, augmentation=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        settings_dict = {
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'batch_size': batch_size,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'pretrained_model': pretrained_model,
            'early_stop': False,
            'patience': 10,
            'scaler_model': scaler_model,
        }

        sets = Settings_reg()
        sets.update_from_dict(settings_dict)
        test_reg = Trainer_MAE_Reg(sets)
        test_reg.dataset_setup()

        latent_dim = pretrained_model.blocks[3].mlp.fc2.out_features
        model = LatentRegressionModel(
            pretrained_encoder=pretrained_model,
            latent_dim=latent_dim,
            output_dim=len(ls_tr),
            hidden_dims=None,
            aggregation="mean",
            normalize_latent=True,
            type_sp=type_sp
        )

        for param in model.encoder.parameters():
            param.requires_grad = fine_tune

        test_reg.model = model
        test_reg.transformation_setup()
        test_reg.criterion_setup()
        test_reg.prepare_optimizers()
        test_reg.gpu_mode()

        test_reg.train_loop(epoch_start=1, num_epochs=n_epochs)

        def evaluate(loader, save=False, path=None):
            outputs, labels = [], []
            for batch in tqdm(loader):
                x, y = batch[:2]
                x = x.to(sets.device)[:, :-1]
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
        
        path_val = os.path.join(checkpoint_dir_mae, 'DS{}_FT{}_val'.format(SEED, fine_tune))
        eval_scores[SEED] = evaluate(valid_loader, save=save, path=path_val)

        del test_reg
        gc.collect()
        torch.cuda.empty_cache()

    # Concatenate all DataFrames along a new axis and calculate mean and std
    all_dfs = pd.concat(eval_scores.values(), keys=eval_scores.keys())  #eval_scores ext_scores

    mean_metrics = all_dfs.groupby(level=1, sort=False).mean()
    std_metrics = all_dfs.groupby(level=1, sort=False).std(ddof=0) ## STDS ddof=1

    if(save):
        mean_metrics.to_csv(path_val + '_eval_scores_avg_val_{}.csv'.format(name))
        std_metrics.to_csv(path_val + '_eval_scores_std_val_{}.csv'.format(name))

    return mean_metrics, std_metrics



from itertools import islice

def run_consistent_experimentCV(checkpoint_dir_mae, path_data_lb, seeds=[155, 381, 187], fine_tune=False, n_epochs = 200, percentage_tr=1, type_sp='full', n_bands=1720,  save=False,  start=0, end=None, name=''):
    
    ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
    batch_size = 256

    # Load pretrained MAE model
    trainer = load_model(checkpoint_dir_mae, type_sp='full', n_bands=1720, seq_size=20,d=10, h=16,mask_ratio=0.75)
    pretrained_model = trainer.model

    eval_scores = {}
    ext_scores = {}

    SEED = seeds[0]
    set_seed(SEED)

    db_lb = pd.read_csv(path_data_lb).drop(['Unnamed: 0'], axis=1)


    for gp, (db_lb_all, samples_val_ext, test_ids) in islice(enumerate(sliding_custom_cv(db_lb, seed=42)), start, end):
    # for gp, (db_lb_all, samples_val_ext, test_ids) in islice(enumerate(sliding_custom_dataset(db_lb, seed=42)), start, end):

        ext_all = samples_val_ext.copy()
        
        if(n_bands != 1720):
            ext_val_x = feature_preparation(ext_all.loc[:, '400':'2500']).loc[:, 400:900]
        else:
            ext_val_x = feature_preparation(ext_all.loc[:, '400':'2500']).loc[:, 400:2450]
        ext_val_y = ext_all[ls_tr]

        x_p_val = torch.tensor(ext_val_x.values, dtype=torch.float)
        lb_p_val = torch.tensor(ext_val_y.values, dtype=torch.float)

        X_labeled, y_labeled = data_prep_db(db_lb_all, ls_tr)
        metadata = db_lb_all.iloc[:, :1]

        red_ed = X_labeled.loc[:,750]
        red_end = X_labeled.loc[:,1300]
        red1000_ = X_labeled.loc[:,1000]
        
        idx = X_labeled[(red_end>red1000_) & (red_ed>red1000_)].index
        
        if(len(idx)>0):
            X_labeled.drop(idx, inplace=True)
            y_labeled.drop(idx, inplace=True)
            metadata.drop(idx, inplace=True)

        fr_sup, val_x = train_test_split(X_labeled, test_size=0.2, stratify=metadata.dataset, random_state=300)

        if(n_bands != 1720):
            fr_sup = fr_sup.loc[:,400:900] ##### half range
            val_x = val_x.loc[:,400:900] ##### half range

        y_sup = y_labeled.loc[fr_sup.index, :]
        val_y = y_labeled.loc[val_x.index, :]
            
        meta_train = metadata.loc[fr_sup.index, :]
        meta_val = metadata.loc[val_x.index, :]

        if percentage_tr < 1:
            fr_sup, _ = train_test_split(fr_sup, test_size=1 - percentage_tr, stratify=meta_train.dataset, random_state=300)
            y_sup = y_sup.loc[fr_sup.index, :]
            meta_train = meta_train.loc[fr_sup.index, :]

        scaler_model = save_scaler(y_sup, standardize=True, scale=True)

        train_dataset = SpectraDataset(fr_sup, y_sup, meta_train, augmentation=True, aug_prob=0.6)
        test_dataset = SpectraDataset(val_x, val_y, meta_val, augmentation=False)
        ext_dataset = TensorDataset(x_p_val, lb_p_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        ext_loader = DataLoader(ext_dataset, batch_size=batch_size, shuffle=False)

        settings_dict = {
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'batch_size': batch_size,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'pretrained_model': pretrained_model,
            'early_stop': False,
            'patience': 10,
            'scaler_model': scaler_model,
        }

        sets = Settings_reg()
        sets.update_from_dict(settings_dict)
        test_reg = Trainer_MAE_Reg(sets)
        test_reg.dataset_setup()

        latent_dim = pretrained_model.blocks[3].mlp.fc2.out_features
        model = LatentRegressionModel(
            pretrained_encoder=pretrained_model,
            latent_dim=latent_dim,
            output_dim=len(ls_tr),
            hidden_dims=None,
            aggregation="mean",
            normalize_latent=True,
            type_sp=type_sp
        )

        for param in model.encoder.parameters():
            param.requires_grad = fine_tune

        test_reg.model = model
        test_reg.transformation_setup()
        test_reg.criterion_setup()
        test_reg.prepare_optimizers()
        test_reg.gpu_mode()

        test_reg.train_loop(epoch_start=1, num_epochs=n_epochs)

        def evaluate(loader, save=False, path=None):
            outputs, labels = [], []
            for batch in tqdm(loader):
                x, y = batch[:2]
                x = x.to(sets.device)[:, :-1]
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
        
        # path_val = os.path.join(checkpoint_dir_mae, 'DS{}_FT{}_val_gp{}_Trans'.format(SEED, fine_tune, gp))
        eval_scores[gp] = evaluate(valid_loader)
        # eval_scores[SEED] = evaluate(valid_loader, save=True, path=path_val)
        
        path_val = os.path.join(checkpoint_dir_mae, 'DS{}_FT{}_ext_gp{}_Trans'.format(SEED, fine_tune, gp))
        # ext_scores[SEED] = evaluate(ext_loader)
        ext_scores[gp] = evaluate(ext_loader, save=True, path=path_val)
        

        del test_reg
        gc.collect()
        torch.cuda.empty_cache()

    return eval_scores, ext_scores
