import sys
import os

# Set the parent directory to allow for relative imports from parent modules
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import utility modules from the project
from utils_data import *
from transformation_utils import *
from utils_all import *


from MAE.utils_mae import *
from MAE.trainer_mae import *
from MAE.MAE_1D import *
from MAE.trainer_trait import *

import torch
import torch.optim as optim
import wandb 
from tqdm import tqdm


# from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
# import numpy as np
# import random
# import pandas as pd
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import lr_scheduler


class Settings_reg:
    """
    Represents the settings/configuration for a regression run using the MAE model.
    
    Attributes are intended to be set either directly or via the update_from_dict method.
    """
    def __init__(self):
        # General training settings (to be set later)
        self.epochs = None  # Total number of epochs (to be defined)
        self.train_loader = None
        self.valid_loader = None
        self.checkpoint_dir = None
        self.batch_size = None
        self.n_lb = 8  # Number of labels
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.pretrained_model = None  # Pretrained MAE model
        self.freeze = False
        self.early_stop = True
        self.load_model_path = None
        self.early_stopping = None
        self.patience = 10
        self.logger = None
        self.scaler_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model-specific settings
        self.latent_dim = None
        self.hidden_dims = None 
        self.aggregation = "mean"  # Aggregation method for latent features
        self.normalize_latent = True  # Whether to normalize latent representations
    
    def update_from_dict(self, settings_dict):
        """
        Update settings attributes from a dictionary.
        
        Only attributes that already exist in the class will be updated.
        
        Args:
            settings_dict (dict): Dictionary containing new settings.
        """
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Trainer_MAE_Reg():
    """
    Trainer class for performing regression using a pretrained MAE model.
    
    Manages dataset setup, model initialization, optimizer preparation,
    training, validation, and logging.
    """
    def __init__(self, settings: Settings_reg):
        self.settings = settings
        
        # Data loaders for training and validation
        self.train_loader: DataLoader = None
        self.valid_loader: Dataset = None
        
        self.scaler_model = None
        
        # Models and optimizers
        self.pretrained_model: Module = None
        self.model: Module = None
        self.optimizer: Optimizer = None
        self.criterion: Module = None

        # Transformation layers (for scaling / inverse scaling)
        self.transformation_layer_inv: Module = None
        self.transformation_layer: Module = None
    
        # Lists for tracking training and validation loss over epochs
        self.train_loss_list = []
        self.valid_loss_list = []  
    
    def dataset_setup(self):
        """
        Setup the data loaders from the settings.
        """
        self.train_loader = self.settings.train_loader
        self.valid_loader = self.settings.valid_loader
    
    def model_setup(self):
        """
        Setup and instantiate the regression model using the pretrained MAE model.
        
        It extracts the latent dimension from a specific block of the pretrained model,
        and then instantiates a LatentRegressionModel with the given configurations.
        Also, sets up the transformation and criterion.
        """
        self.pretrained_model = self.settings.pretrained_model
        # Extract latent dimension from the pretrained model's 4th block's MLP output
        self.settings.latent_dim = self.pretrained_model.blocks[3].mlp.fc2.out_features
        
        self.model = LatentRegressionModel(
            pretrained_encoder=self.pretrained_model,
            latent_dim=self.settings.latent_dim,
            output_dim=self.settings.n_lb,
            hidden_dims=self.settings.hidden_dims,  # Can be None or set to add dense layers
            aggregation=self.settings.aggregation,
            normalize_latent=self.settings.normalize_latent,  # Enable latent normalization if desired
        )
        
        self.transformation_setup()
        self.criterion_setup()
            
    def prepare_optimizers(self):
        """
        Prepare the optimizer (AdamW) for training the model.
        """
        lr = self.settings.learning_rate
        weight_decay = self.settings.weight_decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            amsgrad=True
        )
    
    def gpu_mode(self):
        """
        Move the model to the GPU (if available).
        """
        self.model.to(self.settings.device)
    
    def train_mode(self):
        """
        Set the model to training mode.
        """
        self.model.train()
    
    def eval_mode(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()
        
    def transformation_setup(self):
        """
        Setup transformation layers for scaling labels.
        
        If a scaler model is provided in the settings, create a transformation
        layer and its inverse using a StaticTransformationLayer.
        """
        if self.settings.scaler_model is not None:    
            scaling_layer_ = scaler_layer(self.settings.scaler_model)
        
            self.transformation_layer_inv = StaticTransformationLayer(transformation=scaling_layer_.inverse).to(self.settings.device)
            self.transformation_layer = StaticTransformationLayer(transformation=scaling_layer_).to(self.settings.device)
        else:
            self.transformation_layer_inv = None
            self.transformation_layer = None
    
    def criterion_setup(self):
        """
        Setup the loss criterion. Here, a custom Huber loss is used.
        """
        self.criterion = HuberCustomLoss(threshold=1.0)  # Alternatives: LabeledLoss() or nn.MSELoss()
    
    def early_stopping_setup(self):
        """
        Setup early stopping based on validation loss.
        
        Creates the checkpoint directory if it doesn't exist and instantiates
        an EarlyStopping object.
        """
        if self.settings.early_stop:
            os.makedirs(self.settings.checkpoint_dir, exist_ok=True)
            self.early_stopping = EarlyStopping(
                patience=self.settings.patience, 
                verbose=True, 
                path=self.settings.checkpoint_dir
            )
    
    def train_step(self, labeled_examples, labels, ds):
        """
        Execute a single training step.
        
        Args:
            labeled_examples (torch.Tensor): Input examples for training.
            labels (torch.Tensor): Ground truth labels.
            ds (torch.Tensor): Tensor containing group indices for each example.
            
        Returns:
            loss (torch.Tensor): Computed loss for the batch.
            r2_score_tr (float): R-squared score for the training batch.
        """
        self.train_mode()
        self.optimizer.zero_grad()
        
        # Preprocess inputs: squeeze and remove last column, convert to float and move to device
        labeled_examples = labeled_examples.squeeze()[:, :-1].float().to(self.settings.device)
        labels = labels.to(self.settings.device)
                    
        # Apply transformation to labels if a transformation layer is available
        if self.transformation_layer is not None:  
            labels = self.transformation_layer(labels)
        
        # Forward pass through the model
        outputs = self.model(labeled_examples)
        
        # Count occurrences for each group and compute sample weights
        group_counts = torch.bincount(ds)
        group_frequencies = group_counts[ds]
        w = 1 - (group_frequencies / ds.size(0))
        
        # Compute the loss with sample weights
        loss = self.criterion(outputs, labels, sample_weight=w)
        r2_score_tr = r_squared(labels, outputs).item()
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss, r2_score_tr
    
    def val_step(self, labeled_examples, labels, ds):
        """
        Execute a single validation step.
        
        Args:
            labeled_examples (torch.Tensor): Input examples for validation.
            labels (torch.Tensor): Ground truth labels.
            ds (torch.Tensor): Tensor containing group indices for each example.
            
        Returns:
            loss_val (torch.Tensor): Computed validation loss for the batch.
            r2_val (float): R-squared score for the validation batch.
        """
        self.eval_mode()
        
        with torch.no_grad():
            # Preprocess inputs similarly as in train_step
            labeled_examples = labeled_examples.squeeze()[:, :-1].float().to(self.settings.device)
            labels = labels.float().to(self.settings.device)
            
            if self.transformation_layer is not None:  
                labels = self.transformation_layer(labels)
            
            outputs = self.model(labeled_examples)
            
            # Compute sample weights for validation loss
            group_counts = torch.bincount(ds)
            group_frequencies = group_counts[ds]
            w = 1 - (group_frequencies / ds.size(0))
            
            loss_val = self.criterion(outputs, labels, sample_weight=w)
            r2_val = r_squared(labels, outputs).item()
            
        return loss_val, r2_val

    def train_loop(self, epoch_start, num_epochs):
        """
        Run the full training loop over the specified number of epochs.
        
        Logs metrics using wandb if a logger is provided, and checks for early stopping.
        
        Args:
            epoch_start (int): Starting epoch number.
            num_epochs (int): Total number of epochs.
        """
        # Initialize wandb logging if a logger is provided
        if self.settings.logger is not None:
            wandb.init(
                project=self.settings.project,  # Project name for wandb
                name=f"experiment_{self.settings.run}",
            )
    
        # Main training loop over epochs
        for epoch in range(epoch_start, num_epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0
            valid_r2 = 0.0
            r2_tr_epoch = 0.0
            
            # Training phase over batches
            for labeled_examples, labels, ds in tqdm(self.train_loader):
                loss, r2_score_tr = self.train_step(labeled_examples, labels, ds)
                r2_tr_epoch += r2_score_tr
                train_loss += loss
                
            # Validation phase over batches
            for labeled_examples, labels, ds in tqdm(self.valid_loader):
                loss_val, r2_val = self.val_step(labeled_examples, labels, ds)
                valid_loss += loss_val
                valid_r2 += r2_val
            
            # Compute average losses and R2 scores for the epoch
            train_loss = train_loss / len(self.train_loader)
            valid_loss = valid_loss / len(self.valid_loader)
            valid_r2 /= len(self.valid_loader)
            r2_tr_epoch /= len(self.train_loader)
            
            # Record losses for later analysis
            self.train_loss_list.append(train_loss)
            self.valid_loss_list.append(valid_loss)
                
            # Print training statistics for the current epoch
            print('epoch: {} Training Loss: {:.6f} \t Validation Loss: {:.6f}, tr_r2: {:.6f}, val_r2: {:.6f} '
                  .format(epoch, train_loss, valid_loss, r2_tr_epoch, valid_r2))
            
            # Log metrics using wandb if logger is provided
            if self.settings.logger is not None:
                wandb.log({
                    'epoch': epoch, 
                    'tr_loss': train_loss, 
                    'valid_loss': valid_loss, 
                    'tr_r2': r2_tr_epoch, 
                    'val_r2': valid_r2, 
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Check early stopping criteria if enabled
            if self.settings.early_stop:
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
            if self.settings.logger is not None:
                wandb.finish()
