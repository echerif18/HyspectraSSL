"""
MAE training

This file implements the training pipeline for the Masked Autoencoder (MAE) model.
It sets up the datasets, model, optimizer, learning rate schedulers, and training loop.
"""
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings (e.g., ZeroDivision warnings)

# Set up the parent directory in the system path (allows for relative imports)
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import MAE model and data utilities from local modules
from MAE.MAE_1D import *
from utils_data import *

import random
import numpy as np
from torch.nn import Module
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# from scipy.stats import norm
# from torch import Tensor
# from torch.utils.data.sampler import SubsetRandomSampler


# Define the device: use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # e.g., 'cuda', 'cpu'


class Settings:
    """
    Represents the settings (hyperparameters and configurations) for a given run.
    """
    def __init__(self):
        self.seed = 155
        self.starting_epoch = 1
        self.epochs = 500
        self.checkpoint_dir = None
        self.file_paths = None
        self.batch_size = 256
        self.valid_size = 0.2
        self.augmentation = True
        self.scale = False
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.early_stop = True
        self.early_stopping = None
        self.patience = 10

        self.mask_ratio = 0.75
        self.w_loss = 0.

        # Model-specific settings
        self.type = 'full'
        self.n_bands = 1720
        self.seq_size = 20
        self.in_chans = 1
        self.embed_dim = 128
        self.depth = 4
        self.num_heads = 4
        self.decoder_embed_dim = 128
        self.decoder_depth = 4
        self.decoder_num_heads = 4
        self.mlp_ratio = 4.
        self.norm_layer = nn.LayerNorm
        self.cls_token = False

        self.device = device
        self.logger = None

    def update_from_dict(self, settings_dict):
        """
        Update settings attributes from a dictionary.
        Only updates attributes that already exist in the class.
        """
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


def seed_all(seed=None):
    """
    Seed all the random number generators for reproducibility.
    If seed is None, the current time is used to generate a seed.
    """
    if seed is None:
        seed = int(time.time())

    # Seed Python's random module
    random.seed(seed)
    # Seed NumPy
    np.random.seed(seed)
    # Seed PyTorch (for CPU)
    torch.manual_seed(seed)
    # Seed PyTorch for CUDA (if available)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer():
    """
    A class to manage the training process of the MAE model.
    """
    def __init__(self, settings: Settings):
        self.settings = settings

        # Data loaders for training and validation datasets
        self.train_loader: DataLoader = None
        self.valid_loader: Dataset = None

        # Model and optimizer placeholders
        self.model: Module = None
        self.optimizer: Optimizer = None
        self.starting_epoch = 1

        # Lists to track training and validation loss over epochs
        self.train_loss_list = []
        self.valid_loss_list = []

        self.early_stopping = None

    def train(self):
        """
        Run the complete training process.
        This includes dataset and model setup, optimizer preparation,
        switching to the correct device, and the training loop.
        """
        # Seed all random number generators for reproducibility
        seed_all(self.settings.seed)

        # Prepare datasets and data loaders
        self.dataset_setup()

        # Set up the model architecture
        self.model_setup()

        # Prepare optimizers and learning rate schedulers
        self.prepare_optimizers(self.settings.epochs) ###
        # Move model to GPU if available and set to train mode
        self.gpu_mode()
        self.train_mode()

        # Setup early stopping if enabled
        self.early_stopping_setup()
        # Run the main training loop
        self.training_loop(epoch_start=self.settings.starting_epoch, num_epochs=self.settings.epochs)

    def prepare_optimizers(self, num_epochs):
        """
        Prepare the optimizer and learning rate schedulers (warmup and cosine annealing).
        """
        lr = self.settings.learning_rate
        weight_decay = self.settings.weight_decay

        # Use AdamW optimizer with weight decay and AMSGrad
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            amsgrad=True
        )

        # # Define warmup scheduler: warm up for 10% of total epochs
        # self.warmup_epochs = int(0.1 * num_epochs)
        # self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lambda epoch: min(1.0, epoch / self.warmup_epochs)  # Linear warmup
        # )

        # # Define cosine annealing scheduler for LR decay after warmup
        # self.decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=num_epochs - self.warmup_epochs, eta_min=1e-6
        # )

        # # Track the current scheduler mode (start with warmup)
        # self.scheduler_mode = "warmup"

    def train_mode(self):
        """
        Set the model to training mode.
        """
        self.model.train()
        
        for param in self.model.seq_embed.proj.parameters(): ### non-trainable positinal embeddings
            param.requires_grad = False

    def gpu_mode(self):
        """
        Move the model to the GPU (if available).
        """
        self.model.to(self.settings.device)

    def eval_mode(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()

    def cpu_mode(self):
        """
        Move the model to the CPU.
        """
        self.model.to('cpu')

    def optimizer_to_gpu(self, optimizer):
        """
        Move optimizer state tensors to the GPU.
        """
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    def dataset_setup(self):
        """
        Prepare training and validation datasets and corresponding data loaders.
        """
        # Create training dataset using all file paths except the last one
        dataset = MultiFileAugmentedCSVDataset(
            self.settings.file_paths[:-1], 
            chunk_size=1000, 
            augmentation=self.settings.augmentation, 
            aug_prob=0.6, 
            scale=self.settings.scale
        )
        self.train_loader = DataLoader(
            dataset, 
            batch_size=self.settings.batch_size, 
            shuffle=True
        )

        # Create validation dataset using the last file path
        dataset = MultiFileAugmentedCSVDataset(
            self.settings.file_paths[-1:], 
            chunk_size=1000, 
            augmentation=False, 
            aug_prob=0., 
            scale=self.settings.scale
        )
        self.valid_loader = DataLoader(
            dataset, 
            batch_size=self.settings.batch_size, 
            shuffle=False
        )

    def model_setup(self):
        """
        Initialize the MAE model with the specified hyperparameters.
        """
        self.model = MaskedAutoencoder(
            n_bands=self.settings.n_bands, 
            seq_size=self.settings.seq_size, 
            in_chans=self.settings.in_chans,
            embed_dim=self.settings.embed_dim, 
            depth=self.settings.depth, 
            num_heads=self.settings.num_heads,
            decoder_embed_dim=self.settings.decoder_embed_dim, 
            decoder_depth=self.settings.decoder_depth, 
            decoder_num_heads=self.settings.decoder_num_heads,
            mlp_ratio=self.settings.mlp_ratio, 
            norm_layer=self.settings.norm_layer, 
            cls_token=self.settings.cls_token
        )

    def early_stopping_setup(self):
        """
        Set up early stopping criteria if enabled.
        """
        if self.settings.early_stop:
            # Create checkpoint directory if it doesn't exist
            os.makedirs(self.settings.checkpoint_dir, exist_ok=True)
            self.early_stopping = EarlyStopping(
                patience=self.settings.patience, 
                verbose=True, 
                path=self.settings.checkpoint_dir
            )

    def train_step(self, batch):
        """
        Execute a single training step.
        
        Args:
            batch: A batch of training data.

        Returns:
            z: The latent representation of the input batch.
            loss.item(): The loss value for this batch.
        """
        # Set model to train mode and zero gradients
        self.train_mode()
        self.optimizer.zero_grad()

        x = batch
        # Reshape input to (batch_size, n_bands) and move to correct device
        x = x.to(self.settings.device).view(x.shape[0], x.shape[-1])

        # Forward pass through the model with the specified mask ratio
        loss, pred, mask, z = self.model(x, mask_ratio=self.settings.mask_ratio, w_loss=self.settings.w_loss) ###self.w_loss

        # Process latent representation based on whether a cls token is used
        if self.model.is_cls_token:
            z = z[:, 0, :]
        else:
            z = torch.mean(z[:, 1:, :], dim=1)

        # Backpropagate the loss and update model parameters
        loss.backward()
        self.optimizer.step()
        return z.detach().cpu(), loss.item()

    def val_step(self, batch):
        """
        Execute a single validation step.
        
        Args:
            batch: A batch of validation data.

        Returns:
            z: The latent representation of the input batch.
            loss.item(): The loss value for this batch.
        """
        # Set model to evaluation mode
        self.eval_mode()

        x = batch
        x = x.to(self.settings.device).view(x.shape[0], x.shape[-1])

        with torch.no_grad():
            loss, pred, mask, z = self.model(x, mask_ratio=self.settings.mask_ratio)

        if self.model.is_cls_token:
            z = z[:, 0, :]
        else:
            z = torch.mean(z[:, 1:, :], dim=1)

        return z.cpu(), loss.item()

    def training_loop(self, epoch_start, num_epochs):
        """
        Runs the main training loop over the specified number of epochs.

        Args:
            epoch_start: The starting epoch number.
            num_epochs: Total number of epochs for training.
        """
        # Record starting time for the training process
        curr = time.process_time()

        # Loop over epochs
        for e in range(epoch_start, num_epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            # Training phase: iterate over training batches
            for images in tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc=f'Training epoch {e}'
            ):
                # Remove last column if needed and move images to device
                if(self.settings.type != 'full'):
                    images = images.to(self.settings.device)[:,:self.settings.n_bands] ####half_range
                else: 
                    images = images.to(self.settings.device)[:, :-1]
                    
                z, loss = self.train_step(images)
                train_loss += loss

                # Log step metrics if a logger is provided
                if self.settings.logger is not None:
                    metrics_per_step = {'train_step/train_loss': loss}
                    self.settings.logger.log(metrics_per_step)

            # Validation phase: iterate over validation batches
            for images in self.valid_loader:
                if(self.settings.type != 'full'):
                    images = images.to(self.settings.device)[:,:self.settings.n_bands] ####half_range
                else: 
                    images = images.to(self.settings.device)[:, :-1]
                    
                z, loss_val = self.val_step(images) ##images[:, :-1]
                valid_loss += loss_val

            # Compute average losses for the epoch
            train_loss = train_loss / len(self.train_loader)
            valid_loss = valid_loss / len(self.valid_loader)

            self.train_loss_list.append(train_loss)
            self.valid_loss_list.append(valid_loss)

            # Log epoch metrics if a logger is provided
            if self.settings.logger is not None:
                metrics_per_epoch = {
                    'train_epoch/train_loss': train_loss,
                    'train_epoch/epoch': e,
                    'train_epoch/lr': self.optimizer.param_groups[0]['lr'],
                    'train_epoch/val_loss': valid_loss
                }
                self.settings.logger.log(metrics_per_epoch)

            # Print training statistics for the current epoch
            print('Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(train_loss, valid_loss))

            # Check for early stopping criteria if enabled
            if self.early_stopping is not None:
                self.early_stopping(valid_loss, self.model)

        # Calculate and print the total training time
        end = time.process_time() - curr
        print('The training process took {}s >> {}h'.format(end, end / 3600))