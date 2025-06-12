import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

# Set parent directory for module resolution
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Local modules
from utils_data import *
from transformation_utils import *
from utils_all import *
from AE_RTM.AE_RTM_architectures import *
from Multi_trait.multi_model import *

import wandb
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# List of trait labels
ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


class Settings_ae:
    """Represents the settings for a given run of the transformer model."""
    def __init__(self):
        self.epochs = None  # Should be set later
        self.train_loader = None
        self.valid_loader = None
        self.unlabeled_loader = None
        self.checkpoint_dir = None
        self.batch_size = None
        self.n_lb = 8
        self.input_shape = 1720
        self.type = 'full'
        self.learning_rate = None
        self.weight_decay = 1e-4

        self.early_stop = True
        self.load_model_path = None
        self.should_save_models = True
        self.skip_completed_experiment = True
        self.early_stopping = None
        self.patience = 10
        self.logger = None
        self.scaler_model = None
        self.loss_recons_criterion = CosineSimilarityLoss()  # mse_loss alternative
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lamb = 1e0

    def update_from_dict(self, settings_dict):
        """Update settings attributes from a dictionary."""
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Trainer_AE_RTM:
    def __init__(self, settings: Settings_ae):
        self.settings = settings

        # Data loaders
        self.train_loader = None  # type: DataLoader
        self.valid_loader = None  # type: Dataset

        # Model and training components
        self.model = None  # type: Module
        self.optimizer = None  # type: Optimizer
        self.criterion = None  # type: Module

        # Transformation layers
        self.transformation_layer_inv = None  # type: Module
        self.transformation_layer = None  # type: Module

        # Loss tracking
        self.train_loss_list = []
        self.valid_loss_list = []

        self.stablize_count = 0
        self.stablize_grad = True

    def train(self):
        self.dataset_setup()
        self.model_setup()
        self.prepare_optimizers(self.settings.epochs)
        self.gpu_mode()
        self.early_stopping_setup()
        self.train_loop(epoch_start=1, num_epochs=self.settings.epochs)

    def dataset_setup(self):
        """Setup training, validation, and unlabeled data loaders."""
        self.train_loader = self.settings.train_loader
        self.valid_loader = self.settings.valid_loader
        self.unlabeled_loader = self.settings.unlabeled_loader

    def model_setup(self):
        """Instantiate the model, transformation layer, and loss criterion."""
        self.model = AE_RTM_corr(self.settings.input_shape, self.settings.n_lb,
                                 scaler_list=self.settings.scaler_model)
        self.transformation_setup()
        self.criterion_setup()

    def prepare_optimizers(self, num_epochs):
        """
        Prepare optimizers with learning rate warmup and decay.
        Currently uses Adam optimizer with weight decay.
        """
        lr = self.settings.learning_rate
        weight_decay = self.settings.weight_decay

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay, amsgrad=True)

    def gpu_mode(self):
        """Move the model to GPU if available."""
        self.model.to(self.settings.device)

    def train_mode(self):
        """Set the model to training mode."""
        self.model.train()

    def eval_mode(self):
        """Set the model to evaluation mode."""
        self.model.eval()

    def transformation_setup(self):
        """Setup transformation layers if scaler_model is provided."""
        if self.settings.scaler_model is not None:
            scaling_layer_ = scaler_layer(self.settings.scaler_model)
            self.transformation_layer_inv = StaticTransformationLayer(
                transformation=scaling_layer_.inverse
            ).to(self.settings.device)
            self.transformation_layer = StaticTransformationLayer(
                transformation=scaling_layer_
            ).to(self.settings.device)
        else:
            self.transformation_layer_inv = None
            self.transformation_layer = None

    def criterion_setup(self):
        """Setup the reconstruction loss criterion."""
        self.criterion = self.settings.loss_recons_criterion

    def early_stopping_setup(self):
        """Setup early stopping criteria if enabled."""
        if self.settings.early_stop:
            os.makedirs(self.settings.checkpoint_dir, exist_ok=True)
            self.early_stopping = EarlyStopping(patience=self.settings.patience,
                                                verbose=True,
                                                path=self.settings.checkpoint_dir)

    def grad_stabilizer(self):
        """Stabilize gradients by replacing NaNs and very small values."""
        epsilon = 1e-7  # Minimum stabilizing value
        min_threshold = 1e-5  # Threshold to detect vanishing gradients

        para_grads = [v.grad for v in self.model.parameters()
                      if v.grad is not None and
                      (torch.isnan(v.grad).any() or torch.abs(v.grad).max() < min_threshold)]

        if para_grads:
            for v in para_grads:
                rand_values = torch.rand_like(v, dtype=torch.float) * epsilon
                mask_nan = torch.isnan(v)
                mask_small = torch.abs(v) < min_threshold
                v.copy_(torch.where(mask_nan, rand_values, v))
                v.copy_(torch.where(mask_small, v + epsilon, v))
            self.stablize_count += 1

    def train_step(self, samples):
        """Perform a single training step."""
        self.train_mode()
        self.optimizer.zero_grad()

        data, lb_bx = samples
        lb_bx = lb_bx.float().to(self.settings.device)

        if self.settings.scaler_model is not None:
            lb_bx = self.transformation_layer(lb_bx)

        data = data.squeeze().float().to(self.settings.device)
        x, out = self.model(data)
        output = out[:, list(range(951)) + list(range(1031, 1401)) + list(range(1651, 2051))]

        if self.settings.type is not 'full':
            output = out[:, :self.settings.input_shape]

        data = data.squeeze()[~torch.any(output.isnan(), dim=1)]
        output = output[~torch.any(output.isnan(), dim=1)]

        mae_loss_brightness = nn.L1Loss()
        loss_recos = self.criterion(output, data) + mae_loss_brightness(output, data)

        sup_idx = torch.tensor([0, 1, 2, 3, 4, 6, 7]).to(self.settings.device)
        labeled_loss_function = HuberCustomLoss(threshold=1.0)
        loss_lb = labeled_loss_function(torch.index_select(x, 1, sup_idx),
                                        torch.index_select(lb_bx, 1, sup_idx))

        recon_weight = loss_recos.item() / (loss_lb.item() + 1e-6)
        loss = (recon_weight * loss_recos) + loss_lb

        loss.backward()

        if self.stablize_grad:
            self.grad_stabilizer()

        self.optimizer.step()
        r2_step = r_squared(lb_bx.detach(), x.detach())

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm < 1e-7:
                    print(f"⚠️ Warning: Vanishing gradient detected in {name} | Norm: {grad_norm:.10f}")
                    param.grad.add_(torch.randn_like(param.grad) * 1e-6)

        return loss_recos.item(), loss_lb.item(), loss.item(), r2_step.item()

    def val_epoch(self):
        """Run validation over one epoch."""
        tot_val = 0
        r2_val = 0.
        loss_recons_val = 0.
        loss_lb_val = 0.

        preds = torch.empty(0, self.settings.n_lb).to(self.settings.device)
        ori = torch.empty(0, self.settings.n_lb).to(self.settings.device)

        self.eval_mode()

        with torch.no_grad():
            for batch_idx, val_sample in enumerate(self.valid_loader):
                data_val, lb_bx_val, ds = val_sample

                if self.settings.type is not 'full':
                    data_val = data_val.view(data_val.shape[0], data_val.shape[-1])[:, :self.settings.input_shape].to(self.settings.device)
                lb_bx_val = lb_bx_val.to(self.settings.device).float()
                data_val = data_val.squeeze().float().to(self.settings.device)

                if self.settings.scaler_model is not None:
                    lb_bx_val = self.transformation_layer(lb_bx_val)

                x_val, out_val = self.model(data_val)
                output_val = out_val.data[:, list(range(951)) + list(range(1031, 1401)) + list(range(1651, 2051))]

                if self.settings.type is not 'full':
                    output_val = output_val[:, :self.settings.input_shape]

                data_val = data_val[~torch.any(output_val.isnan(), dim=1)]
                output_val = output_val[~torch.any(output_val.isnan(), dim=1)]

                mae_loss_brightness = nn.L1Loss()
                val_loss_recos = self.criterion(output_val, data_val) + mae_loss_brightness(output_val, data_val)
                loss_recons_val += val_loss_recos

                sup_idx = torch.tensor([0, 1, 2, 3, 4, 6, 7]).to(self.settings.device)
                labeled_loss_function = HuberCustomLoss(threshold=1.0)
                val_loss_lb = labeled_loss_function(torch.index_select(x_val, 1, sup_idx),
                                                    torch.index_select(lb_bx_val, 1, sup_idx))
                loss_lb_val += val_loss_lb

                r2_val += r_squared(lb_bx_val.detach(), x_val.detach())

                recon_weight = val_loss_recos.item() / (val_loss_lb.item() + 1e-6)
                val_loss = (recon_weight * val_loss_recos) + val_loss_lb
                tot_val += val_loss

                ori = torch.cat((ori.data, lb_bx_val.data), dim=0)
                preds = torch.cat((preds.data, x_val[:, :].data), dim=0)

        val_loss_epoch = tot_val / len(self.valid_loader)
        loss_recons_val = loss_recons_val / len(self.valid_loader)
        loss_lb_val = loss_lb_val / len(self.valid_loader)
        r2_val = r2_val / len(self.valid_loader)

        if self.settings.scaler_model is not None:
            ori_lb = pd.DataFrame(self.settings.scaler_model.inverse_transform(
                ori.cpu().detach().numpy()), columns=ls_tr[:])
            df_tr_val = pd.DataFrame(self.settings.scaler_model.inverse_transform(
                preds.cpu().detach().numpy()), columns=ls_tr)
            df_tr_val['cbc'] = df_tr_val['cm'] - df_tr_val['cp']
        else:
            ori_lb = pd.DataFrame(ori.cpu(), columns=ls_tr[:])
            df_tr_val = pd.DataFrame(preds.cpu(), columns=ls_tr[:])
            df_tr_val['cbc'] = df_tr_val['cm'] - df_tr_val['cp']

        val_mertics = eval_metrics(ori_lb, df_tr_val)

        if self.settings.logger is not None:
            wandb.log({"train_epoch/val_metrics": wandb.Table(dataframe=val_mertics)})

        return loss_recons_val.item(), loss_lb_val.item(), val_loss_epoch.item(), r2_val.item()

    def train_loop(self, epoch_start, num_epochs):
        """Main training loop."""
        start_time = time.time()

        for epoch in range(epoch_start, num_epochs + 1):
            loss_epoch = 0.0
            r2_epoch = 0.
            loss_lb_epoch = 0.

            for batch_idx, samples in enumerate(
                    tqdm(infinite_iter(self.train_loader, self.unlabeled_loader),
                         total=len(self.unlabeled_loader),
                         desc=f'Training epoch {epoch}')):
                if self.settings.type is not 'full':
                    sp, lb = samples
                    sp = sp.view(sp.shape[0], sp.shape[-1])[:, :self.settings.input_shape].to(self.settings.device)
                    samples = (sp, lb)
                loss_recos, loss_lb, loss_val, r2_train_step = self.train_step(samples)
                loss_epoch += loss_val
                loss_lb_epoch += loss_lb
                r2_epoch += r2_train_step

                if self.settings.logger is not None:
                    metrics_per_step = {
                        'train_step/train_loss': loss_val,
                        'train_step/step': batch_idx + 1,
                        'train_step/train_reconsloss': loss_recos,
                        'train_step/train_lbloss': loss_lb,
                        'train_step/train_r2': r2_train_step
                    }
                    wandb.log(metrics_per_step)

            loss_epoch /= len(self.unlabeled_loader)
            loss_lb_epoch /= len(self.unlabeled_loader)
            r2_epoch /= len(self.unlabeled_loader)

            self.train_loss_list.append(loss_lb_epoch)

            if self.settings.logger is not None:
                metrics_per_epoch = {
                    'train_epoch/epoch': epoch,
                    'train_epoch/train_loss': loss_epoch,
                    'train_epoch/train_r2': r2_epoch,
                    'train_epoch/lr': self.optimizer.param_groups[0]['lr']
                }
                wandb.log(metrics_per_epoch)

            loss_recons_val, loss_lb_val, val_loss_epoch, r2_val = self.val_epoch()

            if self.settings.logger is not None:
                val_metrics_per_epoch = {
                    'train_epoch/val_loss': val_loss_epoch,
                    'train_epoch/val_reconsloss': loss_recons_val,
                    'train_epoch/val_lbloss': loss_lb_val,
                    'train_epoch/val_r2': r2_val
                }
                wandb.log(val_metrics_per_epoch)

            self.valid_loss_list.append(val_loss_epoch)

            print(f'epoch: {epoch} Training Loss: {loss_epoch:.6f} \t Validation Loss: {val_loss_epoch:.6f}, '
                  f'tr_r2: {r2_epoch:.6f}, val_r2: {r2_val:.6f}')

            if self.early_stopping is not None:
                self.early_stopping(loss_lb_val, self.model)

        if self.settings.logger is not None:
            wandb.finish()

        tr_time = time.time() - start_time
        print('Finished Training')
        print(f"--- Training time {tr_time} seconds ---")