import sys
import os

######## the file looks at the parent directory ###
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)


from utils_data import *
from transformation_utils import *
from utils_all import *
from Multi_trait.multi_model import *


import torch
import torch.optim as optim
import wandb 
from tqdm import tqdm

class Settings:
    """Represents the settings for a given run of transformer."""
    def __init__(self):
        self.epochs = None  # Should be set later
        self.train_loader = None
        self.valid_loader = None
        self.checkpoint_dir = None
        self.batch_size = None
        self.n_lb = 8
        self.learning_rate = None
        self.weight_decay = 0
        self.early_stop = True
        self.load_model_path = None
        self.should_save_models = True
        self.skip_completed_experiment = True
        self.early_stopping = None
        self.patience = 10
        self.logger = None
        self.scaler_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_from_dict(self, settings_dict):
        for key, value in settings_dict.items():
            # Update attribute if it exists in the class
            if hasattr(self, key):
                setattr(self, key, value)


class Trainer_MultiTraits():
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # self.train_dataset: Dataset = None
        self.train_loader: DataLoader = None
        self.valid_loader: Dataset = None
        self.scaler_model = None
        
        self.pretrained_model: Module = None
        self.model: Module = None
        self.optimizer: Optimizer = None
        self.criterion: Module = None

        self.transformation_layer_inv: Module = None
        self.transformation_layer: Module = None
    
        self.train_loss_list = []
        self.valid_loss_list = []  

    def train(self):
        self.dataset_setup()
        self.model_setup()
        self.prepare_optimizers()
        self.gpu_mode()
        self.early_stopping_setup()
        self.train_loop(epoch_start=1, num_epochs=self.settings.epochs)
    
    def dataset_setup(self):
        self.train_loader = self.settings.train_loader
        self.valid_loader = self.settings.valid_loader
    
    def model_setup(self):
        self.model = EfficientNetB0(num_classes=self.settings.n_lb)
        self.transformation_setup()
        self.criterion_setup()
            
    
    def prepare_optimizers(self):
        """Prepares the optimizers of the network."""
        lr = self.settings.learning_rate
        
        weight_decay = self.settings.weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    
    def gpu_mode(self):
        """
        Moves the networks to the GPU (if available).
        """
        self.model.to(self.settings.device)
    
    def train_mode(self):
        """
        Converts the networks to train mode.
        """
        self.model.train()
    
    def eval_mode(self):
        """
        Changes the network to evaluation mode.
        """
        self.model.eval()
        
    def transformation_setup(self):
    
        if(self.settings.scaler_model is not None):    
            scaling_layer_ = scaler_layer(self.settings.scaler_model)
        
            self.transformation_layer_inv = StaticTransformationLayer(transformation= scaling_layer_.inverse).to(self.settings.device)
            self.transformation_layer = StaticTransformationLayer(transformation=scaling_layer_).to(self.settings.device)
        else:
            self.transformation_layer_inv = None
            self.transformation_layer = None
    
    def criterion_setup(self):
        self.criterion = HuberCustomLoss(threshold=1.0) #LabeledLoss() #nn.MSELoss()
    
    def early_stopping_setup(self):
        if(self.settings.early_stop):
            # Early stopping criteria
            os.makedirs(self.settings.checkpoint_dir, exist_ok=True)
            self.early_stopping = EarlyStopping(patience= self.settings.patience, verbose=True, path=self.settings.checkpoint_dir)
    
    def train_step(self, labeled_examples, labels, ds):
        self.train_mode()
        self.optimizer.zero_grad()
        
        labeled_examples = labeled_examples.unsqueeze(dim=1).to(self.settings.device)
        labels = labels.to(self.settings.device)#.float()
                    
        if(self.transformation_layer is not None):  
            labels = self.transformation_layer(labels)
        
        outputs = self.model(labeled_examples)
        
        # Count occurrences of each group
        group_counts = torch.bincount(ds)
        
        # Create a tensor of the same size as `groups`, where each element is the count of its corresponding group
        group_frequencies = group_counts[ds]
        
        w = 1-(group_frequencies/ds.size(0))
        
        loss = self.criterion(outputs, labels, sample_weight=w)
        r2_score_tr = r_squared(labels, outputs).item()
        
        # # # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients by norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss, r2_score_tr
    

    def eval_run(self):
        self.eval_mode()
        
        with torch.no_grad():
            loss_val = 0
            r2_score_val=0
            
            for labeled_examples, labels, ds in self.valid_loader:
                
                labeled_examples = labeled_examples.unsqueeze(dim=1).to(self.settings.device)
                
                if(self.transformation_layer is not None):  
                    labels = self.transformation_layer(labels)
    
                labels = labels.to(self.settings.device).squeeze()
                
                y_pred_sc = self.model(labeled_examples)
                
                # Count occurrences of each group
                group_counts = torch.bincount(ds)
                
                # Create a tensor of the same size as `groups`, where each element is the count of its corresponding group
                group_frequencies = group_counts[ds]
        
                w = 1-(group_frequencies/ds.size(0))
                
                loss = self.criterion(labels, y_pred_sc, sample_weight=w)
                
                loss_val += loss.item()
                r2_score_val += r_squared(labels, y_pred_sc)
    
    
            loss_val /=  len(self.valid_loader)
            r2_score_val /=  len(self.valid_loader)
            return loss_val, r2_score_val

    def train_loop(self, epoch_start, num_epochs):
        
        for epoch in range(epoch_start, num_epochs + 1):
            loss_epoch = 0
            r2_score_epoch = 0
            loss_val = 0
            r2_score_val = 0
            
            train_progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{num_epochs}')
            for labeled_examples, labels, ds in train_progress_bar:
                
                loss, r2_score_tr = self.train_step(labeled_examples, labels, ds)
                
                loss_epoch += loss.item()
                r2_score_epoch += r2_score_tr
                
                train_progress_bar.set_postfix({'loss_tr ': loss_epoch / len(self.train_loader), 'r2_tr ': r2_score_epoch / len(self.train_loader)})
                
                metrics_per_step = {'train_step/train_loss': loss.item()}
                metrics_per_step.update({'train_step/train_r2': r2_score_tr})
                metrics_per_step.update({'train_step/lr': self.optimizer.param_groups[0]['lr']}) #optimizer.param_groups[0]['lr'] scheduler.get_last_lr()[0]
                
                if(self.settings.logger is not None):
                    wandb.log(metrics_per_step) #, step=batch_idx+1
        
            loss_epoch /=  len(self.train_loader)
            r2_score_epoch /=  len(self.train_loader)
            loss_val, r2_score_val =  self.eval_run()

            
            if(self.settings.logger is not None):  
                metrics_per_epoch = {'train_epoch/train_loss': loss_epoch}
                metrics_per_epoch = {'train_epoch/epoch': epoch}
                metrics_per_epoch.update({'train_epoch/train_r2': r2_score_epoch})
                metrics_per_epoch.update({'train_epoch/val_loss': loss_val})
                metrics_per_epoch.update({'train_epoch/val_r2': r2_score_val})
                wandb.log(metrics_per_epoch)
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {loss_epoch:.4f}, Val Loss: {loss_val:.4f}, Train r2_score: {r2_score_epoch:.4f}, Val r2_score: {r2_score_val:.4f}')
        
        
            # Check early stopping criteria
            self.early_stopping(loss_val, self.model)
        
            # if self.early_stopping.early_stop:
            #     print("Early stopping")
            #     break    
            
            if(self.settings.logger is not None):
                wandb.log({'epoch': epoch, 'tr_loss': loss_epoch, 'valid_loss': loss_val, 'tr_r2': r2_score_epoch, 'val_r2': r2_score_val, 'lr': self.optimizer.param_groups[0]['lr']})