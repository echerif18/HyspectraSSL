import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import os

import random


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


class HuberCustomLoss(nn.Module):
    def __init__(self, threshold=1.0):
        """
        Custom Huber Loss with threshold. This loss transitions from mean squared error
        to mean absolute error depending on the error magnitude relative to the threshold.
        
        Args:
            threshold (float): The point at which the loss transitions from MSE to MAE.
        """
        super(HuberCustomLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y_true, y_pred, sample_weight=None):
        """
        Computes the Huber loss between `y_true` and `y_pred`.
        
        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.
            sample_weight (torch.Tensor, optional): Weights for each sample. Defaults to None.
        
        Returns:
            torch.Tensor: The computed Huber loss.
        """
        # Ensure y_true and y_pred are on the same device
        y_true = y_true.to(y_pred.device)
        
        # Filter out non-finite values (infinite or NaN) in y_true
        bool_finite_out = torch.isfinite(y_pred)
        bool_finite_lb = torch.isfinite(y_true)
        
        finite_mask = bool_finite_out & bool_finite_lb
        
        # Calculate the error (residual)
        error = y_pred[finite_mask] - y_true[finite_mask]
        
        # Compute the squared loss and the linear loss
        abs_error = torch.abs(error)
        squared_loss = 0.5 * error**2
        linear_loss = self.threshold * abs_error - 0.5 * self.threshold**2
        
        # Determine where the error is "small" or "large"
        is_small_error = abs_error < self.threshold
        
        # Compute the final loss (use the squared loss for small errors, linear loss for large errors)
        loss = torch.where(is_small_error, squared_loss, linear_loss)

        # If sample weights are provided, apply them
        if sample_weight is not None:
            # Broadcast the weights to the correct shape
            # sample_weight = sample_weight.to(y_pred.device)  # Ensure same device
            sample_weights = torch.stack([sample_weight for i in range(y_true.size(1))], dim=1).to(y_pred.device)
            loss = loss * sample_weights[finite_mask]
            
            # Return the mean loss
            return loss.sum()

        # Return the mean loss
        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target):
        # Compute cosine similarity
        cosine_sim = self.cos_sim(pred, target)
        # Minimize the negative cosine similarity
        loss = 1-torch.mean(cosine_sim)
        return loss

class LabeledLoss(nn.Module):
    def __init__(self):
        super(LabeledLoss, self).__init__()
        self.huber_loss = HuberCustomLoss(threshold=1.0)

    def forward(self, lb_pred, y_train_lb, sample_weight=None):
        return self.huber_loss(y_train_lb, lb_pred, sample_weight=sample_weight)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def mae_loss(output, target):
    return F.mae_loss(output, target)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path to save the model when validation loss improves.
                            Default: 'checkpoint.pt'
            trace_func (function): Function to print logging messages.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        checkpoint_path = os.path.join(self.path, f"model_epoch_{-1*self.best_score}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
        checkpoint_path = os.path.join(self.path, f"best_model.h5")
        torch.save(model, checkpoint_path)
        
        self.val_loss_min = val_loss



########### metrics ###
def r_squared(y_true, y_pred):
    bool_finite = torch.isfinite(y_true) & torch.isfinite(y_pred)  # Ensure finite values
    y_true = y_true[bool_finite]
    y_pred = y_pred[bool_finite]

    if y_true.numel() == 0:  # Edge case: No valid data points
        return torch.tensor(float('nan')).to(y_true.device)

    y_mean = torch.mean(y_true)
    total_var = torch.sum((y_true - y_mean) ** 2)
    
    if total_var == 0:  # Avoid division by zero
        return torch.tensor(0.0).to(y_true.device)
    
    residual_var = torch.sum((y_true - y_pred) ** 2)
    return 1 - (residual_var / total_var)


####### calculate the evaluation score of the regression ##
def eval_metrics(ori_lb, df_tr_val):
    from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
    import math
    from scipy import stats
    
    def calculate_rpd(y_true, y_pred):
        sd = np.std(y_true, ddof=1)  # Standard deviation of observations
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # RMSE
        rpd = sd / rmse
        return rpd
    
    r2_tab = []
    RMSE_tab = []
    nrmse_tab = []
    mae_tab = []
    b_tab = []
    rpd_tab= []
    cc_tab = []
    
    samp_w_ts = None
    
    pred = df_tr_val.copy()
    obs_pf = ori_lb.copy()
    
    test_tr = list(ori_lb.columns)
    
    for j in test_tr:
    
        f = pred.iloc[:,test_tr.index(j)].reset_index(drop=True) # + ' Predictions'
        y = obs_pf.iloc[:,test_tr.index(j)].reset_index(drop=True)
    
        idx = np.union1d(f[f.isna()].index,y[y.isna()].index)
    
        f.drop(idx, axis = 0, inplace=True)
        y.drop(idx, axis = 0, inplace=True)
    
        if (y.notnull().sum()):
            if (samp_w_ts is not None):
                we = pd.DataFrame(samp_w_ts).loc[f.index,:]
            else:
                we = None
    
            if (we is not None) and (we.sum().sum() !=0):
                r2_tab.append(r2_score(y,f,sample_weight= we))
    
                RMSE=math.sqrt(mean_squared_error(y,f,sample_weight= we))
                RMSE_tab.append(RMSE)
                nrmse_tab.append((RMSE*100)/(np.nanquantile(np.array(y),0.99) - np.nanquantile(np.array(y),0.01)))
    
                mae_tab.append(mean_absolute_error(y,f,sample_weight= we))
    
                bias=np.sum(np.array(y)-np.array(f))/len(f)
                b_tab.append(bias)

                rpd_value = calculate_rpd(y, f)
                rpd_tab.append(rpd_value)

                cc = stats.spearmanr(y,f)
                cc_tab.append((cc.statistic)**2)
                
            else:
                r2_tab.append(r2_score(y,f))
    
                RMSE=math.sqrt(mean_squared_error(y,f))
                RMSE_tab.append(RMSE)
                nrmse_tab.append((RMSE*100)/(np.nanquantile(np.array(y),0.99) - np.nanquantile(np.array(y),0.01)))
    
                mae_tab.append(mean_absolute_error(y,f))
    
                bias=np.sum(np.array(y)-np.array(f))/len(f)
                b_tab.append(bias)
                
                rpd_value = calculate_rpd(y, f)
                rpd_tab.append(rpd_value)

                cc = stats.spearmanr(y,f)
                cc_tab.append((cc.statistic)**2)
        else:
            r2_tab.append(np.nan)
            RMSE_tab.append(np.nan)
            nrmse_tab.append(np.nan)
            mae_tab.append(np.nan)
            b_tab.append(np.nan)
            rpd_tab.append(np.nan)
            cc_tab.append(np.nan)
            pass        
    
    # test_tab = pd.DataFrame([r2_tab, RMSE_tab, nrmse_tab,mae_tab,b_tab], columns= test_tr[:len(test_tr)], index=['r2_score','RMSE','nRMSE (%)','MAE','Bias'])
    test_tab = pd.DataFrame([r2_tab, RMSE_tab, nrmse_tab,mae_tab,b_tab, rpd_tab, cc_tab], columns= test_tr[:len(test_tr)], index=['r2_score','RMSE','nRMSE (%)','MAE','Bias', 'RPD', 'spearmanr_squared'])
    
    return test_tab.T