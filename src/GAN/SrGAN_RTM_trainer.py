"""
Regression semi-supervised GAN Trainer.
This is a mpdified version from the method SR_GAN adopted from : https://github.com/golmschenk/sr-gan
"""
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from GAN.Gan_models import *
from GAN.utils_gans import *
from utils_data import *
from transformation_utils import *

from rtm_torch.Resources.PROSAIL.call_model import *

from tqdm import tqdm
import json
import wandb 

import numpy as np
from scipy.stats import norm

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.optim import Adam, AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class Settings:
    """Represents the settings for a given run of SRGAN."""
    def __init__(self):
        self.checkpoint_dir = None

        self.train_loader = None
        self.valid_loader = None
        self.unlabeled_loader = None
        self.scaler_model = None
        
        self.n_lb = 8
        self.input_shape = 1720
        self.type = 'full'
        self.latent_dim = 100
        
        self.learning_rate = 5e-5
        self.weight_decay = 1e-4

        self.n_epochs = None
        self.batch_size = None
        
        self.summary_step_period = 2000

        self.rtm_D = False
        self.rtm_G = False

        self.lambda_fk = 1e0
        self.lambda_un = 1e1

        self.labeled_loss_multiplier = 1e0
        self.matching_loss_multiplier = 1e0
        self.contrasting_loss_multiplier = 1e0

        self.gradient_penalty_on = True
        self.gradient_penalty_multiplier = 1e1
        self.srgan_loss_multiplier = 1e0

        self.early_stop = True
        self.early_stopping = None
        self.patience = 10
        self.logger = None
        self.log_epoch = 10
        
        self.mean_offset = 0

        self.normalize_fake_loss = False
        self.normalize_feature_norm = False
        self.contrasting_distance_function = nn.CosineEmbeddingLoss() #abs_plus_one_sqrt_mean_neg
        self.matching_distance_function = nn.CosineEmbeddingLoss() #abs_mean
        self.labeled_loss_function = HuberCustomLoss(threshold=1.0)

        self.generator_training_step_period = 5
        self.scheduler_step_period = 50
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_from_dict(self, settings_dict):
        for key, value in settings_dict.items():
            # Update attribute if it exists in the class
            if hasattr(self, key):
                setattr(self, key, value)



class SrGAN_RTM():
    """A class to manage an experimental trial."""
    def __init__(self, settings: Settings):
        self.settings = settings
        
        self.train_dataset: Dataset = None
        self.train_dataset_loader: DataLoader = None
        self.unlabeled_dataset: Dataset = None
        self.unlabeled_dataset_loader: DataLoader = None
        self.validation_dataset: Dataset = None
        
        self.D: Module = None
        self.d_optimizer: Optimizer = None
        self.G: Module = None
        self.g_optimizer: Optimizer = None
        self.signal_quit = False
        self.starting_step = 0

        self.labeled_features = None
        self.unlabeled_features = None
        self.fake_features = None
        self.interpolates_features = None
        self.gradient_norm = None

        self.glob_gen_loss = None
        self.glob_disc_loss = None

        self.labeled_loss = 0
        self.unlabeled_loss = 0
        self.fake_loss = 0
        self.gradient_penalty = 0
        self.loss_rtm_D = 0
        self.loss_rtm_G = 0
        self.loss_disc = 0
        self.generator_loss = 0
        self.gen_loss = 0.
        
        self.i_model: Module = None
        self.scaler_list = None
        self.scaler_list = None
        
        self.rtm_D = False
        self.rtm_G = False

        self.lambda_fk = 1
        self.lambda_un = 10
        
        self.transformation_layer_inv = None
        self.transformation_layer = None

    def train(self):
        self.dataset_setup()
        self.model_setup()
        self.prepare_optimizers(self.settings.n_epochs) ##
        self.gpu_mode()
        self.train_mode()
        self.transformation_setup()
        self.early_stopping_setup()
        
        self.train_loop(epoch_start=1, n_epochs=self.settings.n_epochs)
        
        
    def dataset_setup(self):
        self.train_dataset_loader = self.settings.train_loader
        self.valid_loader = self.settings.valid_loader
        self.unlabeled_dataset_loader = self.settings.unlabeled_loader
    
    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        # if(self.settings.type is not 'full'): 
        if(self.settings.type != 'full'):
            self.D = Discriminator_half(self.settings.input_shape, self.settings.n_lb) 
        else:
            self.D = Discriminator(self.settings.input_shape, self.settings.n_lb) 
        self.G = Generator(self.settings.latent_dim, self.settings.input_shape)

    def prepare_optimizers(self, num_epochs):
        """Prepares the optimizers of the network."""
        g_lr = self.settings.learning_rate
        d_lr = 4 * self.settings.learning_rate
        
        weight_decay = self.settings.weight_decay
        
        # 🔥 **Adam Optimizer with Weight Decay**
        self.d_optimizer = AdamW(self.D.parameters(), lr=d_lr, weight_decay=weight_decay, amsgrad=True)
        self.g_optimizer = AdamW(self.G.parameters(), lr=g_lr)
    
        # **📌 Define Warmup Scheduler**
        self.warmup_epochs = int(0.1 * num_epochs)  # 10% of total epochs
        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(
            self.d_optimizer, 
            lambda epoch: min(1.0, epoch / self.warmup_epochs)  # Linear warmup
        )
    
        # **📉 Define Cosine Annealing Scheduler**
        self.decay_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, T_max=num_epochs - self.warmup_epochs, eta_min=1e-5  # Min LR for stability
        )
    
        # Initialize an attribute to track scheduler mode
        self.scheduler_mode = "warmup"  # Start with warmup

    def transformation_setup(self):
        if(self.settings.scaler_model is not None):    
            scaling_layer_ = scaler_layer(self.settings.scaler_model)
        
            self.transformation_layer_inv = StaticTransformationLayer(transformation= scaling_layer_.inverse).to(self.settings.device)
            self.transformation_layer = StaticTransformationLayer(transformation=scaling_layer_).to(self.settings.device)
        else:
            self.transformation_layer_inv = None
            self.transformation_layer = None

    def early_stopping_setup(self):
        if(self.settings.early_stop):
            # Early stopping criteria
            os.makedirs(self.settings.checkpoint_dir, exist_ok=True)
            self.early_stopping = EarlyStopping(patience= self.settings.patience, verbose=True, path=self.settings.checkpoint_dir)
            
    def train_loop(self, epoch_start=1,  n_epochs=200):
        start_time = time.time()
        
        for epoch in range(epoch_start, n_epochs+1):
            torch.cuda.empty_cache()
            
            tr_labeled_loss=0.0
            ts_labeled_loss=0.0
            r2_ts = 0.0
            r2_tr = 0.0
        
            tr_unlabeled_loss = 0.0
            tr_fake_loss = 0.0
            tr_gradient_penalty = 0.0
            tr_loss_rtm_D = 0.0
            tr_loss_rtm_G = 0.0
            tr_generator_loss = 0.0
            tr_gen_loss = 0.0
        
            sup_train_iterator = iter(self.train_dataset_loader) #train_dataset_loader test.train_dataset_loader
            unsup_train_iterator = iter(self.unlabeled_dataset_loader) #unlabeled_dataset_loader test.unlabeled_dataset_loader
        
            for batch_idx, unlabeled_examples in enumerate(tqdm(unsup_train_iterator, total=len(self.unlabeled_dataset_loader), desc=f'Training epoch {epoch}')):
                step = batch_idx +1

                try:
                    # Attempt to fetch the next batch from the labeled dataset iterator
                    labeled_examples, labels, _ = next(sup_train_iterator)
                except StopIteration:
                    # If the labeled dataset iterator is exhausted, reset it
                    sup_train_iterator = iter(self.train_dataset_loader)
                    labeled_examples, labels, _ = next(sup_train_iterator)
                
                if(labeled_examples.size(0)<unlabeled_examples.size(0)):
                    # If the labeled dataset iterator is exhausted, reset it
                    del sup_train_iterator
                    sup_train_iterator = iter(self.train_dataset_loader)
                    labeled_examples, labels, _ = next(sup_train_iterator)
                    

                if(self.settings.type != 'full'):
                    unlabeled_examples = unlabeled_examples.unsqueeze(dim=1)[:,:,:self.settings.input_shape].float().to(gpu)  
                    labeled_examples = labeled_examples.unsqueeze(dim=1)[:unlabeled_examples.size(0),:,:self.settings.input_shape].float().to(gpu) 
                else:
                    unlabeled_examples = unlabeled_examples.unsqueeze(dim=1)[:,:,:-1].float().to(gpu)  
                    labeled_examples = labeled_examples.unsqueeze(dim=1)[:unlabeled_examples.size(0),:,:-1].float().to(gpu) 
                
                labels = labels[:unlabeled_examples.size(0),:].float().to(gpu)
                
                if(self.transformation_layer is not None):  
                    labels = self.transformation_layer(labels)
        
                ######
                self.labeled_loss = 0
                self.unlabeled_loss = 0
                self.fake_loss = 0
                self.gradient_penalty = 0
                self.loss_rtm_D = 0
                self.loss_rtm_G = 0
                self.loss_disc = 0

                self.gen_loss = 0.
                self.generator_loss = 0

                self.gan_training_step(labeled_examples, labels, unlabeled_examples, step)
                
                tr_labeled_loss+= self.labeled_loss
                r2_tr += r_squared(labels.detach(), self.D(labeled_examples)[0].detach())
                
                tr_unlabeled_loss+= self.unlabeled_loss
                tr_fake_loss+= self.fake_loss
                tr_gradient_penalty+= self.gradient_penalty
                tr_loss_rtm_D+= self.loss_rtm_D
                tr_loss_rtm_G+= self.loss_rtm_G
                tr_generator_loss+= self.generator_loss
                tr_gen_loss+= self.gen_loss

            ts_labeled_loss, r2_ts = self.eval_model()

            num_tot_tr = len(self.unlabeled_dataset_loader) 
            num_tot_val = len(self.valid_loader)  
            
            tr_labeled_loss/= num_tot_tr
            # ts_labeled_loss/= num_tot_val
            
            # r2_ts/= num_tot_val
            r2_tr/= num_tot_tr
            
            tr_unlabeled_loss/= num_tot_tr
            tr_fake_loss/= num_tot_tr
            tr_gradient_penalty/= num_tot_tr
            tr_loss_rtm_D/= num_tot_tr
            tr_loss_rtm_G/= num_tot_tr
            tr_generator_loss/= num_tot_tr
            tr_gen_loss/= num_tot_tr

            # **📌 Step the correct scheduler**
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()  # Step warmup
            else:
                if self.scheduler_mode == "warmup":  
                    print(f"🔥 Switching from Warmup to Cosine Annealing at epoch {epoch}")
                    self.scheduler_mode = "cosine"  # Change mode
                self.decay_scheduler.step()  # Step cosine annealing
            
            print('[%d] tr_loss: %.3f val_loss: %.3f tr_r2 score: %.3f val_r2 score: %.3f' %
                  (epoch, tr_labeled_loss, ts_labeled_loss, r2_tr, r2_ts))
            
            if(self.settings.logger is not None):
                self.settings.logger.log({'step':step, 'tr_lab_Dloss': tr_labeled_loss, 'val_lab_Dloss': ts_labeled_loss, 'tr_r2 score': r2_tr, 'val_r2 score': r2_ts, 'learning_rate_D': self.d_optimizer.param_groups[0]['lr'], 'learning_rate_G': self.g_optimizer.param_groups[0]['lr'], 'tr_unlabeled_Dloss':tr_unlabeled_loss , 'tr_fake_Dloss': tr_fake_loss, 'tr_gradient_penalty' : tr_gradient_penalty, 'tr_RTM_loss_D': tr_loss_rtm_D, 'tr_RTM_loss_G': tr_loss_rtm_G, 'tr_loss_G':tr_generator_loss})

            if(self.early_stopping is not None):
                # Check early stopping criteria
                self.early_stopping(ts_labeled_loss, self.D)
                
                if  self.early_stopping.best_score == -1*ts_labeled_loss:
                    checkpoint_path = os.path.join(self.early_stopping.path, f"best_model_G.h5")
                    torch.save(self.G, checkpoint_path)
                    # self.early_stopping(ts_labeled_loss, self.D)
        
            # if self.early_stopping.early_stop:
            #     print("Early stopping")
            #     break
        
        print('Finished Training')
        if(self.settings.logger is not None):
            self.settings.logger.finish()
        
        tr_time = (time.time() - start_time)
        print("--- Training time %s seconds ---" %tr_time )

        
    
    def eval_model(self):
        ts_labeled_loss = 0.
        r2_ts = 0.
        with torch.no_grad():
            for val_examples, val_labels, _ in self.valid_loader:
                if(self.settings.type != 'full'):
                    val_examples = val_examples.unsqueeze(dim=1)[:,:,:self.settings.input_shape].float().to(self.settings.device) #[:,:,:-1] Old Spectra Loader!!
                else:
                    val_examples = val_examples.unsqueeze(dim=1)[:,:,:-1].float().to(self.settings.device) #[:,:,:-1] Old Spectra Loader!!
                    
                val_labels = val_labels.float().to(self.settings.device)
                
                if(self.transformation_layer is not None):  
                    val_labels = self.transformation_layer(val_labels)
            
                self.eval_mode()
                ts_labeled_loss += self.labeled_loss_calculation(val_examples, val_labels)
                r2_ts += r_squared(val_labels.detach(), self.D(val_examples)[0].detach())
                
            num_tot_val = len(self.valid_loader)  
            
            ts_labeled_loss/= num_tot_val
            r2_ts/= num_tot_val
        return ts_labeled_loss, r2_ts
            

    def train_mode(self):
        """
        Converts the networks to train mode.
        """
        self.D.train()
        self.G.train()

    def gpu_mode(self):
        """
        Moves the networks to the GPU (if available).
        """
        self.D.to(gpu)
        self.G.to(gpu)

    def eval_mode(self):
        """
        Changes the network to evaluation mode.
        """
        self.D.eval()
        self.G.eval()

    def cpu_mode(self):
        """
        Moves the networks to the CPU.
        """
        self.D.to('cpu')
        self.G.to('cpu')

    def optimizer_to_gpu(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    def disable_batch_norm_updates(module):
        """Turns off updating of batch norm statistics."""
        # noinspection PyProtectedMember
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    
    def gan_training_step(self, labeled_examples, labels, unlabeled_examples, step):
        """Runs an individual round of GAN training."""
        for p in self.D.parameters():
            p.requires_grad = True
        
        self.d_optimizer.zero_grad()
        
        # Labeled.
        self.labeled_loss = self.labeled_loss_calculation(labeled_examples, labels)
        
        # Unlabeled.
        self.unlabeled_loss = self.unlabeled_loss_calculation(labeled_examples, unlabeled_examples)
        
        # Fake.
        # Generate spectra from generator 
        z = torch.tensor(MixtureModel([norm(-self.settings.mean_offset, 1),
                                       norm(self.settings.mean_offset, 1)]
                                      ).rvs(size=[unlabeled_examples.size(0),
                                                  self.G.latent_dim]).astype(np.float32)).to(gpu)
        fake_examples = self.G(z)
        self.fake_loss = self.fake_loss_calculation(unlabeled_examples, fake_examples)
        
        # Gradient penalty.
        self.gradient_penalty = self.gradient_penalty_calculation(fake_examples, unlabeled_examples)
        

        if(self.rtm_D):
            self.loss_rtm_D = self.RTM_loss_calculation(unlabeled_examples)
            # self.loss_rtm_D.backward()
        else:
            self.loss_rtm_D = 0
        
        self.loss_disc = self.labeled_loss + self.lambda_un*self.unlabeled_loss + self.gradient_penalty + self.lambda_fk*self.fake_loss + self.loss_rtm_D
        self.loss_disc.backward()

        # Clip gradients by norm
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
        
        # Discriminator update.
        self.d_optimizer.step()
        
        # Generator.
        if step % self.settings.generator_training_step_period == 0:
            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation
            
            self.g_optimizer.zero_grad()
            
            z = torch.tensor(MixtureModel([norm(-self.settings.mean_offset, 1),
                                           norm(self.settings.mean_offset, 1)]
                                          ).rvs(size=[unlabeled_examples.size(0),
                                                      self.G.latent_dim]).astype(np.float32)).to(gpu)
            fake_examples = self.G(z)
            self.generator_loss = self.generator_loss_calculation(fake_examples, unlabeled_examples)

            if(self.rtm_G):
                self.loss_rtm_G = self.RTM_loss_calculation(unlabeled_examples, fake_examples=fake_examples)
            else:
                self.loss_rtm_G = 0
            
            self.gen_loss = self.generator_loss+self.loss_rtm_G
            self.gen_loss.backward()

            # Clip gradients by norm
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
            
            self.g_optimizer.step()
            

    def labeled_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predicted_labels = self.D(labeled_examples)[0]
        
        labeled_loss = self.settings.labeled_loss_function(predicted_labels, labels) 
        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss

    def unlabeled_loss_calculation(self, labeled_examples: Tensor, unlabeled_examples: Tensor):
        """Calculates the unlabeled loss."""
        labeled_features = self.D(labeled_examples)[1]
        unlabeled_features = self.D(unlabeled_examples)[1]
                
        x1_flat = labeled_features.view(labeled_features.size(0), -1)  # Flatten to [256, 128 * 215]
        x2_flat = unlabeled_features.view(unlabeled_features.size(0), -1)  # Flatten to [256, 128 * 215]
        target = torch.ones(unlabeled_features.size(0)).to(unlabeled_features.device)
        
        unlabeled_loss = self.settings.matching_distance_function(x1_flat, x2_flat, target)
        
        unlabeled_loss *= self.settings.matching_loss_multiplier
        unlabeled_loss *= self.settings.srgan_loss_multiplier
        return unlabeled_loss


    def fake_loss_calculation(self, unlabeled_examples: Tensor, fake_examples: Tensor):
        """Calculates the fake loss."""
        unlabeled_features = self.D(unlabeled_examples)[1]
        fake_features = self.D(fake_examples)[1] #.detach()
                
        x1_flat = unlabeled_features.view(unlabeled_features.size(0), -1)  # Flatten to [256, 128 * 215]
        x2_flat = fake_features.view(unlabeled_features.size(0), -1)  # Flatten to [256, 128 * 215]

        target = torch.neg(torch.ones(unlabeled_features.size(0))).to(unlabeled_features.device)
        fake_loss = self.settings.contrasting_distance_function(x1_flat, x2_flat, target)
        
        fake_loss *= self.settings.contrasting_loss_multiplier
        fake_loss *= self.settings.srgan_loss_multiplier
        return fake_loss


    def gradient_penalty_calculation(self, fake_examples: Tensor, unlabeled_examples: Tensor) -> Tensor:
        """Calculates the gradient penalty from the given fake and real examples."""
        gpu = fake_examples.device  # Assuming both fake_examples and unlabeled_examples have the same device
        # Generate random alpha values
        alpha_shape = [1] * len(unlabeled_examples.size())
        alpha_shape[0] = unlabeled_examples.size(0)
        alpha = torch.rand(alpha_shape, device=gpu)

        interpolates = (alpha * unlabeled_examples.clone().requires_grad_() +
                (1 - alpha) * fake_examples.clone().requires_grad_())
        
        # Calculate loss for interpolates
        interpolates_loss = self.interpolate_loss_calculation(interpolates)
        
        gradients = torch.autograd.grad(outputs=interpolates_loss, inputs=interpolates,
                                        grad_outputs=torch.ones_like(interpolates_loss, device=gpu),
                                        create_graph=True, retain_graph=True)[0]
        
        # Flatten gradients and calculate the norm for each example in the batch
        gradient_norm = gradients.view(unlabeled_examples.size(0), -1).norm(2,dim=1)
        self.gradient_norm = gradient_norm.mean()  # Store the mean gradient norm as an attribute

        # Compute the gradient penalty
        norm_excesses = torch.clamp(gradient_norm - 1, min=0)
        gradient_penalty = (norm_excesses ** 2).mean() * self.settings.gradient_penalty_multiplier
        return gradient_penalty

    def interpolate_loss_calculation(self, interpolates):
        """Calculates the interpolate loss for use in the gradient penalty."""
        interpolates_features = self.D(interpolates)[1]
        return interpolates_features.norm(dim=1)

    def generator_loss_calculation(self, fake_examples, unlabeled_examples):
        """Calculates the generator's loss."""
        fake_features = self.D(fake_examples)[1]
        detached_unlabeled_features = self.D(unlabeled_examples)[1]
        
        x1_flat = detached_unlabeled_features.view(detached_unlabeled_features.size(0), -1)  # Flatten to [256, 128 * 215]
        x2_flat = fake_features.view(fake_features.size(0), -1)  # Flatten to [256, 128 * 215]

        target = torch.ones(detached_unlabeled_features.size(0)).to(detached_unlabeled_features.device)
        generator_loss = self.settings.matching_distance_function(x1_flat, x2_flat, target)
        
        generator_loss *= self.settings.matching_loss_multiplier
        return generator_loss


    def evaluate(self):
        """Evaluates the model on the test dataset (needs to be overridden by subclass)."""
        self.model_setup()
        self.load_models()
        self.eval_mode()


    def RTM_simulation(self, unlabeled_examples):
        # Generate spectra from RTM 
        if(self.settings.scaler_model is not None):
            preds_D = self.transformation_layer_inv(self.D(unlabeled_examples)[0]) ### shoud keep the sam eorder of labels !!!
        elif(self.settings.scaler_list is not None):
            preds_D = torch.tensor(self.settings.scaler_list.inverse_transform(self.D(unlabeled_examples)[0].cpu().detach().numpy()), dtype=torch.float32).requires_grad_(True)
        else:
            preds_D = self.D(unlabeled_examples)[0]
        
        rtm_paras = json.load(open('./rtm_torch/rtm_paras.json')) ###
        num_samples = unlabeled_examples.size(0)
        
        para_dict = para_sampling(rtm_paras, num_samples=num_samples)
        
        ######## From predictionsof discrim #######
        # ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
        
        para_dict['cab'] = (preds_D[:,0]) # ug/cm2
        para_dict['cw'] =(preds_D[:,1]/1000) # cm
        para_dict['cm'] = (preds_D[:,2]/10000) # g/cm2 if prospect D
        para_dict['LAI'] = (preds_D[:,3])# m2/m2
        para_dict['cp'] = (preds_D[:,4])#
        para_dict['cbc'] = (preds_D[:,5])#
        para_dict['car'] = (preds_D[:,6]) # ug/cm2
        para_dict['anth']= (preds_D[:,7]) # ug/cm2
        
        ######## Fixed parameters ######
        para_dict['cbrown']=0.25* torch.ones(num_samples)
        
        ####### Fixed for canopy spectra ###
        # typeLIDF: Leaf Angle Distribution (LIDF) type: 1 = Beta, 2 = Ellipsoidal
        # if typeLIDF = 2, LIDF is set to between 0 and 90 as Leaf Angle to calculate the Ellipsoidal distribution # degrees
        # if typeLIDF = 1, LIDF is set between 0 and 5 as index of one of the six Beta distributions
        para_dict["typeLIDF"] = 2 * torch.ones(num_samples) # 2
        # LIDF: Leaf Angle (LIDF), only used when LIDF is Ellipsoidal
        para_dict["LIDF"] = 30   * torch.ones(num_samples) # 30 
        # hspot: Hot Spot Size Parameter (Hspot)
        para_dict["hspot"] = 0.01  * torch.ones(num_samples) # unitless
        # tto: Observation zenith angle (Tto)
        para_dict["tto"] = 0  * torch.ones(num_samples) # degrees
        # tts: Sun zenith angle (Tts)
        para_dict["tts"] = 45 * torch.ones(num_samples) # degrees
        # psi: Relative azimuth angle (Psi)
        para_dict["psi"] = 0 * torch.ones(num_samples) # degrees
        
        para_dict['psoil']= 0.8 * torch.ones(num_samples) #0.8 # %
        
        int_boost = 1
        self.i_model = CallModel(soil=None, paras=para_dict)
        
        for key, value in self.i_model.par.items():
            self.i_model.par[key] = self.i_model.par[key].to(self.i_model.device)
            
        if ('cp' in para_dict.keys()):
          self.i_model.call_prospectPro()
        else:
          self.i_model.call_prospectD()  #call_prospect5b call_prospect5 call_prospectD
        
        spectra_leaf = self.i_model.call_prospectPro()
        samples = self.i_model.call_4sail() * int_boost
        
        wv = list(['{}'.format(i) for i in range(400,2501)])
        samples = pd.DataFrame(samples.detach().cpu().numpy(), columns=wv) 
        samples_clean = feature_preparation(samples).iloc[:,:-1]
        return samples_clean

    def RTM_loss_calculation(self, unlabeled_examples, fake_examples=None):
        """Calculates the generator's loss."""
        epsilon = 1e-6
        samples_clean = self.RTM_simulation(unlabeled_examples)
    
        v0 = torch.tensor(samples_clean.values).to(gpu)
        v0 = v0.view(v0.size(0), -1)
        
        if(fake_examples is not None):
            v1 = fake_examples.squeeze(dim=1).to(gpu)
            v1 = v1.view(v1.size(0), -1)
            target = torch.neg(torch.ones(v1.size(0))).to(v1.device)
        else:
            v1 = unlabeled_examples.squeeze(dim=1).to(gpu)   
            v1 = v1.view(v1.size(0), -1)
            target = torch.ones(v1.size(0)).to(v1.device)
    
        criterion = nn.CosineEmbeddingLoss()
        
        loss_rtm = criterion(v0, v1, target)
        return loss_rtm