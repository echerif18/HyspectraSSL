### util modul ########
########### These functions were adopted from : https://github.com/golmschenk/sr-gan ####

"""
General settings.
"""
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from utils_all import *

import random
from copy import deepcopy
from scipy.stats import rv_continuous
import math
import numpy as np
import torch


class MixtureModel(rv_continuous):
    """Creates a combination distribution of multiple scipy.stats model distributions."""
    def __init__(self, submodels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels

    def _pdf(self, x, **kwargs):
        pdf = self.submodels[0].pdf(x)
        for submodel in self.submodels[1:]:
            pdf += submodel.pdf(x)
        pdf /= len(self.submodels)
        return pdf

    def rvs(self, size):
        """Random variates of the mixture model."""
        submodel_choices = np.random.randint(len(self.submodels), size=size)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


def norm_squared(tensor, axis=1):
    """Calculates the norm squared along an axis. The default axis is 1 (the feature axis), with 0 being the batch."""
    return tensor.pow(2).sum(dim=axis)

def square_mean(tensor):
    """Calculates the element-wise square, then the mean of a tensor."""
    return tensor.pow(2).mean()

def abs_plus_one_log_mean_neg(tensor):
    """Takes the absolute value, then adds 1, then takes the log, then mean, then negates."""
    return tensor.abs().add(1).log().mean().neg()


def abs_mean(tensor):
    """Takes the absolute value, then mean."""
    return tensor.abs().mean()

def abs_plus_one_sqrt_mean_neg(tensor):
    """Takes the absolute value, then adds 1, then takes the log, then mean, then negates."""
    return tensor.abs().add(1).sqrt().mean().neg()



def unit_vector(vector):
    """Gets the unit vector version of a vector."""
    return vector.div(vector.norm() + 1e-10)


def angle_between(vector0, vector1):
    """Calculates the angle between two vectors."""
    unit_vector0 = unit_vector(vector0)
    unit_vector1 = unit_vector(vector1)
    epsilon = 1e-6
    return unit_vector0.dot(unit_vector1).clamp(-1.0 + epsilon, 1.0 - epsilon).acos()


def square(tensor):
    """Squares the tensor value."""
    return tensor.pow(2)


def feature_distance_loss_unmeaned(base_features, other_features, distance_function=square):
    """Calculate the loss based on the distance between feature vectors."""
    base_mean_features = base_features.mean(0, keepdim=True)
    distance_vector = distance_function(base_mean_features - other_features)
    return distance_vector.mean()


def feature_distance_loss_both_unmeaned(base_features, other_features, distance_function=norm_squared):
    """Calculate the loss based on the distance between feature vectors."""
    distance_vector = distance_function(base_features - other_features)
    return distance_vector.mean()


def feature_angle_loss(base_features, other_features, target=0, summary_writer=None):
    """Calculate the loss based on the angle between feature vectors."""
    angle = angle_between(base_features.mean(0), other_features.mean(0))
    if summary_writer:
        summary_writer.add_scalar('Feature Vector/Angle', angle.item(), )
    return (angle - target).abs().pow(2)


def feature_corrcoef(x):
    """Calculate the feature vector's correlation coefficients."""
    transposed_x = x.transpose(0, 1)
    return corrcoef(transposed_x)


def corrcoef(x):
    """Calculate the correlation coefficients."""
    mean_x = x.mean(1, keepdim=True)
    xm = x.sub(mean_x)
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


def feature_covariance_loss(base_features, other_features):
    """Calculate the loss between feature vector correlation coefficient distances."""
    base_corrcoef = feature_corrcoef(base_features)
    other_corrcoef = feature_corrcoef(other_features)
    return (base_corrcoef - other_corrcoef).abs().sum()


def disable_batch_norm_updates(module):
    """Turns off updating of batch norm statistics."""
    # noinspection PyProtectedMember
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def enable_batch_norm_updates(module):
    """Turns on updating of batch norm statistics."""
    # noinspection PyProtectedMember
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


def convert_to_settings_list(settings, shuffle=True):
    """
    Creates permutations of settings for any setting that is a list.
    (e.g. if `learning_rate = [1e-4, 1e-5]` and `batch_size = [10, 100]`, a list of 4 settings objects will return)
    This function is black magic. Beware.
    """
    settings_list = [settings]
    next_settings_list = []
    any_contains_list = True
    while any_contains_list:
        any_contains_list = False
        for settings in settings_list:
            contains_list = False
            for attribute_name, attribute_value in vars(settings).items():
                if isinstance(attribute_value, (list, tuple)):
                    for value in attribute_value:
                        settings_copy = deepcopy(settings)
                        setattr(settings_copy, attribute_name, value)
                        next_settings_list.append(settings_copy)
                    contains_list = True
                    any_contains_list = True
                    break
            if not contains_list:
                next_settings_list.append(settings)
        settings_list = next_settings_list
        next_settings_list = []
    if shuffle:
        random.seed()
        random.shuffle(settings_list)
    return settings_list



### for RTM ###
def para_sampling(rtm_paras, num_samples=100):
    pi_tensor = torch.tensor(math.pi)
    
    # run uniform sampling for learnable parameters
    para_dict = {}
    for para_name in rtm_paras.keys():
        min = rtm_paras[para_name]['min']
        max = rtm_paras[para_name]['max']
        para_dict[para_name] = torch.rand(num_samples) * (max - min) + min
    SD = 500
    para_dict['cd'] = torch.sqrt(
        (para_dict['fc']*10000)/(pi_tensor*SD))*2 #pi_tensor torch.pi
    para_dict['h'] = torch.exp(
        2.117 + 0.507*torch.log(para_dict['cd']))

    return para_dict