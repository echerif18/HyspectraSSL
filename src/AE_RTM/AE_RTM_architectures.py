import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from transformation_utils import *
from Multi_trait.multi_model import *

# #### Model definition ###
from rtm_torch.Resources.PROSAIL.call_model import *
from rtm_torch.rtm import RTM

import torch
import torch.nn as nn


###### This is an extrended version of  https://github.com/yihshe/ai-refined-rtm ###

class Discriminator(nn.Module):
    def __init__(self, input_shape, n_classes=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.batchnorm = nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear((input_shape // 8) * 128, n_classes)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.batchnorm(x)
        x = self.activation(self.conv2(x))
        x = self.batchnorm(x)
        x = self.activation(self.conv3(x))
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.dropout(x)
        d_out_layer = self.fc(x)
        return d_out_layer


class AE_RTM(nn.Module):
    """
    Vanilla AutoEncoder (AE) with RTM as the decoder
    input -> encoder (learnable) -> decoder (INFORM) -> output
    """

    def __init__(self, input_dim, hidden_dim, rtm_paras=None, lop='prospectPro', canopy_arch='sail', scaler_list=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaler_list = scaler_list  # Store scaler_list as an instance variable

        # The encoder is a learnable neural network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
        )
        
        # The decoder is the INFORM RTM with fixed parameters
        self.decoder = RTM()
        self.decoder.lop = lop
        self.decoder.canopy_arch = canopy_arch

        # Initialize RTM parameters
        self.rtm_paras = {
            'cab': {"min": 4.44830458, "max": 229.4974769},  # ug/cm2
            'cw': {"min": 0.000048333333, "max": 0.08062008166},  # cm = 1000 mg/cm2
            'cm': {"min": 0.0000136666667, "max": 0.06638073277},  # g/cm2
            'LAI': {"min": 0.063333333, "max": 8.77},  # m2/m2
            'cp': {"min": 3.1320099999999997e-06, "max": 0.00420423377267},  # g/cm2
            'cbc': {"min": 0.001350791, "max": 0.046180855},  # g/cm2
            'car': {"min": 1.182576236, "max": 40.44321686},  # ug/cm2
            'anth': {"min": 0.561035015, "max": 2.981116889},  # ug/cm2
        }

        assert hidden_dim == len(
            self.rtm_paras), "hidden_dim must be equal to the number of RTM parameters"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Ensure the model is on the correct device

    def decode(self, para_dict):
        i_model = CallModel(soil=None, paras=para_dict)
        for key, value in i_model.par.items():
            i_model.par[key] = value.to(self.device)

        # Call the appropriate Prospect version
        if self.decoder.lop == "prospect4":
            i_model.call_prospect4()
        elif self.decoder.lop == "prospect5":
            i_model.call_prospect5()
        elif self.decoder.lop == "prospect5B":
            i_model.call_prospect5b()
        elif self.decoder.lop == "prospectD":
            i_model.call_prospectD()
        elif self.decoder.lop == "prospectPro":
            i_model.call_prospectPro()
        else:
            raise ValueError("Unknown Prospect version. Try 'prospect4', 'prospect5', 'prospect5B', 'prospectD', or 'ProspectPro'.")

        # Call the canopy architecture model
        if self.decoder.canopy_arch == "sail":
            result = i_model.call_4sail()
        elif self.decoder.canopy_arch == "inform":
            result = i_model.call_inform()
        else:
            result = i_model.prospect[:, :, 1]
        return result

    def adjust_dict_values(self, dictionary, desired_length):
        """
        Adjust the tensor arrays within a dictionary to the desired length by truncating or padding with the last number.
        """
        updated_dict = {}
        for key, value in dictionary.items():
            # Get the last number in the tensor array
            last_number = value[-1].item()  # Extract the last number from the tensor
            # Pad the tensor array with the last number to the desired length
            padded_value = torch.nn.functional.pad(value, (0, desired_length - len(value)), value=last_number)
            # Truncate if necessary
            updated_value = padded_value[:desired_length]
            # Update the dictionary with the modified tensor array
            updated_dict[key] = updated_value
    
        return updated_dict

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        encoded = self.encoder(x)  # Encoded parameters

        if self.scaler_list is not None:
            scaling_layer = scaler_layer(self.scaler_list) 
            transformation_layer_inv = StaticTransformationLayer(transformation=scaling_layer.inverse).to(self.device)
            transformed_tensor = transformation_layer_inv(encoded)

            # Constraint: Ensure cp = cm - cbc
            transformed_tensor[:, 5] = transformed_tensor[:, 2] - transformed_tensor[:, 4]
            para_tensor = transformed_tensor
        else:
            para_tensor = encoded.clone()
            para_tensor[:, 5] = para_tensor[:, 2] - para_tensor[:, 4]

        # Create parameter dictionary
        para_dict = {para_name: para_tensor[:, i] for i, para_name in enumerate(self.rtm_paras.keys())}

        # Reset decoder parameters
        self.decoder.para_reset(**para_dict)

        # Adjust dictionary values if necessary
        updated_dict = self.adjust_dict_values(self.decoder.para_dict, batch_size)

        # Decode to get the output spectrum
        out = self.decode(updated_dict)

        return encoded, out

class AE_RTM_corr(AE_RTM):
    """
    AutoEncoder with RTM as the decoder and additional layers for correction
    input -> encoder (learnable) -> decoder (INFORM) -> correction -> output
    """

    def __init__(self, input_dim, hidden_dim, rtm_paras=None, lop='prospectPro', canopy_arch='sail', scaler_list=None):
        super().__init__(input_dim, hidden_dim, rtm_paras, lop, canopy_arch, scaler_list)
        self.spectrum_length = 2101  # Assuming the spectrum length is 2101
        self.correction = nn.Sequential(
            nn.Linear(self.spectrum_length, 4 * self.spectrum_length),
            nn.ReLU(),
            nn.Linear(4 * self.spectrum_length, self.spectrum_length),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoded, decoded_spectrum = super().forward(x)
        corrected_spectrum = self.correction(decoded_spectrum)
        return encoded, corrected_spectrum
