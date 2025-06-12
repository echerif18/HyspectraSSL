# -*- coding: utf-8 -*-
"""
***************************************************************************
    call_model.py
    -----------------------------------------------------------------------
    begin                : 09/2020
    copyright            : (C) 2020 Martin Danner; Matthias Wocher
    email                : m.wocher@lmu.de

***************************************************************************
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
                                                                                                                                                 *
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software. If not, see <http://www.gnu.org/licenses/>.
***************************************************************************
    call_model.py routine handles the execution of PROSAIL in different forms
    This version is vectorized, i.e. the idea is to pass several parameter inputs at once to obtain an array of results
    But it is also used for single outputs

"""

import os
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm
import rtm_torch.Resources.PROSAIL.SAIL as SAIL_v
import rtm_torch.Resources.PROSAIL.INFORM as INFORM_v
import rtm_torch.Resources.PROSAIL.prospect as prospect_v
from rtm_torch.Resources.PROSAIL.dataSpec import lambd
from rtm_torch.Resources.Spec2Sensor.Spec2Sensor_core import Spec2Sensor
import warnings
import time
import math

# do not show warnings (set to 'all' if you want to see warnings, too)
warnings.filterwarnings('ignore')

# This class creates instances of the actual models and is fed with parameter inputs


class CallModel(nn.Module):

    def __init__(self, soil, paras):
        super(CallModel, self).__init__()
        # paras is a dictionary of all input parameters; this allows flexible adding/removing for new models
        self.par = paras
        # cab is always part of self.par, so it is used to obtain ninputs
        self.ninputs = self.par['cab'].shape[0]
        self.soil = soil
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def call_prospect4(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_4(
            self.par["N"], self.par["cab"], self.par["cw"], self.par["cm"]).clone().requires_grad_(True)  # ✅ Preserve gradients

        return self.prospect

    def call_prospect5(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_5(
            self.par["N"], self.par["cab"], self.par["car"], self.par["cw"], self.par["cm"]).clone().requires_grad_(True)  # ✅ Preserve gradients

        return self.prospect

    def call_prospect5b(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_5B(
            self.par["N"], self.par["cab"], self.par["car"], self.par["cbrown"], self.par["cw"], self.par["cm"]).clone().requires_grad_(True)  # ✅ Preserve gradients

        return self.prospect

    def call_prospectD(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_D(
            self.par["N"], self.par["cab"], self.par["car"], self.par["anth"], self.par["cbrown"], self.par["cw"], self.par["cm"]).clone().requires_grad_(True)  # ✅ Preserve gradients
        return self.prospect

    def call_prospectPro(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_Pro(
            self.par["N"], self.par["cab"], self.par["car"], self.par["anth"], self.par["cp"], self.par["cbc"], self.par["cbrown"], self.par["cw"]).clone().requires_grad_(True)  # ✅ Preserve gradients
        return self.prospect

    def call_4sail(self):
        try:
            self.prospect.detach().any()  # 4sail can only be called when PROSAIL has been run first
        except ValueError:
            raise ValueError(
                "A leaf optical properties model needs to be run first!")

        # Convert degrees to radians using PyTorch
        tts_rad = self.par["tts"] * (math.pi / 180)
        tto_rad = self.par["tto"] * (math.pi / 180)
        psi_rad = self.par["psi"] * (math.pi / 180)

        # Create Instance of SAIL and initialize angles
        sail_instance = SAIL_v.Sail(tts_rad, tto_rad, psi_rad)

        self.sail = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2], self.par["LIDF"],
                                           self.par["typeLIDF"], self.par["LAI"], self.par["hspot"], self.par["psoil"],
                                           self.soil).clone().requires_grad_(True)  # ✅ Preserve gradients    # call 4SAIL from the SAIL instance

        return self.sail

    def call_inform(self):
        try:
            self.prospect.any()
        except ValueError:
            raise ValueError(
                "A leaf optical properties model needs to be run first!")

        # Convert degrees to radians
        tts_rad = self.par["tts"] * (math.pi / 180)
        tto_rad = self.par["tto"] * (math.pi / 180)
        psi_rad = self.par["psi"] * (math.pi / 180)

        sail_instance = SAIL_v.Sail(tts_rad, tto_rad, psi_rad)

        # Step 1: call Pro4sail to calculate understory reflectance
        self.sail_understory_refl = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2],
                                                           self.par["LIDF"], self.par["typeLIDF"],
                                                           self.par["LAIu"], self.par["hspot"],
                                                           self.par["psoil"], self.soil)

        # Step 2: call Pro4sail with understory as soil to calculate infinite crown reflectance
        # vectorized: intialize extreme LAI ninputs times
        inform_temp_LAI = torch.full(
            (self.ninputs,), 15, dtype=torch.float32).T.to(self.device)
        # vectorized: initialize hspot = 0 ninputs times
        inform_temp_hspot = torch.full(
            (self.ninputs,), 0, dtype=torch.float32).T.to(self.device)

        self.sail_inf_refl = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2],
                                                    self.par["LIDF"], self.par["typeLIDF"],
                                                    inform_temp_LAI, self.par["hspot"],
                                                    psoil=None, soil=None, understory=self.sail_understory_refl)

        self.sail_tts_trans = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2],
                                                     self.par["LIDF"], self.par["typeLIDF"],
                                                     self.par["LAI"], inform_temp_hspot,
                                                     psoil=None, soil=None, understory=self.sail_understory_refl,
                                                     inform_trans='tts')

        self.sail_tto_trans = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2],
                                                     self.par["LIDF"], self.par["typeLIDF"],
                                                     self.par["LAI"], inform_temp_hspot,
                                                     psoil=None, soil=None, understory=self.sail_understory_refl,
                                                     inform_trans='tto')

        inform_instance = INFORM_v.INFORM(
            sail_instance.costts, sail_instance.costto, sail_instance.cospsi)

        inform = inform_instance.inform(self.par["cd"], self.par["sd"], self.par["h"],
                                        self.sail_understory_refl, self.sail_inf_refl,
                                        self.sail_tts_trans, self.sail_tto_trans).clone().requires_grad_(True)  # ✅ Preserve gradients

        return inform

# The class "Init_Model" initializes the models


class InitModel:

    # __init__ contains default values, but it is recommended to provide actual values for it
    def __init__(self, lop="prospectD", canopy_arch=None, int_boost=1, nodat=-999, s2s="default"):
        # self._dir = os.path.dirname(os.path.realpath(
        #     __file__))  # get current directory
        # os.chdir(self._dir)  # change into current directory
        self.lop = lop
        self.canopy_arch = canopy_arch
        # boost [0...1] of PROSAIL, e.g. int_boos = 10000 -> [0...10000] (EnMAP-default!)
        self.int_boost = int_boost
        self.nodat = nodat
        # which sensor type? Default = Prosail out; "EnMAP", "Sentinel2", "Landsat8" etc.
        self.s2s = s2s
        # "sort" (LUT contains multiple geos) vs. "no geo" (LUT contains ONE Geo)
        self.geo_mode = None
        self.soil = None  # initialize empty

        # List of names of all parameters in order in which they are written into the LUT; serves as labels for output
        self.para_names = ["N", "cab", "car", "anth", "cbrown", "cw", "cm", "cp", "cbc",
                           "LAI", "typeLIDF", "LIDF", "hspot", "psoil", "tts", "tto",
                           "psi", "LAIu", "cd", "sd", "h"]

        # Initialize the spectrum to sensor conversion if a sensor is chosen
        if self.s2s != "default":
            self.s2s_I = Spec2Sensor(sensor=self.s2s, nodat=self.nodat)
            sensor_init_success = self.s2s_I.init_sensor()
            if not sensor_init_success:
                raise Exception(
                    "Could not convert spectra to sensor resolution!")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def initialize_multiple_simple(self, soil=None, **paras):
        # simple tests for vectorized versions
        self.soil = soil
        nparas = len(paras['LAI'])
        para_grid = torch.empty(
            (nparas, len(paras.keys())), dtype=torch.float32).to(self.device)
        for run in range(nparas):
            for ikey, key in enumerate(self.para_names):
                para_grid[run, ikey] = paras[key][run]

        return self.run_model(paras=dict(zip(self.para_names, para_grid.T)))

    def initialize_single(self, soil=None, **paras):
        # Initialize a single run of PROSAIL (simplification for building of para_grid)
        self.soil = soil
        # if self.s2s != "default":
        #     self.s2s_I = Spec2Sensor(sensor=self.s2s, nodat=self.nodat)
        #     sensor_init_success = self.s2s_I.init_sensor()
        #     if not sensor_init_success:
        #         raise Exception(
        #             "Could not convert spectra to sensor resolution!")

        # shape 1 for single run
        # TODO para_grid also needs to be put on device
        para_grid = torch.empty((1, len(paras.keys()))).to(self.device)
        for ikey, key in enumerate(self.para_names):
            para_grid[0, ikey] = paras[key]
        return self.run_model(paras=dict(zip(self.para_names, para_grid.T)))

    def run_model(self, paras):
        # Execution of PROSAIL
        # Create new instance of CallModel
        i_model = CallModel(soil=self.soil, paras=paras)

        # 1: Call one of the Prospect-Versions
        if self.lop == "prospect4":
            i_model.call_prospect4().to(device)
        elif self.lop == "prospect5":
            i_model.call_prospect5().to(device)
        elif self.lop == "prospect5B":
            i_model.call_prospect5b().to(device)
        elif self.lop == "prospectD":
            i_model.call_prospectD().to(device)
        elif self.lop == "prospectPro":
            i_model.call_prospectPro().to(device)
        else:
            print("Unknown Prospect version. Try 'prospect4', 'prospect5', 'prospect5B' or 'prospectD' or ProspectPro")
            return

        # 2: If chosen, call one of the SAIL-versions and multiply with self.int_boost
        if self.canopy_arch == "sail":
            result = i_model.call_4sail() * self.int_boost
        elif self.canopy_arch == "inform":
            result = i_model.call_inform() * self.int_boost
        else:
            result = i_model.prospect[:, :, 1] * self.int_boost

        if self.s2s == "default":
            return result
        else:
            # if a sensor is chosen, run the Spectral Response Function now
            return self.s2s_I.run_srf(result)
