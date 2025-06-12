# -*- coding: utf-8 -*-
"""
***************************************************************************
    Processor_Inversion_core.py - LMU Agri Apps - Artificial Neural Network based spectroscopic image inversion of
    PROSAIL parameters - GUI
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

This module handles the conversion between sensors, mostly in terms of spectral downsampling to the characteristics
of the "target sensor". It was originally designed to obtain pseudo-EnMAP spectra from ASD FieldSpec reflectances
-> 400-2500nm @ 1nm ==> 242 EnMAP bands; without altering the spatial resolution or performing any other radiometric
processing.

There are two classes: one performs the conversion when the spectral response function (SRF) is already there in
the form of numpy files with .srf extension. These .srf-files can only be read by THIS module, they are not
standardized in any form! They are just binary files containing the weights associated with wavelengths for all
bands of the target sensor.
The other class is designed to create these .srf-files from text-based information about weights and wavelengths
in a certain structure which was introduced by Karl Segl for dealing with the EnMAP-end-to-end-simulator (EeteS).

The general concept of a spectral response function is that each band of the target sensor carries information not
only about its central wavelength but also about adjacent wavelengths. For example, EnMAP band 17 @ 503 nm is made up
from wavelengths 492 nm with weight 0.0001, 493 nm with weight 0.0003, 494 nm with weight 0.0014, ... 502 nm with
weight 0.922, 503 nm with weight 1.0, 504 nm with weight 0.922, ... 514 nm with weight 0.0001. Each band of the target
sensor has its own text file which needs to have two columns: wavelengths and weights. Additionally, the central
wavelengths of the target sensors need to be known, as they cannot be extracted from the weights right away. It can
be single column (wavelengths) or two columns (wavelengths & FWHM).
"""

import torch
import numpy as np
import csv
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Execution of a conversion between two sensors


class Spec2Sensor:

    def __init__(self, nodat, sensor):
        self.wl_sensor, self.fwhm = (None, None)
        self.wl = torch.arange(400, 2501, device=device)
        self.n_wl = len(self.wl)
        self.nodat = nodat
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.sensor = sensor

    def init_sensor(self):
        # Initialize values for the sensor
        try:
            srf_file = np.load(self.path + "/srf/" + self.sensor + ".srf")
        except FileNotFoundError:
            print("File {} not found, please check directories!".format(
                self.sensor + ".srf"))
            return False

        # the srf file is a numpy .npz; this file type contains more than one array and are addressed by name like
        # dictionaries from the np.open object
        # Convert the numpy arrays to PyTorch tensors
        self.srf = torch.from_numpy(srf_file['srf']).to(device)
        self.srf_nbands = torch.from_numpy(srf_file['srf_nbands']).to(device)
        self.wl_sensor = torch.from_numpy(srf_file['sensor_wl']).to(device)
        self.n_wl_sensor = len(self.wl_sensor)
        self.ndvi = torch.from_numpy(srf_file['sensor_ndvi']).to(device)

        return True  # return True if everything worked

    def run_srf(self, reflectance, int_factor_wl=1000):
        # convert reflectances to new sensor; the function is vectorized and takes arrays of reflectances
        # dictionary to map wavelengths [nm] to wavebands
        hash_getwl = dict(zip(self.wl.cpu().numpy(), list(range(self.n_wl))))
        # prepare array for corrected refs
        spec_corr = torch.zeros(
            (reflectance.shape[0], self.n_wl_sensor), device=device)

        # for each band of the target sensor
        for sensor_band in range(self.n_wl_sensor):
            # empty the list of wfactors to include in the weighting function
            wfactor_include = list()
            # for each srf_value of the band
            for srf_i in range(self.srf_nbands[sensor_band]):
                try:
                    # obtain wavelength of for the band and srf_i value ([0]) and boost with the int_factor
                    wlambda = int(
                        self.srf[srf_i][sensor_band][0].item() * int_factor_wl)
                except ValueError:
                    print("Error with sensor band {:d} at the srf #{:d}".format(
                        sensor_band, srf_i))
                    return

                if wlambda not in self.wl.cpu().numpy():  # this happens when a wavelength is specified
                    continue                             # in the srf-file outside 400-2500

                # Do the same with the weighting factor for band and srf_i value ([1])
                wfactor = self.srf[srf_i][sensor_band][1]
                # add it to the list of wfactors for this sensor band
                wfactor_include.append(srf_i)
                # add to the spectral sums for this sensor band
                spec_corr[:, sensor_band] += reflectance[:,
                                                         hash_getwl[wlambda]] * wfactor

            # get the total sum for the sensor band
            sum_wfactor = sum(self.srf[i][sensor_band][1].item()
                              for i in wfactor_include)

            # try:
            #     # divide by the sum to get a weighted average
            #     spec_corr[:, sensor_band] /= sum_wfactor
            # except ZeroDivisionError:  # this happens when no srf-value can be extracted from the original data
            #     spec_corr[:, sensor_band] = self.nodat

            if sum_wfactor != 0:
                # divide by the sum to get a weighted average
                spec_corr[:, sensor_band] /= sum_wfactor
            else:
                # this happens when no srf-value can be extracted from the original data
                spec_corr[:, sensor_band] = torch.full_like(
                    spec_corr[:, sensor_band], self.nodat)

        return spec_corr
